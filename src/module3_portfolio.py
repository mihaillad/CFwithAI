#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 3. Оценка портфеля и рекомендации.

Автономный запуск:
    python src/module3_portfolio.py
    python -m src.module3_portfolio --date 2026-10-22
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# ============================================================================
# Константы
# ============================================================================
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")

MAX_AGE_DEALS_SEC = 3600  # 1 час
DEALS_URL = "http://box:78/money/hs/reports/deals"

RANKING_FILES = [
    "stock_ranking.csv",
    "ofz_ranking.csv",
    "corp_ranking.csv",
]


# ============================================================================
# Утилиты
# ============================================================================
def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{date}.log"
    logger = logging.getLogger("module3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_config(config_path: Path) -> dict:
    defaults = {
        "date": "today",
        "stock_sleeve_pct": 65.0,
        "bond_sleeve_pct": 35.0,
        "cash_buffer_pct": 2.0,
        "min_deficit_pct": 0.5,
        "profit_take_pct": 5.0,
        "max_hold_years": 3,
        "top_n_freeze": 8,
        "frozen_limit_pct": 10.0,
        "cooldown_days": 30,
        "flow_limit_pct_week": 1.0,
        "lot_source": "universe",
    }
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(cfg)
    return defaults


def save_csv(df: pd.DataFrame, path: Path, sep: str = ";", float_format: str = "%.4f"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8",
              float_format=float_format, date_format="%Y-%m-%d")


def read_csv_flex(path: Path) -> pd.DataFrame:
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep=";")


def file_age(path: Path) -> float:
    return time.time() - path.stat().st_mtime if path.exists() else float('inf')


def is_fresh(path: Path, max_age: float) -> bool:
    return file_age(path) <= max_age


# ============================================================================
# Загрузка сделок
# ============================================================================
def fetch_and_save_deals(data_dir: Path, logger: logging.Logger) -> None:
    deals_path = data_dir / "deals.csv"
    if is_fresh(deals_path, MAX_AGE_DEALS_SEC):
        logger.info("Файл deals.csv актуален (возраст < 1 часа).")
        return

    logger.info("Обновление файла deals.csv с внешнего источника...")
    try:
        response = requests.get(DEALS_URL, timeout=10)
        response.raise_for_status()
        deals_path.parent.mkdir(parents=True, exist_ok=True)
        deals_path.write_text(response.text, encoding='utf-8')
        logger.info(f"Файл deals.csv успешно обновлён: {deals_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при загрузке deals.csv: {e}")
        if not deals_path.exists():
            raise RuntimeError("Не удалось загрузить deals.csv, и локальный файл отсутствует.")
        logger.warning("Используется устаревший локальный файл deals.csv.")


def deals_to_holdings(deals_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Собирает текущие позиции из файла сделок.
    Себестоимость считается по FIFO (как в dividend_rating.py).
    """
    deals = read_csv_flex(deals_path)
    if deals.empty:
        logger.warning("deals.csv пуст")
        return pd.DataFrame()

    deals = deals.dropna(subset=["SECID"])
    deals.columns = [c.upper() for c in deals.columns]
    trades = deals[deals["TYPE"].isin(["BUY", "SELL"])].copy()
    if trades.empty:
        logger.warning("В deals.csv нет операций BUY/SELL")
        return pd.DataFrame()

    trades["QTY_SIGNED"] = trades.apply(
        lambda r: r["COUNT"] if r["TYPE"] == "BUY" else -r["COUNT"], axis=1
    )
    inventory = trades.groupby("SECID")["QTY_SIGNED"].sum().reset_index()
    inventory = inventory[inventory["QTY_SIGNED"] != 0]
    if inventory.empty:
        logger.warning("После сверки BUY/SELL позиций не осталось")
        return pd.DataFrame()

    rows = []
    for _, inv_row in inventory.iterrows():
        secid = inv_row["SECID"]
        current_qty = int(inv_row["QTY_SIGNED"])
        buys = trades[(trades["SECID"] == secid) & (trades["TYPE"] == "BUY")].copy()
        if buys.empty:
            continue

        # FIFO: от новых покупок к старым — "снимаем" с хвоста, пока не наберём current_qty
        buys = buys.sort_values("DATE", ascending=False).reset_index(drop=True)
        buys["CUMSUM"] = buys["COUNT"].cumsum()
        last_idx = buys[buys["CUMSUM"] >= abs(current_qty)].index.max()
        if pd.isna(last_idx):
            # Покупок меньше, чем текущая позиция — берём все
            relevant_buys = buys.copy()
        else:
            relevant_buys = buys.loc[:last_idx].copy()
            total_in_relevant = relevant_buys["COUNT"].sum()
            if total_in_relevant > abs(current_qty):
                last_buy = relevant_buys.iloc[-1].copy()
                excess = total_in_relevant - abs(current_qty)
                portion = (last_buy["COUNT"] - excess) / last_buy["COUNT"]
                relevant_buys.iloc[-1, relevant_buys.columns.get_loc("COUNT")] = last_buy["COUNT"] - excess
                relevant_buys.iloc[-1, relevant_buys.columns.get_loc("SUM")] = last_buy["SUM"] * portion

        acq_value = relevant_buys["SUM"].sum()
        avg_price = acq_value / abs(current_qty) if current_qty != 0 else 0.0
        first_date = buys["DATE"].iloc[-1]  # самая старая из отобранных
        last_date = buys["DATE"].iloc[0]    # самая новая

        rows.append({
            "asset_type": "stock",
            "SECID_or_ISIN": secid,
            "quantity": abs(current_qty),
            "avg_price": avg_price,
            "first_purchase_date": first_date,
            "last_purchase_date": last_date,
        })

    holdings = pd.DataFrame(rows)
    logger.info(f"Собрано позиций из deals.csv (FIFO): {len(holdings)}")
    return holdings


# ============================================================================
# Проверка входных файлов
# ============================================================================
def check_required_files(data_dir: Path, logger: logging.Logger) -> tuple[Path, bool]:
    holdings_path = data_dir / "portfolio_holdings.csv"
    deals_path = data_dir / "deals.csv"

    if holdings_path.exists():
        logger.info(f"Используется существующий {holdings_path.name}")
        # Параллельно гарантируем cash_balance.csv
        cash_path = data_dir / "cash_balance.csv"
        if not cash_path.exists():
            empty_cash = pd.DataFrame(columns=["date", "currency", "amount"])
            save_csv(empty_cash, cash_path)
            logger.info(f"Создан пустой {cash_path.name}")
        return holdings_path, False

    if deals_path.exists():
        logger.info(f"{holdings_path.name} не найден, собираю позиции из {deals_path.name}...")
        holdings = deals_to_holdings(deals_path, logger)
        if not holdings.empty:
            save_csv(holdings, holdings_path)
            logger.info(f"Создан {holdings_path.name}: {len(holdings)} позиций")
        else:
            logger.warning("Не удалось собрать позиции из deals.csv")
            empty = pd.DataFrame(columns=[
                "asset_type", "SECID_or_ISIN", "quantity",
                "avg_price", "first_purchase_date", "last_purchase_date"
            ])
            save_csv(empty, holdings_path)
    else:
        logger.warning("portfolio_holdings.csv и deals.csv отсутствуют. Создаю пустой шаблон.")
        empty = pd.DataFrame(columns=[
            "asset_type", "SECID_or_ISIN", "quantity",
            "avg_price", "first_purchase_date", "last_purchase_date"
        ])
        save_csv(empty, holdings_path)

    cash_path = data_dir / "cash_balance.csv"
    if not cash_path.exists():
        empty_cash = pd.DataFrame(columns=["date", "currency", "amount"])
        save_csv(empty_cash, cash_path)
        logger.info(f"Создан пустой {cash_path.name}")

    return holdings_path, True


def check_ranking_files(output_dir: Path, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    result = {}
    for name in RANKING_FILES:
        path = output_dir / name
        if path.exists():
            result[name] = read_csv_flex(path)
            logger.info(f"Загружен {name}: {len(result[name])} строк")
        else:
            logger.warning(f"{name} не найден в {output_dir} — будет пустой рейтинг")
            result[name] = pd.DataFrame()
    return result


# ============================================================================
# Детальная таблица портфеля
# ============================================================================
def build_portfolio_details(date: str, holdings: pd.DataFrame,
                            latest_stock_prices: pd.DataFrame,
                            latest_bond_prices: pd.DataFrame,
                            stock_ranking: pd.DataFrame,
                            ofz_ranking: pd.DataFrame,
                            corp_ranking: pd.DataFrame,
                            total_value: float,
                            date_dt: pd.Timestamp) -> pd.DataFrame:
    """Формирует детальную таблицу портфеля для HTML-отчёта."""
    if holdings.empty:
        return pd.DataFrame()

    # Быстрые словари для поиска целевых весов и DPS
    stock_target = {}
    stock_dps = {}
    if not stock_ranking.empty and "SECID" in stock_ranking.columns:
        for _, r in stock_ranking.iterrows():
            stock_target[r["SECID"]] = float(r.get("target_weight_portfolio_pct", 0) or 0)
            dps_val = r.get("dps_reg_next", 0)
            stock_dps[r["SECID"]] = float(dps_val) if pd.notna(dps_val) else 0.0

    bond_target = {}
    for r_df in (ofz_ranking, corp_ranking):
        if not r_df.empty and "ISIN" in r_df.columns:
            for _, r in r_df.iterrows():
                bond_target[r["ISIN"]] = float(r.get("target_weight_portfolio_pct", 0) or 0)

    rows = []
    for _, h in holdings.iterrows():
        asset_type = h["asset_type"]
        secid_or_isin = h["SECID_or_ISIN"]
        qty = float(h["quantity"])
        avg_price = float(h["avg_price"]) if pd.notna(h.get("avg_price")) else 0.0
        first_purchase = h.get("first_purchase_date")

        current_price = 0.0
        current_value = 0.0
        target_w = 0.0
        dps_next = 0.0
        div_sum = 0.0
        div_pct = 0.0

        if asset_type == "stock":
            if secid_or_isin not in latest_stock_prices.index:
                continue
            current_price = float(latest_stock_prices.loc[secid_or_isin, "close"])
            current_value = qty * current_price
            target_w = stock_target.get(secid_or_isin, 0.0)
            dps_next = stock_dps.get(secid_or_isin, 0.0)
            div_sum = dps_next * qty
            div_pct = (div_sum / current_value * 100) if current_value > 0 else 0.0

        elif asset_type == "bond":
            if secid_or_isin not in latest_bond_prices.index:
                continue
            p_row = latest_bond_prices.loc[secid_or_isin]
            nominal = 1000.0
            # Безопасное получение НКД (колонка может отсутствовать)
            nkd = float(p_row.get("nkd_rub", 0.0) or 0.0)
            price_pct = float(p_row.get("price_pct", 0.0) or 0.0)
            current_price = (price_pct / 100 * nominal) + nkd
            current_value = qty * current_price
            target_w = bond_target.get(secid_or_isin, 0.0)
        else:
            continue

        cost_basis = qty * avg_price
        target_value = total_value * target_w / 100
        deviation = current_value - target_value
        current_weight = (current_value / total_value * 100) if total_value > 0 else 0.0
        accounting_profit = current_value - cost_basis
        accounting_profit_pct = ((current_value / cost_basis) - 1) * 100 if cost_basis > 0 else 0.0

        years_held = 0.0
        if pd.notna(first_purchase) and str(first_purchase).strip():
            try:
                fp = pd.to_datetime(first_purchase)
                years_held = round((date_dt - fp).days / 365.25, 2)
            except Exception:
                years_held = 0.0

        rows.append({
            "date": date, "asset_type": asset_type,
            "SECID_or_ISIN": secid_or_isin, "quantity": qty,
            "current_price": round(current_price, 4),
            "current_value": round(current_value, 2),
            "target_weight_pct": round(target_w, 4),
            "target_value": round(target_value, 2),
            "deviation_value": round(deviation, 2),
            "current_weight_pct": round(current_weight, 4),
            "cost_basis": round(cost_basis, 2),
            "accounting_profit": round(accounting_profit, 2),
            "accounting_profit_pct": round(accounting_profit_pct, 4),
            "years_held": years_held,
            "dividend_sum": round(div_sum, 2),
            "dividend_pct": round(div_pct, 4),
            "first_purchase_date": first_purchase,
        })

    return pd.DataFrame(rows)


# ============================================================================
# Основная функция модуля
# ============================================================================
def evaluate_portfolio(date: str, config: dict, data_dir: Path, output_dir: Path,
                       logger: logging.Logger, diagnostics: list
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Оценка портфеля, расчёт дефицитов, генерация рекомендаций и аутсайдеров.
    Возвращает: (snapshot, recommendations, outsiders, portfolio_details)
    """
    date_dt = pd.to_datetime(date)

    # --- Загрузка рейтингов ---
    rankings = check_ranking_files(output_dir, logger)
    stock_ranking = rankings["stock_ranking.csv"]
    ofz_ranking = rankings["ofz_ranking.csv"]
    corp_ranking = rankings["corp_ranking.csv"]

    # --- Загрузка holdings ---
    holdings_path = data_dir / "portfolio_holdings.csv"
    holdings = read_csv_flex(holdings_path) if holdings_path.exists() else pd.DataFrame()

    # --- Загрузка cash ---
    cash_path = data_dir / "cash_balance.csv"
    cash_value = 0.0
    if cash_path.exists():
        cash = read_csv_flex(cash_path)
        if "currency" in cash.columns and "amount" in cash.columns:
            cash_value = float(cash[cash["currency"] == "RUB"]["amount"].sum())
    else:
        logger.warning("cash_balance.csv не найден, cash_value = 0")

    # --- Загрузка цен ---
    latest_stock_prices = pd.DataFrame()
    prices_stocks_path = data_dir / "stock_prices.csv"
    if prices_stocks_path.exists():
        p = read_csv_flex(prices_stocks_path)
        if "date" in p.columns and "SECID" in p.columns and "close" in p.columns:
            p["date"] = pd.to_datetime(p["date"])
            latest_stock_prices = (p[p["date"] <= date].sort_values("date")
                                    .groupby("SECID").tail(1).set_index("SECID"))

    latest_bond_prices = pd.DataFrame()
    prices_bonds_path = data_dir / "bond_prices.csv"
    if prices_bonds_path.exists():
        p = read_csv_flex(prices_bonds_path)
        if "date" in p.columns and "ISIN" in p.columns:
            p["date"] = pd.to_datetime(p["date"])
            latest_bond_prices = (p[p["date"] <= date].sort_values("date")
                                   .groupby("ISIN").tail(1).set_index("ISIN"))

    # --- Загрузка universe для лотности ---
    universe_stocks = pd.DataFrame()
    if (data_dir / "stocks_universe.csv").exists():
        universe_stocks = read_csv_flex(data_dir / "stocks_universe.csv")

    # --- Оценка стоимости ---
    stock_val = 0.0
    bond_val = 0.0
    bond_val_clean = 0.0
    recommendations = []
    outsiders = []

    # Быстрый словарь лотности
    lot_map = {}
    if not universe_stocks.empty and "SECID" in universe_stocks.columns and "lot_size" in universe_stocks.columns:
        lot_map = dict(zip(universe_stocks["SECID"],
                           pd.to_numeric(universe_stocks["lot_size"], errors="coerce").fillna(1).astype(int)))

    for _, h in holdings.iterrows():
        asset_type = h["asset_type"]
        secid_or_isin = h["SECID_or_ISIN"]
        qty = float(h["quantity"])
        avg_price = float(h["avg_price"]) if pd.notna(h.get("avg_price")) else 0.0

        if asset_type == "stock":
            if secid_or_isin not in latest_stock_prices.index:
                continue
            close = float(latest_stock_prices.loc[secid_or_isin, "close"])
            val = qty * close
            stock_val += val

            holding_days = 0
            if pd.notna(h.get("first_purchase_date")):
                try:
                    holding_days = (date_dt - pd.to_datetime(h["first_purchase_date"])).days
                except Exception:
                    holding_days = 0

            pnl_pct = ((close - avg_price) / avg_price * 100) if avg_price > 0 else np.nan

            stock_info = stock_ranking[stock_ranking["SECID"] == secid_or_isin] if not stock_ranking.empty else pd.DataFrame()
            rank = int(stock_info["rank"].iloc[0]) if not stock_info.empty and "rank" in stock_info.columns else 999
            stop_flag = bool(stock_info["stop_flag"].iloc[0]) if not stock_info.empty and "stop_flag" in stock_info.columns else False
            in_top_freeze = rank <= config["top_n_freeze"]

            action = "HOLD"
            reason = "no_action"

            if stop_flag:
                action = "SELL"
                reason = "stop_list"
            elif pd.notna(pnl_pct) and pnl_pct >= config["profit_take_pct"]:
                action = "SELL"
                reason = "profit_take"
            elif holding_days >= 365 * config["max_hold_years"] and pd.notna(pnl_pct) and pnl_pct < 0:
                if in_top_freeze:
                    action = "FREEZE"
                    reason = "three_year_top_freeze"
                else:
                    action = "SELL"
                    reason = "three_year_exit"

            outsiders.append({
                "date": date, "SECID": secid_or_isin,
                "first_purchase_date": h.get("first_purchase_date", ""),
                "holding_days": holding_days, "avg_price": avg_price,
                "current_price": close, "pnl_pct": pnl_pct,
                "in_top_freeze_ay": in_top_freeze,
                "top_n_freeze_used": config["top_n_freeze"],
                "stop_flag": stop_flag, "action": action, "reason": reason,
            })

        elif asset_type == "bond":
            if secid_or_isin not in latest_bond_prices.index:
                continue
            p_row = latest_bond_prices.loc[secid_or_isin]
            nominal = 1000.0
            nkd = float(p_row.get("nkd_rub", 0.0) or 0.0)
            if pd.isna(p_row.get("nkd_rub")):
                diagnostics.append({"code": "nkd_missing", "details": f"ISIN {secid_or_isin}"})
            price_pct = float(p_row.get("price_pct", 0.0) or 0.0)
            val = qty * ((price_pct / 100 * nominal) + nkd)
            clean_val = qty * (price_pct / 100 * nominal)
            bond_val += val
            bond_val_clean += clean_val

    total_value = stock_val + bond_val + cash_value

    # --- Формирование snapshot (всегда!) ---
    if total_value == 0:
        logger.warning("total_value == 0 — портфель пуст")
        snapshot = pd.DataFrame([{
            "date": date, "total_value": 0, "stock_value": 0, "bond_value": 0,
            "bond_value_clean": 0, "cash_value": cash_value,
            "stock_share_pct": 0, "bond_share_pct": 0, "cash_share_pct": 100,
            "target_stock_pct": config["stock_sleeve_pct"],
            "target_bond_pct": config["bond_sleeve_pct"],
            "deviation_stock_pct": -config["stock_sleeve_pct"],
            "deviation_bond_pct": -config["bond_sleeve_pct"],
            "cash_buffer_excess_pct": 100 - config["cash_buffer_pct"],
        }])
        portfolio_details = pd.DataFrame()
        return snapshot, pd.DataFrame(recommendations), pd.DataFrame(outsiders), portfolio_details

    snapshot = pd.DataFrame([{
        "date": date,
        "total_value": total_value,
        "stock_value": stock_val,
        "bond_value": bond_val,
        "bond_value_clean": bond_val_clean,
        "cash_value": cash_value,
        "stock_share_pct": stock_val / total_value * 100,
        "bond_share_pct": bond_val / total_value * 100,
        "cash_share_pct": cash_value / total_value * 100,
        "target_stock_pct": config["stock_sleeve_pct"],
        "target_bond_pct": config["bond_sleeve_pct"],
        "deviation_stock_pct": stock_val / total_value * 100 - config["stock_sleeve_pct"],
        "deviation_bond_pct": bond_val / total_value * 100 - config["bond_sleeve_pct"],
        "cash_buffer_excess_pct": cash_value / total_value * 100 - config["cash_buffer_pct"],
    }])

    # --- Рекомендации по докупке акций (по дефициту веса) ---
    if not stock_ranking.empty and "eligible_for_buy" in stock_ranking.columns:
        eligible = stock_ranking[stock_ranking["eligible_for_buy"] == True].copy()

        # Текущие веса по акциям
        current_weights = {}
        if not holdings.empty:
            stock_holdings = holdings[holdings["asset_type"] == "stock"]
            for _, h_row in stock_holdings.iterrows():
                sec = h_row["SECID_or_ISIN"]
                if sec in latest_stock_prices.index:
                    cur_val = float(h_row["quantity"]) * float(latest_stock_prices.loc[sec, "close"])
                    current_weights[sec] = current_weights.get(sec, 0.0) + cur_val

        for _, row in eligible.iterrows():
            secid = row["SECID"]
            target_w = float(row.get("target_weight_portfolio_pct", 0) or 0)
            current_val = current_weights.get(secid, 0.0)
            current_w = current_val / total_value * 100
            deficit_pct = target_w - current_w

            if deficit_pct > config["min_deficit_pct"]:
                price = float(row["price"])
                lot_size = lot_map.get(secid, 1)
                lot_source = "universe" if secid in lot_map else "assumed"

                deficit_rub = deficit_pct / 100 * total_value
                lots_delta = math.floor(deficit_rub / (price * lot_size)) if (price * lot_size) > 0 else 0
                qty_delta = lots_delta * lot_size

                action = "BUY" if qty_delta > 0 else "NO_ACTION"
                reason_code = "deficit" if action == "BUY" else "lot_too_small"

                recommendations.append({
                    "date": date, "asset_type": "stock",
                    "SECID_or_ISIN": secid,
                    "action": action, "reason_code": reason_code,
                    "reason_text": f"Deficit {deficit_pct:.2f}%",
                    "current_qty": current_val / price if price > 0 else 0,
                    "current_weight_portfolio_pct": current_w,
                    "target_weight_portfolio_pct": target_w,
                    "deficit_pct": deficit_pct,
                    "qty_delta": qty_delta, "qty_delta_lots": lots_delta,
                    "lot_size": lot_size, "lot_source": lot_source,
                    "amount_rub": qty_delta * price,
                    "priority": 4, "constraints_hit": "",
                })

    # --- Детальная таблица портфеля ---
    portfolio_details = build_portfolio_details(
        date, holdings, latest_stock_prices, latest_bond_prices,
        stock_ranking, ofz_ranking, corp_ranking, total_value, date_dt
    )

    return snapshot, pd.DataFrame(recommendations), pd.DataFrame(outsiders), portfolio_details


# ============================================================================
# Точка входа
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Module 3: оценка портфеля (изолированный запуск)")
    ap.add_argument("--date", default="", help="Дата расчёта YYYY-MM-DD")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Путь к config.yaml")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Папка с входными CSV")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Папка с выходами")
    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    resolve = lambda p: Path(p) if Path(p).is_absolute() else project_root / p
    config_path = resolve(args.config)
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    date_from_config = config.get("date", "today")
    if not date_from_config or str(date_from_config).lower() in ("today", "auto", ""):
        date = args.date or datetime.now().strftime("%Y-%m-%d")
    else:
        date = args.date or str(date_from_config)

    logger = setup_logging(date, output_dir)
    logger.info(f"=== Module 3 start: date={date}, data_dir={data_dir}, output_dir={output_dir} ===")

    diagnostics: list[dict] = []

    try:
        # Обновление сделок
        fetch_and_save_deals(data_dir, logger)

        # Проверка входных файлов
        holdings_path, created = check_required_files(data_dir, logger)
        if created:
            diagnostics.append({"code": "holdings_autogenerated", "details": str(holdings_path)})

        # Выполнение оценки
        snapshot, rebalance, outsiders, portfolio_details = evaluate_portfolio(
            date, config, data_dir, output_dir, logger, diagnostics
        )

        # Сохранение результатов
        save_csv(snapshot, output_dir / "portfolio_snapshot.csv")
        save_csv(rebalance, output_dir / "rebalance_recommendations.csv")
        save_csv(outsiders, output_dir / "outsiders.csv")
        save_csv(portfolio_details, output_dir / "portfolio_details.csv")

        logger.info(f"portfolio_snapshot: {len(snapshot)} строк")
        logger.info(f"rebalance_recommendations: {len(rebalance)} строк")
        logger.info(f"outsiders: {len(outsiders)} строк")
        logger.info(f"portfolio_details: {len(portfolio_details)} строк")

        if not snapshot.empty:
            row = snapshot.iloc[0]
            logger.info(f"Total value: {row['total_value']:,.2f} RUB".replace(",", " "))
            logger.info(f"Stock: {row['stock_share_pct']:.2f}%, Bond: {row['bond_share_pct']:.2f}%, Cash: {row['cash_share_pct']:.2f}%")

        # diagnostics
        if diagnostics:
            diag_path = output_dir / "diagnostics.csv"
            existing = pd.DataFrame()
            if diag_path.exists():
                try:
                    existing = read_csv_flex(diag_path)
                except Exception:
                    existing = pd.DataFrame()
            merged = pd.concat([existing, pd.DataFrame(diagnostics)], ignore_index=True)
            save_csv(merged, diag_path)
            logger.info(f"diagnostics: всего {len(merged)} записей")

        logger.info("=== Module 3 finished successfully ===")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Не найден входной файл: {e}")
        return 2
    except RuntimeError as e:
        logger.error(str(e))
        return 2
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())