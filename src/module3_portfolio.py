#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 3. Оценка портфеля и рекомендации.

Автономный запуск (для отладки):
    python -m src.module3_portfolio
    # или
    python src/module3_portfolio.py

При запуске как __main__ автоматически:
  - создаёт data/raw/ и data/output/
  - читает stock_ranking.csv, ofz_ranking.csv, corp_ranking.csv из data/output/
  - читает portfolio_holdings.csv + cash_balance.csv из data/raw/
    (или собирает holdings из deals.csv, если holdings нет)
  - выполняет evaluate_portfolio() и сохраняет:
      portfolio_snapshot.csv, rebalance_recommendations.csv, outsiders.csv
"""
from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import time
import requests
from pathlib import Path
import logging

# ============================================================================
# Константы
# ============================================================================
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")
MAX_AGE_DEALS_SEC = 3600  # 1 час
DEALS_URL = "http://box:78/money/hs/reports/deals"

# Критичные файлы (без них модуль не может работать)
CRITICAL_RAW_FILES = [
    "portfolio_holdings.csv",   # или deals.csv — будет авто-конвертация
]

# Файлы-результаты модулей 1 и 2 (должны быть в output_dir)
RANKING_FILES = [
    "stock_ranking.csv",
    "ofz_ranking.csv",
    "corp_ranking.csv",
]


# ============================================================================
# Встроенные утилиты
# ============================================================================

def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    """Логгер: файл output/run_<date>.log + stdout. Режим 'w' — идемпотентность."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{date}.log"
    logger = logging.getLogger("module3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")  # 'a' — дописываем к логам m1/m2
    fh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_config(config_path: Path) -> dict:
    """Загружает config.yaml. Если файла нет — возвращает дефолты из ТЗ."""
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
    """Сохраняет DataFrame в CSV с разделителем ; и заданной точностью float."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8",
              float_format=float_format, date_format="%Y-%m-%d")


def read_csv_flex(path: Path) -> pd.DataFrame:
    """Читает CSV с автоопределением разделителя (';' или ',')."""
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep=";")


# ============================================================================
# Конвертация deals.csv -> portfolio_holdings.csv
# ============================================================================

def deals_to_holdings(deals_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Собирает текущие позиции из файла сделок (как в dividend_rating.py).
    Возвращает DataFrame в формате portfolio_holdings.csv.
    """
    deals = read_csv_flex(deals_path)
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

    # FIFO-себестоимость (упрощённо — берём среднюю цену покупок)
    rows = []
    for secid, qty in zip(inventory["SECID"], inventory["QTY_SIGNED"]):
        buys = trades[(trades["SECID"] == secid) & (trades["TYPE"] == "BUY")].copy()
        if buys.empty:
            continue
        buys = buys.sort_values("DATE", ascending=True)
        total_cost = (buys["COUNT"] * buys["SUM"] / buys["COUNT"]).sum()  # SUM уже = стоимость
        # В deals.csv SUM — это полная сумма сделки, COUNT — количество
        total_cost = buys["SUM"].sum()
        total_qty_bought = buys["COUNT"].sum()
        avg_price = total_cost / total_qty_bought if total_qty_bought > 0 else 0.0
        first_date = buys["DATE"].iloc[0]
        last_date = buys["DATE"].iloc[-1]

        rows.append({
            "asset_type": "stock",
            "SECID_or_ISIN": secid,
            "quantity": abs(qty),
            "avg_price": avg_price,
            "first_purchase_date": first_date,
            "last_purchase_date": last_date,
        })

    holdings = pd.DataFrame(rows)
    logger.info(f"Собрано позиций из deals.csv: {len(holdings)}")
    return holdings

def file_age(path: Path) -> float:
    """Возвращает возраст файла в секундах. Если файла нет, возвращает бесконечность."""
    return time.time() - path.stat().st_mtime if path.exists() else float('inf')

def is_fresh(path: Path, max_age: float) -> bool:
    """Проверяет, что файл существует и его возраст не превышает max_age секунд."""
    return file_age(path) <= max_age

def fetch_and_save_deals(data_dir: Path, logger: logging.Logger) -> None:
    """
    Загружает файл сделок (deals.csv) с внешнего источника.
    Обновляет файл, если он отсутствует или устарел (> 1 часа).
    """
    deals_path = data_dir / "deals.csv"
    
    if is_fresh(deals_path, MAX_AGE_DEALS_SEC):
        logger.info("Файл deals.csv актуален (возраст < 1 часа).")
        return
    
    logger.info("Обновление файла deals.csv с внешнего источника...")
    try:
        response = requests.get(DEALS_URL, timeout=10)
        response.raise_for_status()
        
        deals_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохраняем файл в том виде, в котором его отдает сервер (обычно UTF-8)
        deals_path.write_text(response.text, encoding='utf-8')
        logger.info(f"Файл deals.csv успешно обновлен и сохранен в {deals_path}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при загрузке deals.csv: {e}")
        if not deals_path.exists():
            raise RuntimeError("Не удалось загрузить deals.csv, и локальный файл отсутствует.")
        logger.warning("Используется устаревший локальный файл deals.csv.")

# ============================================================================
# Проверка входных файлов
# ============================================================================

def check_required_files(data_dir: Path, logger: logging.Logger) -> tuple[Path, bool]:
    """
    Проверяет наличие portfolio_holdings.csv или deals.csv.
    Если holdings нет, но есть deals — конвертирует.
    Возвращает (путь к holdings, создан ли файл заново).
    """
    holdings_path = data_dir / "portfolio_holdings.csv"
    deals_path = data_dir / "deals.csv"

    if holdings_path.exists():
        logger.info(f"Используется существующий {holdings_path.name}")
        return holdings_path, False

    if deals_path.exists():
        logger.info(f"{holdings_path.name} не найден, собираю позиции из {deals_path.name}...")
        holdings = deals_to_holdings(deals_path, logger)
        if not holdings.empty:
            save_csv(holdings, holdings_path)
            logger.info(f"Создан {holdings_path.name}: {len(holdings)} позиций")
            return holdings_path, True
        else:
            logger.warning("Не удалось собрать позиции из deals.csv")

    # Создаём пустой шаблон holdings
    logger.warning("portfolio_holdings.csv и deals.csv отсутствуют. Создаю пустой шаблон.")
    empty = pd.DataFrame(columns=[
        "asset_type", "SECID_or_ISIN", "quantity",
        "avg_price", "first_purchase_date", "last_purchase_date"
    ])
    save_csv(empty, holdings_path)
    
    # 3. Создаём пустой cash_balance.csv, если его нет
    cash_path = data_dir / "cash_balance.csv"
    if not cash_path.exists():
        logger.info("cash_balance.csv не найден. Создаю пустой шаблон.")
        empty_cash = pd.DataFrame(columns=["date", "currency", "amount"])
        save_csv(empty_cash, cash_path)
        logger.info(f"Создан пустой {cash_path.name}")
        
    return holdings_path, True


def check_ranking_files(output_dir: Path, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """Загружает рейтинги из module1 и module2. Если файла нет — возвращает пустой DataFrame."""
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
# Основная функция модуля
# ============================================================================

def evaluate_portfolio(date: str, config: dict, data_dir: Path, output_dir: Path,
                       logger: logging.Logger, diagnostics: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Оценка портфеля, расчёт дефицитов, генерация рекомендаций и аутсайдеров.
    """
    date_dt = pd.to_datetime(date)

    # --- Загрузка рейтингов ---
    rankings = check_ranking_files(output_dir, logger)
    stock_ranking = rankings["stock_ranking.csv"]
    ofz_ranking = rankings["ofz_ranking.csv"]
    corp_ranking = rankings["corp_ranking.csv"]
    bond_ranking = pd.concat([ofz_ranking, corp_ranking], ignore_index=True) if not ofz_ranking.empty or not corp_ranking.empty else pd.DataFrame()

    # --- Загрузка входных данных ---
    holdings_path = data_dir / "portfolio_holdings.csv"
    if holdings_path.exists():
        holdings = read_csv_flex(holdings_path)
    else:
        holdings = pd.DataFrame()

    cash_path = data_dir / "cash_balance.csv"
    if cash_path.exists():
        cash = read_csv_flex(cash_path)
        cash_value = cash[cash["currency"] == "RUB"]["amount"].sum() if "RUB" in cash["currency"].values else 0.0
    else:
        logger.warning("cash_balance.csv не найден, cash_value = 0")
        cash_value = 0.0

    prices_stocks_path = data_dir / "stock_prices.csv"
    prices_bonds_path = data_dir / "bond_prices.csv"

    latest_stock_prices = pd.DataFrame()
    if prices_stocks_path.exists():
        p = read_csv_flex(prices_stocks_path)
        p["date"] = pd.to_datetime(p["date"])
        latest_stock_prices = p[p["date"] <= date].sort_values("date").groupby("SECID").tail(1).set_index("SECID")

    latest_bond_prices = pd.DataFrame()
    if prices_bonds_path.exists():
        p = read_csv_flex(prices_bonds_path)
        p["date"] = pd.to_datetime(p["date"])
        latest_bond_prices = p[p["date"] <= date].sort_values("date").groupby("ISIN").tail(1).set_index("ISIN")

    # --- Загрузка universe для лотности ---
    universe_stocks = pd.DataFrame()
    if (data_dir / "stocks_universe.csv").exists():
        universe_stocks = read_csv_flex(data_dir / "stocks_universe.csv")
    universe_bonds_ofz = pd.DataFrame()
    if (data_dir / "bonds_OFZ.csv").exists():
        universe_bonds_ofz = read_csv_flex(data_dir / "bonds_OFZ.csv")
    universe_bonds_corp = pd.DataFrame()
    if (data_dir / "bonds_CORPS.csv").exists():
        universe_bonds_corp = read_csv_flex(data_dir / "bonds_CORPS.csv")

    # --- Оценка стоимости ---
    stock_val = 0.0
    bond_val = 0.0
    bond_val_clean = 0.0
    recommendations = []
    outsiders = []

    for _, h in holdings.iterrows():
        asset_type = h["asset_type"]
        secid_or_isin = h["SECID_or_ISIN"]
        qty = float(h["quantity"])
        avg_price = float(h["avg_price"]) if pd.notna(h.get("avg_price")) else 0.0

        if asset_type == "stock":
            if secid_or_isin in latest_stock_prices.index:
                close = float(latest_stock_prices.loc[secid_or_isin, "close"])
                val = qty * close
                stock_val += val

                holding_days = (date_dt - pd.to_datetime(h["first_purchase_date"])).days if pd.notna(h.get("first_purchase_date")) else 0
                pnl_pct = ((close - avg_price) / avg_price * 100) if avg_price > 0 else np.nan

                stock_info = stock_ranking[stock_ranking["SECID"] == secid_or_isin]
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
            else:
                # logger.warning(f"Нет цены для акции {secid_or_isin}")
                a = 0

        elif asset_type == "bond":
            if secid_or_isin in latest_bond_prices.index:
                p_row = latest_bond_prices.loc[secid_or_isin]
                nominal = 1000.0
                nkd = float(p_row["nkd_rub"]) if pd.notna(p_row.get("nkd_rub")) else 0.0
                if pd.isna(p_row.get("nkd_rub")):
                    diagnostics.append({"code": "nkd_missing", "details": f"ISIN {secid_or_isin}"})

                price_pct = float(p_row["price_pct"]) if pd.notna(p_row.get("price_pct")) else 0.0
                val = qty * ((price_pct / 100 * nominal) + nkd)
                clean_val = qty * (price_pct / 100 * nominal)
                bond_val += val
                bond_val_clean += clean_val
            else:
                logger.warning(f"Нет цены для облигации {secid_or_isin}")

    total_value = stock_val + bond_val + cash_value
    if total_value == 0:
        logger.error("total_value == 0 — нечего оценивать")
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
        return snapshot, pd.DataFrame(recommendations), pd.DataFrame(outsiders)

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
        for _, row in eligible.iterrows():
            secid = row["SECID"]
            target_w = float(row.get("target_weight_portfolio_pct", 0))
            # Текущий вес этой акции
            current_val = 0.0
            if secid in latest_stock_prices.index:
                h_rows = holdings[(holdings["asset_type"] == "stock") & (holdings["SECID_or_ISIN"] == secid)]
                if not h_rows.empty:
                    current_val = float(h_rows["quantity"].sum()) * float(latest_stock_prices.loc[secid, "close"])
            current_w = current_val / total_value * 100
            deficit_pct = target_w - current_w

            if deficit_pct > config["min_deficit_pct"]:
                price = float(row["price"])
                # Лотность
                lot_size = 1
                lot_source = "assumed"
                if not universe_stocks.empty:
                    u = universe_stocks[universe_stocks["SECID"] == secid]
                    if not u.empty and "lot_size" in u.columns:
                        lot_size = int(u["lot_size"].iloc[0])
                        lot_source = "universe"

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

    return snapshot, pd.DataFrame(recommendations), pd.DataFrame(outsiders)


# ============================================================================
# Точка входа
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """CLI для отладки module3_portfolio.py."""
    import argparse

    ap = argparse.ArgumentParser(description="Module 3: оценка портфеля (изолированный запуск)")
    ap.add_argument("--date", default="", help="Дата расчёта YYYY-MM-DD (по умолчанию — из config.yaml или today)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Путь к config.yaml")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Папка с входными CSV (data/raw)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Папка с выходами модулей 1-2 и модуля 3 (data/output)")
    args = ap.parse_args(argv)

    # Корневая директория проекта
    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else project_root / args.config
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем конфиг
    config = load_config(config_path)

    # Обработка даты
    date_from_config = config.get("date", "today")
    if not date_from_config or str(date_from_config).lower() in ("today", "auto", ""):
        date = args.date or datetime.now().strftime("%Y-%m-%d")
    else:
        date = args.date or str(date_from_config)

    logger = setup_logging(date, output_dir)
    logger.info(f"=== Module 3 start: date={date}, data_dir={data_dir}, output_dir={output_dir} ===")

    diagnostics: list[dict] = []

    try:
        # Проверка входных файлов
        holdings_path, created = check_required_files(data_dir, logger)
        if created:
            diagnostics.append({"code": "holdings_autogenerated", "details": str(holdings_path)})

        # Выполнение оценки
        snapshot, rebalance, outsiders = evaluate_portfolio(
            date, config, data_dir, output_dir, logger, diagnostics
        )

        # Сохранение результатов
        save_csv(snapshot, output_dir / "portfolio_snapshot.csv")
        save_csv(rebalance, output_dir / "rebalance_recommendations.csv")
        save_csv(outsiders, output_dir / "outsiders.csv")

        logger.info(f"portfolio_snapshot: {len(snapshot)} строк")
        logger.info(f"rebalance_recommendations: {len(rebalance)} строк")
        logger.info(f"outsiders: {len(outsiders)} строк")

        if not snapshot.empty:
            row = snapshot.iloc[0]
            logger.info(f"Total value: {row['total_value']:,.2f} RUB".replace(",", " "))
            logger.info(f"Stock: {row['stock_share_pct']:.2f}%, Bond: {row['bond_share_pct']:.2f}%, Cash: {row['cash_share_pct']:.2f}%")

        # diagnostics
        if diagnostics:
            # Читаем существующий diagnostics.csv (если есть от m1/m2) и дополняем
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