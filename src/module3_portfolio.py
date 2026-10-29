#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 3. Оценка портфеля и рекомендации.

Работает с матрицей субпортфелей portfolio_units.csv:
- Целевые веса бумаг рассчитываются как sum(units по строке) / sum(всех units) * 100%
- Стоп-лист читается из stoplist.csv (SECID, stop_date, reason)
- Позиции вне матрицы: SELL если прибыль, HOLD если убыток и <3 лет,
  SELL если убыток и >3 лет
- Тип бумаги определяется по справочникам (bonds_OFZ, bonds_CORPS, stocks_universe),
  с fallback по формату ISIN.

Автономный запуск:
    python src/module3_portfolio.py
    python -m src.module3_portfolio --date 2026-10-23
"""
from __future__ import annotations

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

MAX_AGE_DEALS_SEC = 3600
DEALS_URL = "http://box:78/money/hs/reports/deals"

# Схемы выходных файлов
DETAIL_COLS = ["SECID", "asset_type", "bucket", "quantity", "current_price",
               "current_value", "target_weight_pct", "target_weight_from_rating_pct",
               "target_value", "deviation_value", "current_weight_pct",
               "cost_basis", "accounting_profit", "accounting_profit_pct",
               "years_held", "dividend_sum", "dividend_pct",
               "first_purchase_date", "in_stoplist", "action", "reason"]

REC_COLS = ["date", "asset_type", "bucket", "SECID", "action", "reason_code", "reason_text",
            "current_qty", "current_weight_portfolio_pct", "target_weight_portfolio_pct",
            "deficit_pct", "qty_delta", "qty_delta_lots", "LOTSIZE", "lot_source",
            "amount_rub", "priority", "constraints_hit"]

SNAP_COLS = ["date", "total_value", "stock_value", "bond_value", "bond_value_clean",
             "cash_value", "stock_share_pct", "bond_share_pct", "cash_share_pct",
             "core_ofz_value", "core_ofz_share_pct",
             "long_ofz_value", "long_ofz_share_pct",
             "corp_value", "corp_share_pct",
             "target_stock_pct", "target_bond_pct",
             "deviation_stock_pct", "deviation_bond_pct",
             "cash_buffer_excess_pct"]

OUT_COLS = ["date", "asset_type", "SECID", "quantity", "first_purchase_date",
            "holding_days", "avg_price", "current_price", "current_value", "pnl_pct",
            "in_stoplist", "action", "reason", "priority"]

HOLDINGS_COLS = ["asset_type", "SECID", "quantity",
                 "avg_price", "first_purchase_date", "last_purchase_date"]


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


# ============================================================================
# Определение типа бумаги по справочникам
# ============================================================================
def load_asset_type_maps(data_dir: Path, logger: logging.Logger) -> tuple[set, set]:
    """Загружает SECID из справочников облигаций и акций."""
    bond_secids: set[str] = set()
    stock_secids: set[str] = set()

    for fn in ("bonds_OFZ.csv", "bonds_CORPS.csv"):
        p = data_dir / fn
        if not p.exists():
            continue
        try:
            df = read_csv_flex(p)
            if "SECID" in df.columns:
                bond_secids.update(df["SECID"].astype(str).str.strip().str.upper())
        except Exception as e:
            logger.warning(f"Не удалось загрузить {fn}: {e}")

    p = data_dir / "stocks_universe.csv"
    if p.exists():
        try:
            df = read_csv_flex(p)
            if "SECID" in df.columns:
                stock_secids.update(df["SECID"].astype(str).str.strip().str.upper())
        except Exception as e:
            logger.warning(f"Не удалось загрузить stocks_universe.csv: {e}")

    logger.info(f"Карты типов бумаг: bonds={len(bond_secids)}, stocks={len(stock_secids)}")
    return bond_secids, stock_secids

def infer_asset_type(secid: str,
                     bond_secids: set | None = None,
                     stock_secids: set | None = None) -> str:
    """Определяет тип бумаги по справочникам (приоритет) или по формату ISIN (fallback)."""
    s = str(secid).strip().upper()

    if bond_secids is not None and s in bond_secids:
        return "bond"
    if stock_secids is not None and s in stock_secids:
        return "stock"

    if (s.startswith("RU") or s.startswith("XS")) and len(s) >= 10:
        return "bond"

    return "stock"

    # Fallback по формату ISIN
    if (s.startswith("RU") or s.startswith("XS")) and len(s) >= 10:
        return "bond"

    return "stock"

def build_isin_to_secid_map(data_dir: Path, logger: logging.Logger) -> dict:
    """Строит маппинг ISIN -> SECID из справочников bonds_OFZ.csv и bonds_CORPS.csv."""
    isin_to_secid = {}
    for fn in ("bonds_OFZ.csv", "bonds_CORPS.csv"):
        p = data_dir / fn
        if not p.exists():
            continue
        try:
            df = read_csv_flex(p)
            if "ISIN" in df.columns and "SECID" in df.columns:
                for isin, secid in zip(df["ISIN"], df["SECID"]):
                    isin_s = str(isin).strip().upper()
                    secid_s = str(secid).strip().upper()
                    if isin_s and secid_s:
                        isin_to_secid[isin_s] = secid_s
        except Exception as e:
            logger.warning(f"Не удалось загрузить {fn} для маппинга ISIN->SECID: {e}")
    logger.info(f"Маппинг ISIN->SECID построен: {len(isin_to_secid)} бумаг")
    return isin_to_secid

# ============================================================================
# Загрузка deals → holdings
# ============================================================================
def file_age(path: Path) -> float:
    return time.time() - path.stat().st_mtime if path.exists() else float("inf")


def is_fresh(path: Path, max_age: float) -> bool:
    return file_age(path) <= max_age


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
        deals_path.write_text(response.text, encoding="utf-8")
        logger.info(f"Файл deals.csv успешно обновлен")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при загрузке deals.csv: {e}")
        if not deals_path.exists():
            raise RuntimeError("Не удалось загрузить deals.csv, и локальный файл отсутствует.")
        logger.warning("Используется устаревший локальный файл deals.csv.")


def aggregate_holdings(raw: pd.DataFrame, logger: logging.Logger,
                       bond_secids: set | None = None,
                       stock_secids: set | None = None) -> pd.DataFrame:
    """Агрегирует holdings по (asset_type, SECID). avg_price — средневзвешенная."""
    if raw is None or raw.empty:
        return empty_df(HOLDINGS_COLS)
    h = raw.copy()
    h.columns = [str(c).strip() for c in h.columns]
    for c in HOLDINGS_COLS:
        if c not in h.columns:
            h[c] = np.nan
    h["asset_type"] = h["asset_type"].fillna("").astype(str).str.strip().str.lower()
    h["SECID"] = h["SECID"].fillna("").astype(str).str.strip().str.upper()
    h["quantity"] = pd.to_numeric(h["quantity"], errors="coerce").fillna(0.0)
    h["avg_price"] = pd.to_numeric(h["avg_price"], errors="coerce")
    for c in ("first_purchase_date", "last_purchase_date"):
        h[c] = pd.to_datetime(h[c], errors="coerce")

    # Если asset_type пустой — определяем по SECID через справочники
    mask = h["asset_type"].isin(["stock", "bond"])
    if (~mask).any():
        h.loc[~mask, "asset_type"] = h.loc[~mask, "SECID"].apply(
            lambda x: infer_asset_type(x, bond_secids, stock_secids)
        )
        
    valid = h["asset_type"].isin(["stock", "bond"]) & h["SECID"].ne("")
    h = h[valid & h["quantity"].gt(0)].copy()
    if h.empty:
        return empty_df(HOLDINGS_COLS)

    rows = []
    for (atype, code), g in h.groupby(["asset_type", "SECID"], sort=True):
        qty = float(g["quantity"].sum())
        if g["avg_price"].isna().any():
            avg = float("nan")
        else:
            avg = float((g["quantity"] * g["avg_price"]).sum() / qty)
        rows.append({
            "asset_type": atype, "SECID": code, "quantity": qty,
            "avg_price": avg,
            "first_purchase_date": g["first_purchase_date"].min(),
            "last_purchase_date": g["last_purchase_date"].max(),
        })
    return pd.DataFrame(rows, columns=HOLDINGS_COLS)


def deals_to_holdings(deals_path: Path, logger: logging.Logger,
                      bond_secids: set | None = None,
                      stock_secids: set | None = None) -> pd.DataFrame:
    """Собирает текущие позиции из файла сделок (FIFO)."""
    d = read_csv_flex(deals_path)
    if d.empty:
        return empty_df(HOLDINGS_COLS)
    d.columns = [str(c).strip().upper() for c in d.columns]
    aliases = {"ТИКЕР": "SECID", "ВИД": "TYPE", "ОПЕРАЦИЯ": "TYPE",
               "КОЛ-ВО": "COUNT", "КОЛИЧЕСТВО": "COUNT", "QTY": "COUNT",
               "СУММА": "SUM", "СТОИМОСТЬ": "SUM", "AMOUNT": "SUM",
               "ДАТА": "DATE", "TRADE_DATE": "DATE"}
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d.columns})
    need = {"SECID", "TYPE", "COUNT", "SUM", "DATE"}
    if need - set(d.columns):
        raise RuntimeError(f"deals.csv: нет колонок {sorted(need - set(d.columns))}")

    d["SECID"] = d["SECID"].astype(str).str.strip().str.upper()
    d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce")
    d["COUNT"] = pd.to_numeric(d["COUNT"], errors="coerce")
    d["SUM"] = pd.to_numeric(d["SUM"], errors="coerce")
    typ = d["TYPE"].astype(str).str.strip().str.upper()
    d["TYPE"] = typ.where(typ.isin(["BUY", "SELL"]),
                          typ.map({"ПОКУПКА": "BUY", "ПРОДАЖА": "SELL"}))
    d = d[d["TYPE"].isin(["BUY", "SELL"])].dropna(subset=["DATE", "COUNT", "SUM"])
    d = d[d["COUNT"].gt(0)].sort_values(["SECID", "DATE"], kind="stable")

    rows = []
    for code, g in d.groupby("SECID", sort=True):
        queue = []
        for _, t in g.iterrows():
            q = float(t["COUNT"])
            price = float(t["SUM"]) / q
            if t["TYPE"] == "BUY":
                queue.append([q, price, t["DATE"]])
                continue
            rest = q
            while rest > 1e-9 and queue:
                take = min(queue[0][0], rest)
                queue[0][0] -= take
                rest -= take
                if queue[0][0] <= 1e-9:
                    queue.pop(0)
        qty = sum(l[0] for l in queue)
        if qty <= 1e-9:
            continue
        cost = sum(l[0] * l[1] for l in queue)
        atype = infer_asset_type(code, bond_secids, stock_secids)
        rows.append({
            "asset_type": atype, "SECID": code, "quantity": qty,
            "avg_price": cost / qty,
            "first_purchase_date": min(l[2] for l in queue),
            "last_purchase_date": g["DATE"].max(),
        })
    logger.info(f"deals -> FIFO позиций: {len(rows)}")
    return pd.DataFrame(rows, columns=HOLDINGS_COLS) if rows else empty_df(HOLDINGS_COLS)


# ============================================================================
# Загрузка матрицы субпортфелей
# ============================================================================
def load_matrix(path: Path, logger: logging.Logger,
                bond_secids: set | None = None,
                stock_secids: set | None = None,
                isin_to_secid_map: dict | None = None) -> pd.DataFrame:
    """Загружает матрицу, агрегирует дубликаты по SECID (суммирует units).
    Выполняет миграцию: если в SECID лежит ISIN — переносит в ISIN, восстанавливает SECID."""
    if not path.exists():
        logger.warning(f"Матрица {path.name} не найдена")
        return empty_df(["SECID", "asset_type", "ISIN"])
    df = read_csv_flex(path)
    if df.empty or "SECID" not in df.columns:
        return empty_df(["SECID", "asset_type", "ISIN"])

    # Гарантируем наличие ISIN
    if "ISIN" not in df.columns:
        df["ISIN"] = ""
    df["ISIN"] = df["ISIN"].fillna("").astype(str).str.strip().str.upper()

    # Миграция: если SECID похож на ISIN — переносим в ISIN, восстанавливаем SECID
    migrated = 0
    if isin_to_secid_map:
        for idx, row in df.iterrows():
            secid = str(row.get("SECID", "")).strip().upper()
            if (secid.startswith("RU") or secid.startswith("XS")) and len(secid) >= 10:
                if not df.at[idx, "ISIN"]:
                    df.at[idx, "ISIN"] = secid
                new_secid = isin_to_secid_map.get(secid, "")
                if new_secid:
                    df.at[idx, "SECID"] = new_secid
                    migrated += 1
    if migrated:
        logger.info(f"Матрица: мигрировано {migrated} строк (ISIN -> SECID+ISIN)")

    # Мета-колонки (ISIN не трогаем при агрегации)
    meta_cols = {"SECID", "asset_type", "ISIN"}
    num_cols = [c for c in df.columns if c not in meta_cols]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "asset_type" not in df.columns:
        df["asset_type"] = ""
    df["asset_type"] = df["asset_type"].fillna("").astype(str).str.strip().str.lower()

    # Если asset_type не stock/bond — определяем через справочники
    mask = ~df["asset_type"].isin(["stock", "bond"])
    if mask.any():
        df.loc[mask, "asset_type"] = df.loc[mask, "SECID"].apply(
            lambda x: infer_asset_type(x, bond_secids, stock_secids)
        )

    df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()

    # Агрегация дубликатов по SECID
    if num_cols:
        agg_dict = {c: "sum" for c in num_cols}
        agg_dict["asset_type"] = "first"
        agg_dict["ISIN"] = "first"
        df = df.groupby("SECID", as_index=False).agg(agg_dict)
    df = df.sort_values("SECID").reset_index(drop=True)

    logger.info(f"Матрица загружена: {len(df)} бумаг, {len(num_cols)} субпортфелей")
    return df


def compute_target_weights(matrix: pd.DataFrame) -> dict:
    """Считает целевые веса для каждой бумаги из матрицы."""
    meta_cols = {"SECID", "asset_type", "ISIN"}  
    num_cols = [c for c in matrix.columns if c not in meta_cols]

    m = matrix.copy()
    m["total_units"] = m[num_cols].sum(axis=1) if num_cols else 0
    total_all = m["total_units"].sum()

    if total_all <= 0:
        return {}

    m["target_weight_pct"] = m["total_units"] / total_all * 100
    return dict(zip(m["SECID"], m["target_weight_pct"]))


# ============================================================================
# Загрузка стоп-листа
# ============================================================================
def load_stoplist(path: Path, date_dt: pd.Timestamp, logger: logging.Logger) -> set:
    """Загружает stoplist.csv, возвращает set активных SECID (stop_date >= date)."""
    if not path.exists():
        logger.info("stoplist.csv не найден — стоп-лист пуст")
        return set()
    df = read_csv_flex(path)
    if df.empty or "SECID" not in df.columns or "stop_date" not in df.columns:
        logger.warning("stoplist.csv пуст или некорректен")
        return set()
    df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()
    df["stop_date"] = pd.to_datetime(df["stop_date"], errors="coerce")
    active = df[df["stop_date"] >= date_dt]
    result = set(active["SECID"])
    logger.info(f"Стоп-лист: {len(result)} активных бумаг")
    return result


# ============================================================================
# Загрузка цен и метаданных
# ============================================================================
def load_stock_prices(path: Path, date_dt: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning("stock_prices.csv не найден")
        return pd.DataFrame()
    df = read_csv_flex(path)
    if df.empty:
        return pd.DataFrame()
    df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= date_dt].sort_values("date").groupby("SECID").tail(1)
    return df.set_index("SECID")


def load_bond_prices(path: Path, date_dt: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning("bond_prices.csv не найден")
        return pd.DataFrame()
    df = read_csv_flex(path)
    if df.empty:
        return pd.DataFrame()
    df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= date_dt].sort_values("date").groupby("SECID").tail(1)
    return df.set_index("SECID")


def load_bond_meta(data_dir: Path, logger: logging.Logger) -> tuple:
    """Возвращает (nominals, lots, maturities) — словари {ISIN: value}."""
    nominals, lots, maturities = {}, {}, {}
    for fn in ("bonds_OFZ.csv", "bonds_CORPS.csv"):
        p = data_dir / fn
        if not p.exists():
            continue
        d = read_csv_flex(p)
        if "ISIN" not in d.columns:
            continue
        d["ISIN"] = d["ISIN"].astype(str).str.strip().str.upper()
        if "nominal" in d.columns:
            for isin, v in zip(d["ISIN"], pd.to_numeric(d["nominal"], errors="coerce")):
                n = _num(v, 1000.0)
                if n > 0:
                    nominals[isin] = n
        # Лотность: пробуем разные варианты имени колонки
        for lot_col in ("LOTSIZE", "LOTSIZE", "lotsize"):
            if lot_col in d.columns:
                for isin, v in zip(d["ISIN"], pd.to_numeric(d[lot_col], errors="coerce")):
                    lots[isin] = max(1, int(_num(v, 1.0)))
                break
        if "maturity_date" in d.columns:
            for isin, v in zip(d["ISIN"], d["maturity_date"]):
                md = pd.to_datetime(v, errors="coerce")
                if pd.notna(md):
                    maturities[isin] = md
    logger.info(f"Мета облигаций: номиналы={len(nominals)}, лоты={len(lots)}, погашения={len(maturities)}")
    return nominals, lots, maturities


def load_stock_lots(data_dir: Path, logger: logging.Logger) -> dict:
    """Загружает lot_size для акций из stocks_universe.csv."""
    p = data_dir / "stocks_universe.csv"
    if not p.exists():
        return {}
    df = read_csv_flex(p)
    if df.empty or "SECID" not in df.columns:
        return {}
    df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()
    result = {}
    for lot_col in ("LOTSIZE", "LOTSIZE", "lotsize"):
        if lot_col in df.columns:
            for secid, v in zip(df["SECID"], pd.to_numeric(df[lot_col], errors="coerce")):
                result[secid] = max(1, int(_num(v, 1.0)))
            break
    logger.info(f"Лотность акций загружена: {len(result)} тикеров")
    return result


def load_cash(data_dir: Path, logger: logging.Logger) -> float:
    p = data_dir / "cash_balance.csv"
    if not p.exists():
        logger.warning("cash_balance.csv не найден, cash_value = 0")
        return 0.0
    df = read_csv_flex(p)
    if df.empty:
        return 0.0
    if "currency" in df.columns and "amount" in df.columns:
        rub = df["currency"].astype(str).str.upper() == "RUB"
        return float(df.loc[rub, "amount"].sum())
    return 0.0


def load_stock_ranking(output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    p = output_dir / "stock_ranking.csv"
    if not p.exists():
        return pd.DataFrame()
    df = read_csv_flex(p)
    if "SECID" in df.columns:
        df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()
    return df

def load_bond_rankings(output_dir: Path, logger: logging.Logger) -> tuple:
    """
    Загружает три рейтинга облигаций: core_ofz, long_ofz, corp.
    Возвращает:
      - объединённый DataFrame всех рейтингов
      - словарь {ISIN: {bucket, target_weight_portfolio_pct, rank, eligible, ytm_gross_pct, current_yield_pct}}
    """
    rankings = {}
    for bucket, filename in [
        ("core_ofz", "core_ofz_ranking.csv"),
        ("long_ofz", "long_ofz_ranking.csv"),
        ("corp", "corp_ranking.csv"),
    ]:
        path = output_dir / filename
        if not path.exists():
            logger.warning(f"Рейтинг {filename} не найден")
            continue
        df = read_csv_flex(path)
        if df.empty:
            continue
        df["bucket"] = bucket
        if "ISIN" in df.columns:
            df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()
        rankings[bucket] = df
        logger.info(f"Загружен рейтинг {filename}: {len(df)} бумаг, eligible: {int(df.get('eligible', pd.Series(dtype=bool)).sum())}")
    
    if not rankings:
        return pd.DataFrame(), {}
    
    all_rankings = pd.concat(rankings.values(), ignore_index=True)
    
    # Словарь для быстрого доступа по ISIN
    rating_info = {}
    for _, row in all_rankings.iterrows():
        isin = str(row.get("ISIN", "")).strip().upper()
        if not isin:
            continue
        rating_info[isin] = {
            "bucket": row.get("bucket", ""),
            "target_weight_portfolio_pct": _num(row.get("target_weight_portfolio_pct"), 0.0),
            "rank": _num(row.get("rank"), float("nan")),
            "eligible": bool(row.get("eligible", False)),
            "ytm_gross_pct": _num(row.get("ytm_gross_pct"), float("nan")),
            "current_yield_pct": _num(row.get("current_yield_pct"), float("nan")),
        }
    
    return all_rankings, rating_info

# ============================================================================
# Проверка входных файлов
# ============================================================================
def check_required_files(data_dir: Path, logger: logging.Logger,
                         bond_secids: set | None = None,
                         stock_secids: set | None = None) -> tuple:
    """Проверяет holdings/deals, создаёт пустые шаблоны при необходимости."""
    data_dir.mkdir(parents=True, exist_ok=True)
    hp = data_dir / "portfolio_holdings.csv"
    dp = data_dir / "deals.csv"
    created = False

    try:
        fetch_and_save_deals(data_dir, logger)
    except Exception as e:
        logger.warning(f"deals.csv не обновлён: {e}")

    if dp.exists() and (not hp.exists() or file_age(dp) <= file_age(hp)):
        h = deals_to_holdings(dp, logger, bond_secids, stock_secids)
        if not h.empty:
            save_csv(aggregate_holdings(h, logger, bond_secids, stock_secids), hp)
            created = True
            logger.info(f"portfolio_holdings пересобран из deals: {len(h)} позиций")

    if not hp.exists():
        save_csv(empty_df(HOLDINGS_COLS), hp)
        created = True
        logger.warning("Создан пустой portfolio_holdings.csv")

    cp = data_dir / "cash_balance.csv"
    if not cp.exists():
        save_csv(pd.DataFrame(columns=["currency", "amount"]), cp)
        logger.warning("Создан пустой cash_balance.csv")

    return hp, created


# ============================================================================
# Вспомогательные
# ============================================================================
def _num(v, default=float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def empty_df(cols: list) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)

# ============================================================================
# Основная функция
# ============================================================================
def evaluate_portfolio(date: str, config: dict, data_dir: Path, output_dir: Path,
                       logger: logging.Logger, diagnostics: list) -> tuple:
    date_dt = pd.to_datetime(date)

    # 0. Загрузка карт типов бумаг (SECID → asset_type)
    bond_secids, stock_secids = load_asset_type_maps(data_dir, logger)

    # 0.5. Маппинг ISIN -> SECID для миграции матрицы
    isin_to_secid_map = build_isin_to_secid_map(data_dir, logger)

    # 1. Загрузка матрицы (с миграцией)
    matrix = load_matrix(data_dir / "portfolio_units.csv", logger,
                         bond_secids, stock_secids, isin_to_secid_map)
    if matrix.empty:
        raise RuntimeError("Матрица субпортфелей пуста — нечего оценивать")

    # 2. Целевые веса (по SECID)
    target_weights = compute_target_weights(matrix)
    total_units_all = sum(target_weights.values())
    logger.info(f"Целевые веса рассчитаны: {len(target_weights)} бумаг, сумма units = {total_units_all:.2f}")

    # 3. Asset types из матрицы
    asset_types = dict(zip(matrix["SECID"], matrix["asset_type"]))

    # 3.5. Маппинг SECID -> ISIN для облигаций (для поиска цен/метаданных/рейтинга)
    # secid_to_isin = {}
    # for _, row in matrix.iterrows():
    #     secid = str(row.get("SECID", "")).strip().upper()
    #     isin = str(row.get("ISIN", "")).strip().upper()
    #     if secid and isin:
    #         secid_to_isin[secid] = isin

    # 4. Загрузка holdings
    holdings = aggregate_holdings(
        read_csv_flex(data_dir / "portfolio_holdings.csv"), logger,
        bond_secids, stock_secids
    )

    # 5. Загрузка цен и метаданных
    stock_prices = load_stock_prices(data_dir / "stock_prices.csv", date_dt, logger)
    bond_prices = load_bond_prices(data_dir / "bond_prices.csv", date_dt, logger)
    nominals, bond_lots, maturities = load_bond_meta(data_dir, logger)
    stock_lots = load_stock_lots(data_dir, logger)

    # 6. Стоп-лист
    stoplist = load_stoplist(data_dir / "stoplist.csv", date_dt, logger)

    # 7. Рейтинг акций (для dps_reg_next)
    stock_ranking = load_stock_ranking(output_dir, logger)
    dps_map = {}
    if not stock_ranking.empty and "dps_reg_next" in stock_ranking.columns:
        for _, r in stock_ranking.iterrows():
            dps_map[r["SECID"]] = _num(r.get("dps_reg_next"), 0.0)

    # 7.5. Рейтинги облигаций (core_ofz, long_ofz, corp)
    bond_rankings_df, bond_rating_info = load_bond_rankings(output_dir, logger)
    logger.info(f"Рейтинги облигаций: {len(bond_rating_info)} бумаг, бакеты: "
                f"{set(v['bucket'] for v in bond_rating_info.values())}")

    # 8. Кэш
    cash_value = load_cash(data_dir, logger)

    # 9. Оценка позиций
    positions = []
    all_secids = set(holdings["SECID"].str.upper()) | set(target_weights.keys())
    for secid in all_secids:
        atype = asset_types.get(secid, "stock")

        # Поиск в holdings (для облигаций — по ISIN, для акций — по SECID)
        h_row = pd.DataFrame()
        if atype == "bond":
            # isin = secid_to_isin.get(secid, "")
            if secid:
                h_row = holdings[holdings["SECID"].str.upper() == secid]
        if h_row.empty:
            h_row = holdings[holdings["SECID"].str.upper() == secid]
        if not h_row.empty:
            h_row = h_row.iloc[0]
            qty = float(h_row["quantity"])
            avg_price = _num(h_row["avg_price"])
            first_purchase = h_row["first_purchase_date"]
        else:
            qty = 0.0
            avg_price = float("nan")
            first_purchase = pd.NaT

        # Цена и текущая стоимость
        current_price = float("nan")
        current_value = float("nan")
        clean_value = 0.0
        if atype == "stock":
            if secid in stock_prices.index:
                current_price = _num(stock_prices.loc[secid].get("close"))
                if math.isfinite(current_price) and qty > 0:
                    current_value = qty * current_price
        else:  # bond
            # isin = secid_to_isin.get(secid, "")
            if secid and secid in bond_prices.index:
                bp = bond_prices.loc[secid]
                price_pct = _num(bp.get("price_pct"))
                nkd = _num(bp.get("nkd_rub"), 0.0)
                nominal = nominals.get(secid, 1000.0)
                if math.isfinite(price_pct):
                    current_price = price_pct / 100.0 * nominal + nkd
                    if qty > 0:
                        current_value = qty * current_price
                    clean_value = qty * price_pct / 100.0 * nominal

        cost_basis = qty * avg_price if (qty > 0 and math.isfinite(avg_price)) else float("nan")

        # Целевой вес и бакет
        target_w = target_weights.get(secid, 0.0)
        bucket = ""
        target_w_from_rating = 0.0

        # Для облигаций: обогащаем информацией из рейтингов (по ISIN)
        if atype == "bond":
            # isin = secid_to_isin.get(secid, "")
            if secid and secid in bond_rating_info:
                rating_data = bond_rating_info[secid]
                bucket = rating_data.get("bucket", "")
                target_w_from_rating = rating_data.get("target_weight_portfolio_pct", 0.0)
                # Если есть целевой вес из рейтинга — используем его (приоритет над матрицей)
                if target_w_from_rating > 0:
                    target_w = target_w_from_rating

        # Дивиденды (только для акций)
        div_sum = dps_map.get(secid, 0.0) * qty if atype == "stock" else 0.0
        div_pct = div_sum / current_value * 100 if (math.isfinite(current_value) and current_value > 0) else float("nan")

        # PnL
        pnl_pct = ((current_value / cost_basis) - 1) * 100 if (math.isfinite(current_value) and math.isfinite(cost_basis) and cost_basis > 0) else float("nan")

        # Years held
        years_held = float("nan")
        holding_days = float("nan")
        if pd.notna(first_purchase):
            try:
                fp = pd.to_datetime(first_purchase)
                holding_days = (date_dt - fp).days
                years_held = holding_days / 365.25
            except Exception:
                pass

        # In stoplist
        in_stop = secid in stoplist

        # Определяем action/reason/priority
        action, reason, priority = "HOLD", "no_action", 100

        # Учётная прибыль (нереализованная P&L) для дивидендных правил
        accounting_profit_val = (
            (current_value - cost_basis)
            if (math.isfinite(current_value) and math.isfinite(cost_basis))
            else float("nan")
        )

        # Дивидендные правила (только для акций с ожидаемыми дивидендами):
        # Вариант 1: profit > div_sum → продать, зафиксировать прибыль
        # Вариант 2: 0 < profit <= div_sum → держать, ждать дивиденд (перехватывает profit_take и three_year_exit)
        div_exceeds_profit = (
            (div_sum > 0)
            and math.isfinite(accounting_profit_val)
            and (accounting_profit_val > div_sum)
        )
        div_covers_profit = (
            (div_sum > 0)
            and math.isfinite(accounting_profit_val)
            and (accounting_profit_val > 0)
            and (accounting_profit_val <= div_sum)
        )

        if in_stop:
            action, reason, priority = "SELL", "stop_list", 1
        elif atype == "bond":
            # isin = secid_to_isin[secid]
            if secid in maturities and maturities[secid] <= date_dt:
                action, reason, priority = "SELL", "matured", 2
        elif target_w > 0:  # есть в матрице
            if div_exceeds_profit:
                # Вариант 1: прибыль больше ожидаемых дивидендов → продать
                action, reason, priority = "SELL", "profit_exceeds_dividend", 3
            elif div_covers_profit:
                # Вариант 2: прибыль есть, но меньше дивидендов → держать
                action, reason, priority = "HOLD", "hold_for_dividend", 100
            elif math.isfinite(years_held) and years_held >= 3 and math.isfinite(pnl_pct) and pnl_pct < 0:
                action, reason, priority = "SELL", "three_year_exit", 3
            elif math.isfinite(pnl_pct) and pnl_pct >= config.get("profit_take_pct", 5.0):
                action, reason, priority = "SELL", "profit_take", 4
            else:
                action, reason, priority = "HOLD", "in_matrix", 100
        else:  # нет в матрице
            if math.isfinite(current_value) and math.isfinite(cost_basis):
                if current_value > cost_basis:
                    action, reason, priority = "SELL", "not_in_matrix", 4
                elif math.isfinite(years_held) and years_held > 3:
                    action, reason, priority = "SELL", "over_3_years", 3
                else:
                    action, reason, priority = "HOLD", "not_in_matrix_loss", 100
            elif not math.isfinite(current_value):
                action, reason, priority = "HOLD", "no_price", 99

        positions.append({
            "SECID": secid,
            "asset_type": atype,
            "bucket": bucket,
            "quantity": qty,
            "current_price": round(current_price, 4) if math.isfinite(current_price) else float("nan"),
            "current_value": round(current_value, 2) if math.isfinite(current_value) else float("nan"),
            "target_weight_pct": round(target_w, 4),
            "target_weight_from_rating_pct": round(target_w_from_rating, 4),
            "cost_basis": round(cost_basis, 2) if math.isfinite(cost_basis) else float("nan"),
            "accounting_profit": round(current_value - cost_basis, 2) if (math.isfinite(current_value) and math.isfinite(cost_basis)) else float("nan"),
            "accounting_profit_pct": round(pnl_pct, 4) if math.isfinite(pnl_pct) else float("nan"),
            "years_held": round(years_held, 2) if math.isfinite(years_held) else float("nan"),
            "dividend_sum": round(div_sum, 2),
            "dividend_pct": round(div_pct, 4) if math.isfinite(div_pct) else float("nan"),
            "first_purchase_date": first_purchase,
            "in_stoplist": in_stop,
            "action": action,
            "reason": reason,
            "priority": priority,
            "_clean_value": clean_value,
            "_pnl_pct": pnl_pct,
            "_holding_days": holding_days,
        })

    df_pos = pd.DataFrame(positions)

    # 10. Расчёт total_value
    stock_val = df_pos.loc[df_pos["asset_type"] == "stock", "current_value"].sum()
    bond_val = df_pos.loc[df_pos["asset_type"] == "bond", "current_value"].sum()
    bond_clean = df_pos.loc[df_pos["asset_type"] == "bond", "_clean_value"].sum()
    total_value = stock_val + bond_val + cash_value
    if total_value <= 0:
        raise RuntimeError("total_value == 0 — нечего оценивать")

    # 11. Заполнение deviation, target_value, current_weight_pct
    # База для target_value — сумма current_value только по бумагам в плане
    plan_value = df_pos.loc[df_pos["target_weight_pct"] > 0, "current_value"].fillna(0).sum()
    if plan_value <= 0:
        plan_value = total_value  # fallback, если в плане ничего нет
    df_pos["target_value"] = df_pos["target_weight_pct"] / 100 * plan_value
    # df_pos["target_value"] = df_pos["target_weight_pct"] / 100 * total_value
    df_pos["deviation_value"] = df_pos["current_value"].fillna(0) - df_pos["target_value"]
    df_pos["current_weight_pct"] = df_pos["current_value"].fillna(0) / total_value * 100

    # 12. Snapshot
    target_stock_pct = df_pos.loc[df_pos["asset_type"] == "stock", "target_weight_pct"].sum()
    target_bond_pct = df_pos.loc[df_pos["asset_type"] == "bond", "target_weight_pct"].sum()

    # Разбивка по бакетам облигаций
    bond_by_bucket = {}
    for b in ["core_ofz", "long_ofz", "corp"]:
        mask = (df_pos["asset_type"] == "bond") & (df_pos["bucket"] == b)
        val = df_pos.loc[mask, "current_value"].sum()
        bond_by_bucket[f"{b}_value"] = val
        bond_by_bucket[f"{b}_share_pct"] = (val / total_value * 100) if total_value > 0 else 0.0

    snapshot = pd.DataFrame([{
        "date": date,
        "total_value": total_value,
        "stock_value": stock_val,
        "bond_value": bond_val,
        "bond_value_clean": bond_clean,
        "cash_value": cash_value,
        "stock_share_pct": stock_val / total_value * 100,
        "bond_share_pct": bond_val / total_value * 100,
        "cash_share_pct": cash_value / total_value * 100,
        **bond_by_bucket,  # core_ofz_value, core_ofz_share_pct, long_ofz_value, ...
        "target_stock_pct": target_stock_pct,
        "target_bond_pct": target_bond_pct,
        "deviation_stock_pct": stock_val / total_value * 100 - target_stock_pct,
        "deviation_bond_pct": bond_val / total_value * 100 - target_bond_pct,
        "cash_buffer_excess_pct": cash_value / total_value * 100 - config.get("cash_buffer_pct", 2.0),
    }], columns=SNAP_COLS)

    # 13. Outsiders (все позиции с action != HOLD или с проблемами)
    outsiders = []
    for _, p in df_pos.iterrows():
        if p["action"] != "HOLD" or p["in_stoplist"] or not math.isfinite(p["current_value"]):
            outsiders.append({
                "date": date,
                "asset_type": p["asset_type"],
                "SECID": p["SECID"],
                "quantity": p["quantity"],
                "first_purchase_date": p["first_purchase_date"],
                "holding_days": p["_holding_days"] if math.isfinite(p["_holding_days"]) else float("nan"),
                "avg_price": p["cost_basis"] / p["quantity"] if (p["quantity"] > 0 and math.isfinite(p["cost_basis"])) else float("nan"),
                "current_price": p["current_price"],
                "current_value": p["current_value"],
                "pnl_pct": p["_pnl_pct"],
                "in_stoplist": p["in_stoplist"],
                "action": p["action"],
                "reason": p["reason"],
                "priority": p["priority"],
            })
    df_out = pd.DataFrame(outsiders, columns=OUT_COLS) if outsiders else empty_df(OUT_COLS)

    # 14. Recommendations (SELL + BUY)
    recs = []
    available = max(0.0, cash_value - total_value * config.get("cash_buffer_pct", 2.0) / 100)

    # SELL рекомендации
    for _, p in df_pos.iterrows():
        if p["action"] != "SELL":
            continue
        if not math.isfinite(p["current_price"]) or p["quantity"] <= 0:
            continue
        isin_for_lot = p["SECID"]
        lot_size = stock_lots.get(p["SECID"], 1) if p["asset_type"] == "stock" else bond_lots.get(isin_for_lot, 1)
        lot_source = "universe" if (p["SECID"] in stock_lots or isin_for_lot in bond_lots) else "assumed"
        nlots = math.floor(p["quantity"] / lot_size)
        qd = nlots * lot_size
        if qd <= 0:
            continue
        amount = qd * p["current_price"]
        recs.append({
            "date": date, "asset_type": p["asset_type"], "SECID": p["SECID"], "bucket": p.get("bucket", ""),
            "action": "SELL", "reason_code": p["reason"], "reason_text": p["reason"],
            "current_qty": p["quantity"], "current_weight_portfolio_pct": p["current_weight_pct"],
            "target_weight_portfolio_pct": p["target_weight_pct"],
            "deficit_pct": p["target_weight_pct"] - p["current_weight_pct"],
            "qty_delta": -qd, "qty_delta_lots": -nlots, "LOTSIZE": lot_size, "lot_source": lot_source,
            "amount_rub": amount, "priority": p["priority"], "constraints_hit": "",
        })
        available += amount  # Освобождаем кэш от продаж

    # BUY рекомендации
    buy_candidates = []
    for secid, target_w in target_weights.items():
        if target_w <= 0:
            continue
        p_row = df_pos[df_pos["SECID"] == secid]
        current_w = float(p_row["current_weight_pct"].iloc[0]) if not p_row.empty else 0.0
        deficit = target_w - current_w
        if deficit <= config.get("min_deficit_pct", 0.5):
            continue
        atype = asset_types.get(secid, "stock")
        if atype == "stock":
            price = _num(stock_prices.loc[secid].get("close")) if secid in stock_prices.index else float("nan")
            lot_size = stock_lots.get(secid, 1)
            lot_source = "universe" if secid in stock_lots else "assumed"
        else:  # bond
            # isin = secid_to_isin.get(secid, "")
            if secid and secid in bond_prices.index:
                bp = bond_prices.loc[secid]
                price_pct = _num(bp.get("price_pct"))
                nkd = _num(bp.get("nkd_rub"), 0.0)
                nominal = nominals.get(secid, 1000.0)
                price = price_pct / 100.0 * nominal + nkd if math.isfinite(price_pct) else float("nan")
            else:
                price = float("nan")
            lot_size = bond_lots.get(secid, 1) if secid else 1
            lot_source = "universe" if (secid and secid in bond_lots) else "assumed"
        buy_candidates.append((target_w, atype, secid, target_w, price, lot_size, lot_source, deficit))

    buy_candidates.sort(key=lambda x: -x[0])  # Сортируем по убыванию target_weight

    for i, (_, atype, secid, target_w, price, lot_size, lot_source, deficit) in enumerate(buy_candidates, start=5):
        if not math.isfinite(price) or price <= 0:
            bucket_for_rec = ""
            if atype == "bond":
                # isin = secid_to_isin.get(secid, "")
                if secid and secid in bond_rating_info:
                    bucket_for_rec = bond_rating_info[secid].get("bucket", "")
            recs.append({
                "date": date, "asset_type": atype, "SECID": secid, "bucket": bucket_for_rec,
                "action": "NO_ACTION", "reason_code": "no_price", "reason_text": "Нет цены",
                "current_qty": 0, "current_weight_portfolio_pct": 0,
                "target_weight_portfolio_pct": target_w, "deficit_pct": deficit,
                "qty_delta": 0, "qty_delta_lots": 0, "LOTSIZE": lot_size, "lot_source": lot_source,
                "amount_rub": 0, "priority": 99, "constraints_hit": "no_price",
            })
            continue
        desired = deficit / 100 * total_value
        afford = min(desired, available)
        lot_price = price * lot_size
        nlots = math.floor(afford / lot_price) if lot_price > 0 else 0
        if nlots <= 0:
            bucket_for_rec = ""
            if atype == "bond":
                # isin = secid_to_isin.get(secid, "")
                if secid and secid in bond_rating_info:
                    bucket_for_rec = bond_rating_info[secid].get("bucket", "")
            recs.append({
                "date": date, "asset_type": atype, "SECID": secid, "bucket": bucket_for_rec,
                "action": "NO_ACTION", "reason_code": "lot_too_small" if afford < lot_price else "insufficient_cash",
                "reason_text": "Лот слишком дорог" if afford < lot_price else "Недостаточно кэша",
                "current_qty": 0, "current_weight_portfolio_pct": 0,
                "target_weight_portfolio_pct": target_w, "deficit_pct": deficit,
                "qty_delta": 0, "qty_delta_lots": 0, "LOTSIZE": lot_size, "lot_source": lot_source,
                "amount_rub": 0, "priority": 99, "constraints_hit": "cash_limit" if desired > afford else "",
            })
            continue
        qd = nlots * lot_size
        amount = qd * price
        available -= amount
        bucket_for_rec = ""
        if atype == "bond":
            # isin = secid_to_isin.get(secid, "")
            if secid and secid in bond_rating_info:
                bucket_for_rec = bond_rating_info[secid].get("bucket", "")
        recs.append({
            "date": date, "asset_type": atype, "SECID": secid, "bucket": bucket_for_rec,
            "action": "BUY", "reason_code": "deficit", "reason_text": f"Deficit {deficit:.2f}%",
            "current_qty": 0, "current_weight_portfolio_pct": 0,
            "target_weight_portfolio_pct": target_w, "deficit_pct": deficit,
            "qty_delta": qd, "qty_delta_lots": nlots, "LOTSIZE": lot_size, "lot_source": lot_source,
            "amount_rub": amount, "priority": i, "constraints_hit": "",
        })

    df_rec = pd.DataFrame(recs, columns=REC_COLS) if recs else empty_df(REC_COLS)

    # 15. Portfolio details (без служебных колонок)
    df_details = df_pos.drop(columns=["_clean_value", "_pnl_pct", "_holding_days"], errors="ignore")
    df_details = df_details[DETAIL_COLS]

    return snapshot, df_rec, df_out, df_details


# ============================================================================
# Точка входа
# ============================================================================
def main(argv: list = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Module 3: оценка портфеля (изолированный запуск)")
    ap.add_argument("--date", default="")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else project_root / args.config
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir

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

    diagnostics = []

    try:
        # Загружаем карты типов бумаг ДО проверки файлов, чтобы deals_to_holdings
        # корректно определял asset_type для облигаций с SECID-тикером (SU...)
        bond_secids, stock_secids = load_asset_type_maps(data_dir, logger)

        holdings_path, created = check_required_files(data_dir, logger, bond_secids, stock_secids)
        if created:
            diagnostics.append({"code": "holdings_autogenerated", "details": str(holdings_path)})

        snapshot, rebalance, outsiders, portfolio_details = evaluate_portfolio(
            date, config, data_dir, output_dir, logger, diagnostics
        )

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
            logger.info(f"  Bonds by bucket: core_ofz={row.get('core_ofz_share_pct', 0):.2f}%, "
                        f"long_ofz={row.get('long_ofz_share_pct', 0):.2f}%, "
                        f"corp={row.get('corp_share_pct', 0):.2f}%")
        
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