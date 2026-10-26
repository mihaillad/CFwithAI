#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 1. Рейтинг акций по дивидендной доходности.

Автономный запуск (для отладки):
    python -m src.module1_stocks
    # или
    python src/module1_stocks.py

При запуске как __main__ автоматически:
  - создаёт data/raw/ и data/output/
  - загружает календарь БКС -> divs_data.csv
  - загружает справочник акций с MOEX ISS -> stocks_universe.csv
  - загружает цены MOEX ISS -> stock_prices.csv
  - загружает ставку ЦБ -> risk_free_rate.csv
  - создаёт пустой stock_fundamentals.csv (если нет)
  - обновляет матрицу субпортфелей portfolio_units.csv (колонка "Дивиденды")
  - выполняет rank_stocks() и сохраняет stock_ranking.csv + stock_excluded.csv
"""
from __future__ import annotations

import csv
import logging
import math
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# ============================================================================
# Константы источников данных
# ============================================================================
BCS_URL = "https://be.broker.ru/bcsexpress-partners-gateway/express-divcalendar/api/v2/dividends"
BCS_PARAMS = {"isActual": "true", "limit": "500", "order": "2",
              "sorting": "0", "isForeign": "false"}
MOEX_PRICES_URL = "http://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
CBR_URL = "http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
HTTP_TIMEOUT = 30

# ============================================================================
# Пути и константы по умолчанию
# ============================================================================
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")

# Критичные файлы (без них модуль не может работать)
CRITICAL_RAW_FILES = [
    "stocks_universe.csv",
]

# Опциональные файлы (могут быть получены из внешних источников)
OPTIONAL_RAW_FILES = [
    "divs_data.csv",
    "stock_prices.csv",
    "risk_free_rate.csv",
    "special_dividends_registry.csv",
    "stock_fundamentals.csv",
]

# ============================================================================
# Стоп-лист
# ============================================================================
STOP_LIST_PATH = DEFAULT_DATA_DIR / "stoplist.csv"


def load_stoplist(path: Path, date_dt: pd.Timestamp, logger: logging.Logger) -> set:
    """
    Загружает stoplist.csv, возвращает set активных SECID (stop_date >= date).
    Формат файла: SECID;stop_date;reason
    """
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
    logger.info(f"Стоп-лист: {len(result)} активных бумаг (из {len(df)} записей)")
    return result

# ============================================================================
# Матрица субпортфелей
# ============================================================================
PORTFOLIO_UNITS_PATH = DEFAULT_DATA_DIR / "portfolio_units.csv"
DIV_SUBPORTFOLIO_COL = "Дивиденды"  # Имя авто-колонки, которую заполняет Module1


# ============================================================================
# Встроенные утилиты (для автономного запуска, без common.py)
# ============================================================================

def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    """Настраивает логгер: файл output/run_<date>.log + stdout. Режим 'w' — идемпотентность."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{date}.log"
    logger = logging.getLogger("module1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
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
        "top_n_stocks": 16,
        "weight_top1_pct": 10.0,
        "weight_step_pct": 0.5,
        "stock_sleeve_pct": 65.0,
        "d_min_days": 5,
        "min_dy_gross_pct": 5.0,
        "payout_warn_pct": 90.0,
        "payout_max_pct": 100.0,
        "registry_missing_policy": "warn_and_uncertain",
        "registry_strict_mode": False,
    }
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(cfg)
    return defaults


def save_csv(df: pd.DataFrame, path: Path, sep: str = ";", float_format: str = "%.4f"):
    """Сохраняет DataFrame в CSV с заданным разделителем и точностью float."""
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
# Работа с матрицей субпортфелей
# ============================================================================

def load_portfolio_units(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Загружает матрицу субпортфелей. Если файла нет — возвращает пустой DataFrame
    с обязательными колонками SECID и asset_type.
    """
    if not path.exists():
        logger.info(f"Матрица {path.name} не найдена — будет создана")
        return pd.DataFrame(columns=["SECID", "asset_type"])
    df = read_csv_flex(path)
    if "SECID" not in df.columns:
        logger.warning(f"В {path.name} нет колонки SECID — создаю новую матрицу")
        return pd.DataFrame(columns=["SECID", "asset_type"])
    # Приводим числовые колонки (все кроме SECID/asset_type) к numeric,
    # пустые значения -> 0
    for c in df.columns:
        if c not in ("SECID", "asset_type"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def update_subportfolio_column(
    matrix: pd.DataFrame,
    col_name: str,
    values: dict,  # {SECID: units}
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Обновляет (или создаёт) колонку sub-портфеля в матрице.
    - Если SECID уже есть — перезаписывает значение.
    - Если SECID новый — добавляет строку (asset_type='stock' для Module1).
    - Строки, где сумма по всем sub-портфелям = 0, удаляются.
    """
    df = matrix.copy()

    # Гарантируем наличие колонки
    if col_name not in df.columns:
        df[col_name] = 0

    # Обнуляем старые значения в колонке
    df[col_name] = 0
    
    # Обновляем/добавляем значения
    for secid, units in values.items():
        secid_s = str(secid).strip().upper()
        mask = df["SECID"].astype(str).str.upper() == secid_s
        if mask.any():
            df.loc[mask, col_name] = units
        else:
            new_row = pd.DataFrame([{
                "SECID": secid_s,
                "asset_type": "stock",
                col_name: units,
            }])
            df = pd.concat([df, new_row], ignore_index=True)

    # Заполняем NaN в остальных строках нулями (для новых колонок)
    for c in df.columns:
        if c not in ("SECID", "asset_type"):
            df[c] = df[c].fillna(0)

    # Удаляем строки с total_units_row = 0
    sub_cols = [c for c in df.columns if c not in ("SECID", "asset_type")]
    if sub_cols:
        df["_total"] = df[sub_cols].sum(axis=1)
        removed = int((df["_total"] == 0).sum())
        df = df[df["_total"] > 0].drop(columns=["_total"]).reset_index(drop=True)
        if removed:
            logger.info(f"Матрица: удалено {removed} строк с нулевыми units")

    # Сортировка по SECID
    df = df.sort_values(["asset_type","SECID"], ascending=[False,True ]).reset_index(drop=True)
    return df


def save_portfolio_units(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет матрицу с разделителем ; и целыми числами для units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in df.columns if c not in ("SECID", "asset_type")]
    out = df.copy()
    for c in num_cols:
        out[c] = out[c].round().astype("Int64")
    out.to_csv(path, index=False, sep=";", encoding="utf-8")


# ============================================================================
# Загрузка данных из внешних источников
# ============================================================================

def fetch_bcs_calendar() -> dict:
    r = requests.get(BCS_URL, params=BCS_PARAMS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Некорректный ответ БКС")
    return payload


def bcs_to_divs_data(payload: dict) -> pd.DataFrame:
    rows = []
    for rec in payload.get("data", []):
        if rec.get("isActual") is False:
            continue
        secid = rec.get("securCode")
        record_date = rec.get("recordDate")
        div_sum = rec.get("size")
        if not (secid and record_date and div_sum):
            continue
        try:
            rows.append({
                "SECID": str(secid).strip(),
                "close_date": pd.to_datetime(str(record_date)[:10]),
                "div_sum": float(div_sum),
            })
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.groupby(["SECID", "close_date"], as_index=False)["div_sum"].sum()
    return df


def _read_divs_csv(path: Path) -> pd.DataFrame:
    """
    Читает divs_data.csv с автоопределением разделителя (',' или ';').
    """
    required_cols = {"SECID", "close_date", "div_sum"}

    df = pd.read_csv(path, sep=";")
    if required_cols.issubset(df.columns):
        return df

    df = pd.read_csv(path, sep=",")
    if required_cols.issubset(df.columns):
        return df

    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
    except csv.Error:
        dialect = None

    df = pd.read_csv(path, sep=dialect.delimiter if dialect else ",")
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"В {path.name} отсутствуют обязательные колонки: "
            f"{required_cols - set(df.columns)}. "
            f"Найденные колонки: {list(df.columns)}"
        )
    return df


def update_divs_data(divs_path: Path, bcs_df: pd.DataFrame,
                     today: pd.Timestamp) -> pd.DataFrame:
    """
    Обновляет divs_data.csv:
    - из старого файла оставляем только записи <= today (история фактов);
    - будущие записи из old удаляем (могли быть устаревшими прогнозами);
    - добавляем свежие данные из БКС;
    - сохраняем результат с разделителем ';' согласно ТЗ §0.4.
    """
    if divs_path.exists():
        old = _read_divs_csv(divs_path)
        old["close_date"] = pd.to_datetime(old["close_date"])
        old["div_sum"] = pd.to_numeric(old["div_sum"], errors="coerce")
        old = old.dropna(subset=["SECID", "close_date", "div_sum"])
        old = old[old["close_date"] <= today]
    else:
        old = pd.DataFrame(columns=["SECID", "close_date", "div_sum"])

    merged = pd.concat([old, bcs_df], ignore_index=True)
    merged = merged.sort_values("close_date").drop_duplicates(
        subset=["SECID", "close_date"], keep="last"
    )
    merged = merged.sort_values(["SECID", "close_date"]).reset_index(drop=True)

    save_csv(merged, divs_path, sep=";", float_format="%.9f")
    return merged


def fetch_moex_prices() -> pd.DataFrame:
    params = {
        "iss.meta": "off",
        "iss.only": "marketdata,securities",
        "marketdata.columns": "SECID,LCURRENTPRICE,LAST,LCLOSEPRICE",
        "securities.columns": "SECID,ISIN",
    }
    r = requests.get(MOEX_PRICES_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if "marketdata" not in payload or "data" not in payload["marketdata"]:
        raise ValueError("В ответе ISS нет блока marketdata")

    md = pd.DataFrame(payload["marketdata"]["data"],
                      columns=payload["marketdata"]["columns"])
    for col in ["LCURRENTPRICE", "LAST", "LCLOSEPRICE"]:
        if col in md.columns:
            md[col] = pd.to_numeric(md[col], errors="coerce")

    md["close"] = np.nan
    for col in ["LCURRENTPRICE", "LAST", "LCLOSEPRICE"]:
        if col in md.columns:
            mask = md["close"].isna() & md[col].notna() & (md[col] > 0)
            md.loc[mask, "close"] = md.loc[mask, col]

    md = md[["SECID", "close"]].dropna(subset=["close"])
    md["date"] = datetime.now().strftime("%Y-%m-%d")
    md["volume_rub"] = 0.0
    return md[["date", "SECID", "close", "volume_rub"]]


def fetch_cbr_rate() -> pd.DataFrame:
    today = datetime.now()
    d_from = today - timedelta(days=370)
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
        '<KeyRate xmlns="http://web.cbr.ru/">'
        f"<fromDate>{d_from:%Y-%m-%d}T00:00:00</fromDate>"
        f"<ToDate>{today:%Y-%m-%d}T00:00:00</ToDate>"
        "</KeyRate></soap:Body></soap:Envelope>"
    )
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": '"http://web.cbr.ru/KeyRate"',
    }
    r = requests.post(CBR_URL, data=body.encode("utf-8"),
                      headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    rows = []
    for el in root.iter():
        leaves = {c.tag.split("}")[-1]: (c.text or "").strip()
                  for c in el if len(c) == 0}
        if len(leaves) < 2:
            continue
        date = next((leaves[k] for k in ("DT", "Date", "date") if k in leaves), None)
        rate = next((leaves[k] for k in ("Rate", "BaseRate", "Value") if k in leaves), None)
        if date and rate:
            try:
                rows.append({"date": date[:10],
                             "rate_pct": float(rate.replace(",", "."))})
            except ValueError:
                continue
    if not rows:
        raise ValueError("В ответе ЦБ не найдено пар дата/ставка")
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date")


def fetch_and_save_stocks_universe(data_dir: Path, logger: logging.Logger) -> None:
    """
    Загружает актуальный справочник акций с Мосбиржи (ISS) и сохраняет в stocks_universe.csv.
    """
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities.json"
    params = {
        "iss.meta": "off",
        "securities.columns": "SECID,SHORTNAME,LATNAME,LOTSIZE,ISIN,LISTLEVEL,BOARDID"
    }
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error(f"Не удалось загрузить справочник акций с MOEX ISS: {e}")
        raise RuntimeError(f"Ошибка загрузки stocks_universe: {e}")

    if "securities" not in payload or "data" not in payload["securities"]:
        raise RuntimeError("Некорректный ответ MOEX ISS: нет блока securities.data")

    cols = payload["securities"]["columns"]
    data = payload["securities"]["data"]
    df = pd.DataFrame(data, columns=cols)

    df["LOTSIZE"] = pd.to_numeric(df["LOTSIZE"], errors="coerce").fillna(1).astype(int)
    df["LISTLEVEL"] = pd.to_numeric(df["LISTLEVEL"], errors="coerce").fillna(-1)

    df = df.sort_values(by=["LISTLEVEL", "BOARDID"], ascending=[False, True])
    df_unique = df.drop_duplicates(subset=["SECID"], keep="first").copy()

    result = pd.DataFrame({
        "SECID": df_unique["SECID"],
        "ISIN": df_unique["ISIN"],
        "issuer": df_unique["SHORTNAME"],
        "name": df_unique["SHORTNAME"],
        "sector": "Unknown",
        "LOTSIZE": df_unique["LOTSIZE"],
        "board": df_unique["BOARDID"],
        "lot_as_of": datetime.now().strftime("%Y-%m-%d")
    })

    result = result.sort_values("SECID").reset_index(drop=True)
    out_path = data_dir / "stocks_universe.csv"
    save_csv(result, out_path)
    logger.info(f"Справочник акций обновлён: {len(result)} тикеров сохранено в {out_path.name}")


# ============================================================================
# Проверка входных файлов
# ============================================================================

def check_required_files(data_dir: Path, logger: logging.Logger) -> None:
    """
    Проверяет наличие критичных входных файлов.
    - Если stocks_universe.csv отсутствует, загружает его с MOEX ISS.
    - Если stock_fundamentals.csv отсутствует, создаёт пустой файл с требуемой схемой.
    """
    universe_path = data_dir / "stocks_universe.csv"
    if not universe_path.exists():
        logger.warning("stocks_universe.csv не найден. Загружаю актуальный справочник с MOEX ISS...")
        fetch_and_save_stocks_universe(data_dir, logger)

    fund_path = data_dir / "stock_fundamentals.csv"
    if not fund_path.exists():
        logger.info("stock_fundamentals.csv не найден. Создаю пустой файл с требуемой схемой...")
        empty_fund = pd.DataFrame(columns=[
            "SECID", "report_date", "period", "eps_ttm", "eps_prev_ttm",
            "eps_3y_ago", "net_income_ttm", "shares_outstanding",
            "dividend_policy_url", "notes"
        ])
        save_csv(empty_fund, fund_path)
        logger.info(f"Пустой {fund_path.name} успешно создан.")

    for f in OPTIONAL_RAW_FILES:
        if not (data_dir / f).exists():
            logger.info(f"Опциональный файл отсутствует: {f} (будет получен из внешнего источника или пропущен)")


# ============================================================================
# Генерация прогнозов дивидендов
# ============================================================================

def generate_forecast_dividends(divs_df: pd.DataFrame, today: pd.Timestamp,
                                match_days: int = 45) -> pd.DataFrame:
    """
    Генерирует прогнозные дивиденды на основе выплат за последние 365 дней.
    """
    if divs_df.empty:
        res = divs_df.copy()
        if "dps_source" not in res.columns:
            res["dps_source"] = "declared"
        return res

    df = divs_df.copy()
    if "dps_source" not in df.columns:
        df["dps_source"] = "declared"

    hist = df[
        (df["close_date"] <= today) &
        (df["close_date"] > today - pd.Timedelta(days=365)) &
        (df["div_sum"] > 0)
    ].copy()

    if hist.empty:
        return df

    fc = hist[["SECID", "close_date", "div_sum"]].copy()
    fc["close_date"] = fc["close_date"] + pd.Timedelta(days=365)
    fc = fc[fc["close_date"] > today].reset_index(drop=True)

    if fc.empty:
        return df

    declared = df[df["close_date"] > today][["SECID", "close_date"]].copy()

    if not declared.empty and not fc.empty:
        m = fc.reset_index().merge(
            declared.rename(columns={"close_date": "dec_date"}),
            on="SECID"
        )
        covered_idx = m.loc[
            (m["close_date"] - m["dec_date"]).dt.days.abs() <= match_days,
            "index"
        ].unique()
        fc = fc.drop(index=covered_idx)

    fc["dps_source"] = "forecast"

    result = pd.concat([df, fc], ignore_index=True)

    result["source_priority"] = result["dps_source"].map({"declared": 1, "forecast": 2})
    result = result.sort_values(["SECID", "close_date", "source_priority"]).drop_duplicates(
        subset=["SECID", "close_date"], keep="first"
    ).drop(columns=["source_priority"]).reset_index(drop=True)

    return result


# ============================================================================
# Основная функция модуля
# ============================================================================

def rank_stocks(date: str, config: dict, data_dir: Path,
                logger: logging.Logger, diagnostics: list):
    date_dt = pd.to_datetime(date)

    # Загрузка стоп-листа (применяется ко всему рейтингу)
    stoplist = load_stoplist(data_dir / "stoplist.csv", date_dt, logger)
    if stoplist:
        logger.info(f"Стоп-лист активен: {len(stoplist)} бумаг")

    check_required_files(data_dir, logger)
        
    # ------------------------------------------------------------------
    # Шаг 1. Обновление данных из внешних источников
    # ------------------------------------------------------------------

    try:
        bcs_payload = fetch_bcs_calendar()
        bcs_df = bcs_to_divs_data(bcs_payload)
        logger.info(f"BCS calendar loaded: {len(bcs_df)} records")
    except Exception as e:
        logger.error(f"BCS calendar fetch failed: {e}")
        bcs_df = pd.DataFrame(columns=['SECID', 'close_date', 'div_sum'])
        diagnostics.append({'code': 'bcs_fetch_failed', 'details': str(e)})

    divs_path = data_dir / 'divs_data.csv'
    divs_df = update_divs_data(divs_path, bcs_df, date_dt)
    logger.info(f"divs_data.csv updated: {len(divs_df)} records, "
                f"{divs_df['SECID'].nunique()} tickers")

    prices_path = data_dir / 'stock_prices.csv'
    try:
        prices_df = fetch_moex_prices()
        prices_df.to_csv(prices_path, index=False, sep=';', encoding='utf-8')
        logger.info(f"MOEX prices loaded: {len(prices_df)} records")
    except Exception as e:
        logger.error(f"MOEX prices fetch failed: {e}")
        diagnostics.append({'code': 'moex_fetch_failed', 'details': str(e)})
        if prices_path.exists():
            prices_df = pd.read_csv(prices_path, sep=';')
            logger.warning("Using cached stock_prices.csv")
        else:
            prices_df = pd.DataFrame(columns=['date', 'SECID', 'close', 'volume_rub'])

    rate_path = data_dir / 'risk_free_rate.csv'
    try:
        cbr_df = fetch_cbr_rate()
        cbr_df.to_csv(rate_path, index=False, encoding='utf-8')
        logger.info(f"CBR rate loaded: {len(cbr_df)} records")
    except Exception as e:
        logger.error(f"CBR rate fetch failed: {e}")
        diagnostics.append({'code': 'cbr_fetch_failed', 'details': str(e)})
        if rate_path.exists():
            cbr_df = pd.read_csv(rate_path)
            logger.warning("Using cached risk_free_rate.csv")
        else:
            cbr_df = pd.DataFrame(columns=['date', 'rate_pct'])

    # ------------------------------------------------------------------
    # Шаг 2. Загрузка остальных входных файлов
    # ------------------------------------------------------------------
    universe = pd.read_csv(data_dir / 'stocks_universe.csv', sep=';')

    reg_path = data_dir / 'special_dividends_registry.csv'
    registry = (pd.read_csv(reg_path, sep=';')
                if reg_path.exists() else pd.DataFrame())

    fund_path = data_dir / 'stock_fundamentals.csv'
    if fund_path.exists():
        fundamentals = pd.read_csv(fund_path, sep=';')
        if fundamentals.empty:
            logger.warning("stock_fundamentals.csv пустой — проверки eps_ttm/payout будут пропущены")
            fundamentals = pd.DataFrame()
    else:
        fundamentals = pd.DataFrame()

    skip_fund_checks = fundamentals.empty
    if skip_fund_checks:
        logger.info("Фундаментальные данные отсутствуют — стоп-лист по eps/payout отключён")

    prices_df['date'] = pd.to_datetime(prices_df['date'])
    latest_prices = (prices_df[prices_df['date'] <= date_dt]
                     .sort_values('date')
                     .groupby('SECID').tail(1)
                     .set_index('SECID'))

    divs_df['close_date'] = pd.to_datetime(divs_df['close_date'])

    # Генерация прогнозов на основе истории за 365 дней
    divs_df = generate_forecast_dividends(divs_df, date_dt, match_days=45)
    logger.info(f"divs_data после генерации прогнозов: {len(divs_df)} записей, "
                f"прогнозных: {(divs_df['dps_source'] == 'forecast').sum()}")

    bad_registry_secids = set()
    if not registry.empty and 'classification' in registry.columns:
        bad_registry_secids = set(
            registry[registry['classification'].isin(['special', 'uncertain'])]['SECID'].unique()
        )

    # ------------------------------------------------------------------
    # Шаг 3. Расчёт рейтинга: обходим только SECID с дивидендами
    # ------------------------------------------------------------------

    future_divs = divs_df[divs_df['close_date'] > date_dt]
    active_secids = future_divs['SECID'].unique().tolist()
    logger.info(f"Активных SECID с будущими дивидендами: {len(active_secids)}")

    universe_secids = set(universe['SECID'])
    without_divs = universe_secids - set(active_secids)
    if without_divs:
        logger.info(f"В universe {len(without_divs)} тикеров без дивидендов (пропущены)")
        diagnostics.append({
            'code': 'universe_without_dividends',
            'details': f"count={len(without_divs)}"
        })

    missing_in_universe = set(active_secids) - universe_secids
    if missing_in_universe:
        logger.warning(f"SECID в дивидендах, но нет в universe: {len(missing_in_universe)}")
        diagnostics.append({
            'code': 'secid_not_in_universe',
            'details': f"tickers={','.join(sorted(missing_in_universe)[:20])}"
        })

    universe_dict = universe.set_index('SECID').to_dict('index')

    results = []
    excluded = []

    for secid in active_secids:
        uni_row = universe_dict.get(secid, {})
        LOTSIZE = int(uni_row.get('LOTSIZE', 1)) if uni_row else 1

        if secid not in latest_prices.index:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_price', 'details': 'No recent price'})
            continue
        price = latest_prices.loc[secid, 'close']
        if pd.isna(price) or price <= 0:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_price', 'details': 'Price NaN or <= 0'})
            continue

        if secid in bad_registry_secids:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'registry_missing',
                             'details': 'Excluded by special_dividends_registry'})
            continue

        secid_divs = divs_df[divs_df['SECID'] == secid].copy()
        secid_divs["source_priority"] = secid_divs["dps_source"].map({"declared": 1, "forecast": 2}).fillna(3)
        secid_divs = secid_divs.sort_values(["close_date", "source_priority"])

        secid_divs = secid_divs.rename(columns={
            'close_date': 'record_date',
            'div_sum': 'amount_per_share'
        })

        future_secid_divs = secid_divs[secid_divs['record_date'] > date_dt]
        if future_secid_divs.empty:
            continue

        next_div = future_secid_divs.iloc[0]

        days_to_record = (next_div['record_date'] - date_dt).days
        if days_to_record < config['d_min_days']:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'too_close_to_record',
                             'details': f'd={days_to_record}'})
            continue

        dps = next_div['amount_per_share']
        dps_source = next_div.get('dps_source', 'declared')

        ay_pct = ((dps / price) / max(days_to_record, config['d_min_days'])
                  * 365 * 100)
        dy_gross_pct = (dps / price) * 100

        eps_ttm = np.nan
        payout_fwd = 0.0
        eps_yoy = np.nan
        eps_3y = np.nan

        if not skip_fund_checks:
            fund_row = (fundamentals[fundamentals['SECID'] == secid]
                        .sort_values('report_date'))
            if not fund_row.empty:
                last_fund = fund_row.iloc[-1]
                eps_ttm = last_fund.get('eps_ttm', np.nan)
                payout_fwd = last_fund.get('payout_fwd_pct', 0.0) or 0.0
                eps_yoy = last_fund.get('eps_growth_yoy_pct', np.nan)
                eps_3y = last_fund.get('eps_growth_3y_pct', np.nan)

        stop_flag = False
        stop_reasons = []
        if dy_gross_pct < config['min_dy_gross_pct']:
            stop_flag = True
            stop_reasons.append("low_dy")

        if not skip_fund_checks:
            if pd.isna(eps_ttm) or eps_ttm <= 0:
                stop_flag = True
                stop_reasons.append("eps_negative")
            if payout_fwd > config['payout_max_pct']:
                stop_flag = True
                stop_reasons.append("payout_max")
            if (payout_fwd > config['payout_warn_pct']
                    and pd.notna(eps_yoy) and eps_yoy < 0):
                stop_flag = True
                stop_reasons.append("payout_warn_no_growth")

        results.append({
            'date': date, 'SECID': secid,
            'name': uni_row.get('name', ''),
            'issuer': uni_row.get('issuer', ''),
            'sector': uni_row.get('sector', 'Unknown'),
            'price': price,
            'dps_reg_next': dps,
            'dps_source': dps_source,
            'record_date': next_div['record_date'].strftime('%Y-%m-%d'),
            'days_to_record': days_to_record,
            'ay_pct': ay_pct,
            'dy_gross_pct': dy_gross_pct,
            'payout_fwd_pct': payout_fwd,
            'eps_growth_yoy_pct': eps_yoy,
            'eps_growth_3y_pct': eps_3y,
            'stop_flag': stop_flag,
            'stop_reason': ";".join(stop_reasons),
            'LOTSIZE': LOTSIZE
        })

    # ------------------------------------------------------------------
    # Шаг 4. Фильтрация пар обычка/преф, сортировка и нормировка весов
    # ------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # ------------------------------------------------------------------
        # 4.1 Исключение по стоп-листу (до всех остальных фильтров)
        # ------------------------------------------------------------------
        if stoplist:
            before = len(df_res)
            stop_mask = df_res["SECID"].astype(str).str.upper().isin(stoplist)
            stopped = df_res[stop_mask]
            if not stopped.empty:
                for _, row in stopped.iterrows():
                    excluded.append({
                        'date': date,
                        'SECID': row["SECID"],
                        'reason': 'stop_list',
                        'details': 'Исключена по стоп-листу'
                    })
                df_res = df_res[~stop_mask].copy().reset_index(drop=True)
                logger.info(f"Стоп-лист: исключено {before - len(df_res)} бумаг")
                diagnostics.append({
                    'code': 'stop_list_applied',
                    'details': f"count={before - len(df_res)}"
                })
        # ------------------------------------------------------------------

        # Фильтрация пар обычка/преф: остаётся бумага с большей AY (при равенстве с большей DY)
        secs = set(df_res["SECID"])
        drop = []
        for s in secs:
            p = s + "P"
            if p in secs:
                a = df_res[df_res["SECID"] == s].iloc[0]
                b = df_res[df_res["SECID"] == p].iloc[0]
                drop.append(p if (a["ay_pct"], a["dy_gross_pct"]) >= (b["ay_pct"], b["dy_gross_pct"]) else s)

        if drop:
            logger.info(f"обычка и префы: убраны {', '.join(sorted(drop))}")
            for secid in drop:
                excluded.append({
                    'date': date,
                    'SECID': secid,
                    'reason': 'common_pref_pair',
                    'details': 'Filtered out in favor of better AY/CY in pair'
                })
            df_res = df_res[~df_res["SECID"].isin(drop)].copy()

        df_res = df_res.copy()
        df_res['_src_priority'] = df_res['dps_source'].map({'declared': 1, 'forecast': 2}).fillna(3)
        df_res = df_res.sort_values(
            ['_src_priority','ay_pct' ],
            ascending=[True, False]
        ).drop(columns=['_src_priority']).reset_index(drop=True)
        df_res['rank'] = df_res.index + 1

        top_n = min(config['top_n_stocks'], len(df_res))
        df_res['eligible_for_buy'] = False
        valid_top = df_res[~df_res['stop_flag']].head(top_n)

        weights = [config['weight_top1_pct'] - config['weight_step_pct'] * i
                   for i in range(len(valid_top))]
        sum_w = sum(weights)
        norm_weights = [(w / sum_w * 100) if sum_w > 0 else 0 for w in weights]

        df_res.loc[valid_top.index, 'eligible_for_buy'] = True
        df_res.loc[valid_top.index, 'target_weight_sleeve_pct'] = norm_weights
        df_res['target_weight_portfolio_pct'] = (
            df_res['target_weight_sleeve_pct'] * (config['stock_sleeve_pct'] / 100)
        )

    # ------------------------------------------------------------------
    # Шаг 5. Обновление матрицы субпортфелей (колонка "Дивиденды")
    # ------------------------------------------------------------------
    units_path = data_dir / "portfolio_units.csv"
    try:
        matrix = load_portfolio_units(units_path, logger)

        eligible = df_res[df_res.get("eligible_for_buy") == True].copy() \
            if "eligible_for_buy" in df_res.columns else pd.DataFrame()

        div_values = {}
        if not eligible.empty:
            for _, row in eligible.iterrows():
                secid = str(row["SECID"]).strip().upper()
                units = float(row.get("target_weight_sleeve_pct", 0) or 0)
                if units > 0:
                    div_values[secid] = units

        matrix = update_subportfolio_column(matrix, DIV_SUBPORTFOLIO_COL, div_values, logger)
        save_portfolio_units(matrix, units_path)
        logger.info(
            f"Матрица обновлена: {len(matrix)} строк, "
            f"колонок-субпортфелей: {len([c for c in matrix.columns if c not in ('SECID','asset_type')])}, "
            f"заполнено в '{DIV_SUBPORTFOLIO_COL}': {len(div_values)}"
        )
    except Exception as e:
        logger.error(f"Не удалось обновить матрицу субпортфелей: {e}")
        diagnostics.append({"code": "portfolio_units_update_failed", "details": str(e)})

    df_exc = pd.DataFrame(excluded)
    return df_res, df_exc


# ============================================================================
# Точка входа для изолированного запуска
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """CLI для отладки module1_stocks.py."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Module 1: рейтинг акций по дивидендной доходности (изолированный запуск)"
    )
    ap.add_argument("--date", default="",
                    help="Дата расчёта YYYY-MM-DD (по умолчанию — из config.yaml или today)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="Путь к config.yaml")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="Папка с входными CSV (data/raw)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Папка для выходных CSV (data/output)")
    ap.add_argument("--update-universe", action="store_true",
                    help="Принудительно перезагрузить stocks_universe.csv с MOEX ISS")
    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    config_path = (Path(args.config) if Path(args.config).is_absolute()
                   else project_root / args.config)
    data_dir = (Path(args.data_dir) if Path(args.data_dir).is_absolute()
                else project_root / args.data_dir)
    output_dir = (Path(args.output_dir) if Path(args.output_dir).is_absolute()
                  else project_root / args.output_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    date_from_config = config.get("date", "today")
    if not date_from_config or str(date_from_config).lower() in ("today", "auto", ""):
        date = args.date or datetime.now().strftime("%Y-%m-%d")
    else:
        date = args.date or str(date_from_config)

    logger = setup_logging(date, output_dir)
    logger.info(f"=== Module 1 start: date={date}, data_dir={data_dir}, output_dir={output_dir} ===")

    diagnostics: list[dict] = []

    if args.update_universe:
        logger.info("Принудительное обновление справочника акций...")
        fetch_and_save_stocks_universe(data_dir, logger)

    try:
        check_required_files(data_dir, logger)

        stock_ranking, stock_excluded = rank_stocks(date, config, data_dir, logger, diagnostics)

        save_csv(stock_ranking, output_dir / "stock_ranking.csv")
        save_csv(stock_excluded, output_dir / "stock_excluded.csv")

        logger.info(f"stock_ranking: {len(stock_ranking)} строк")
        logger.info(f"stock_excluded: {len(stock_excluded)} строк")
        if not stock_ranking.empty:
            top5 = stock_ranking.head(5)[["rank", "SECID", "ay_pct", "dy_gross_pct", "stop_flag"]]
            logger.info(f"Top-5:\n{top5.to_string(index=False)}")

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
            logger.info(f"diagnostics: {len(merged)} записей")

        logger.info("=== Module 1 finished successfully ===")
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