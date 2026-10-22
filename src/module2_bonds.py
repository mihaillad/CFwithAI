#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 2. Рейтинг облигаций (ОФЗ + корпораты).

Автономный запуск (для отладки):
    python -m src.module2_bonds
    # или
    python src/module2_bonds.py

При запуске как __main__ автоматически:
  - создаёт data/raw/ и data/output/
  - загружает справочник облигаций с MOEX ISS -> bonds_OFZ.csv, bonds_CORPS.csv
  - загружает цены облигаций с MOEX ISS -> bond_prices.csv
  - выполняет rank_bonds() и сохраняет:
      ofz_ranking.csv, corp_ranking.csv, bond_allocation.csv
  - обновляет матрицу субпортфелей portfolio_units.csv
    (колонки "ОФЗ" и "Корпораты")
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# ============================================================================
# Константы источников данных
# ============================================================================
MOEX_BONDS_URL = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"
HTTP_TIMEOUT = 30
MAX_AGE_BONDS_SEC = 86400   # 24 часа для справочника облигаций
MAX_AGE_BOND_PRICES = 3600  # 1 час для текущих цен облигаций

# ============================================================================
# Пути и константы по умолчанию
# ============================================================================
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")

# Имена колонок субпортфелей в матрице (заполняются Module2)
OFZ_SUBPORTFOLIO_COL = "ОФЗ"
CORP_SUBPORTFOLIO_COL = "Корпораты"

# Значение units для каждой eligible облигации (целое, фиксированное)
BOND_UNITS_VALUE = 1


# ============================================================================
# Встроенные утилиты (для автономного запуска)
# ============================================================================

def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    """Логгер: файл output/run_<date>.log + stdout. Режим 'w' — идемпотентность."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{date}.log"
    logger = logging.getLogger("module2")
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
        "bond_sleeve_pct": 35.0,
        "ofz_core_share_pct": 40.0,
        "short_corp_share_pct": 60.0,
        "ofz_core_min_years": 2.5,
        "ofz_core_max_years": 3.0,
        "short_corp_max_years": 1.0,
        "min_credit_rating": "A-",
        "issuer_limit_pct": 3.0,
        "bond_min_volume_rub": 1_000_000.0,
        "bond_min_spread_pct": 2.0,
        "bond_spread_basis": "gross",
        "tax_rate_pct": 13.0,
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


def file_age(path: Path) -> float:
    """Возвращает возраст файла в секундах. Если файла нет — бесконечность."""
    return time.time() - path.stat().st_mtime if path.exists() else float('inf')


def is_fresh(path: Path, max_age: float) -> bool:
    """Проверяет, что файл существует и его возраст не превышает max_age секунд."""
    return file_age(path) <= max_age


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
    asset_type: str,  # 'stock' или 'bond'
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Обновляет (или создаёт) колонку sub-портфеля в матрице.
    - Если SECID уже есть — перезаписывает значение.
    - Если SECID новый — добавляет строку с указанным asset_type.
    - Строки, где сумма по всем sub-портфелям = 0, удаляются.
    """
    df = matrix.copy()

    # Гарантируем наличие колонки
    if col_name not in df.columns:
        df[col_name] = 0

    # Обновляем/добавляем значения
    for secid, units in values.items():
        secid_s = str(secid).strip().upper()
        mask = df["SECID"].astype(str).str.upper() == secid_s
        if mask.any():
            df.loc[mask, col_name] = units
            # Если asset_type не задан — проставляем
            if "asset_type" in df.columns and df.loc[mask, "asset_type"].isna().any():
                df.loc[mask, "asset_type"] = asset_type
        else:
            new_row = pd.DataFrame([{
                "SECID": secid_s,
                "asset_type": asset_type,
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
    df = df.sort_values("SECID").reset_index(drop=True)
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
# Загрузка данных с MOEX ISS
# ============================================================================

def fetch_and_save_bonds_universe(data_dir: Path, logger: logging.Logger) -> None:
    """
    Загружает справочник облигаций с MOEX ISS и разделяет на ОФЗ и Корпораты.
    Обновляет файлы, если они отсутствуют или устарели (> 24 часов).
    """
    ofz_path = data_dir / "bonds_OFZ.csv"
    corps_path = data_dir / "bonds_CORPS.csv"

    if is_fresh(ofz_path, MAX_AGE_BONDS_SEC) and is_fresh(corps_path, MAX_AGE_BONDS_SEC):
        logger.info("Справочники облигаций актуальны (возраст < 24 ч).")
        return

    logger.info("Обновление справочников облигаций с MOEX ISS...")
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "securities.columns": "SECID,ISIN,SHORTNAME,NAME,LOTSIZE,BOARDID,MATDATE,COUPONPERCENT,FACEVALUE"
    }
    try:
        r = requests.get(MOEX_BONDS_URL, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error(f"Ошибка загрузки справочника облигаций: {e}")
        if not ofz_path.exists() or not corps_path.exists():
            raise RuntimeError("Не удалось загрузить справочник облигаций, и локальные файлы отсутствуют.")
        logger.warning("Используются устаревшие локальные файлы справочников.")
        return

    if "securities" not in payload or "data" not in payload["securities"]:
        raise RuntimeError("Некорректный ответ MOEX ISS: нет блока securities.data")

    cols = payload["securities"]["columns"]
    df = pd.DataFrame(payload["securities"]["data"], columns=cols)

    # Приведение типов
    df["LOTSIZE"] = pd.to_numeric(df["LOTSIZE"], errors="coerce").fillna(1).astype(int)
    df["FACEVALUE"] = pd.to_numeric(df["FACEVALUE"], errors="coerce").fillna(1000.0)
    df["COUPONPERCENT"] = pd.to_numeric(df["COUPONPERCENT"], errors="coerce").fillna(0.0)
    df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce").dt.strftime("%Y-%m-%d")

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Разделение на ОФЗ и Корпораты
    is_ofz = df["BOARDID"].str.startswith("TQOB", na=False) | df["SHORTNAME"].str.contains("ОФЗ", na=False)

    df_ofz = df[is_ofz].copy()
    df_corps = df[~is_ofz].copy()

    # Формирование схемы для OFZ (согласно ТЗ §1.7)
    ofz_res = pd.DataFrame({
        "ISIN": df_ofz["ISIN"],
        "name": df_ofz["SHORTNAME"],
        "maturity_date": df_ofz["MATDATE"],
        "offer_date": pd.NA,
        "coupon_type": "fixed",
        "coupon_rate_pct": df_ofz["COUPONPERCENT"],
        "nominal": df_ofz["FACEVALUE"],
        "lot_size": df_ofz["LOTSIZE"],
        "board": df_ofz["BOARDID"],
        "lot_as_of": today_str
    })

    # Формирование схемы для CORPS (согласно ТЗ §1.8)
    corps_res = pd.DataFrame({
        "ISIN": df_corps["ISIN"],
        "name": df_corps["SHORTNAME"],
        "issuer": df_corps["SHORTNAME"],
        "issuer_group": df_corps["SHORTNAME"],
        "maturity_date": df_corps["MATDATE"],
        "offer_date": pd.NA,
        "coupon_type": "fixed",
        "coupon_rate_pct": df_corps["COUPONPERCENT"],
        "nominal": df_corps["FACEVALUE"],
        "credit_rating": "AAA",  # ISS не отдаёт рейтинг, ставим дефолт
        "rating_agency": "Unknown",
        "lot_size": df_corps["LOTSIZE"],
        "board": df_corps["BOARDID"],
        "lot_as_of": today_str
    })

    # Дедупликация по ISIN
    ofz_res = ofz_res.drop_duplicates(subset=["ISIN"], keep="first").sort_values("ISIN").reset_index(drop=True)
    corps_res = corps_res.drop_duplicates(subset=["ISIN"], keep="first").sort_values("ISIN").reset_index(drop=True)

    save_csv(ofz_res, ofz_path)
    save_csv(corps_res, corps_path)
    logger.info(f"Справочники обновлены: ОФЗ={len(ofz_res)}, Корпораты={len(corps_res)}")


def fetch_and_save_bond_prices(data_dir: Path, logger: logging.Logger) -> None:
    """
    Загружает текущие цены облигаций с MOEX ISS.
    Обновляет файл, если он отсутствует или устарел (> 1 часа).
    """
    prices_path = data_dir / "bond_prices.csv"

    if is_fresh(prices_path, MAX_AGE_BOND_PRICES):
        logger.info("Цены облигаций актуальны (возраст < 1 ч).")
        return

    logger.info("Обновление цен облигаций с MOEX ISS...")
    params = {
        "iss.meta": "off",
        "iss.only": "marketdata,securities",
        "marketdata.columns": "SECID,BOARDID,LCURRENTPRICE,LAST,LCLOSEPRICE,YIELD,DURATION,VOLTODAY",
        "securities.columns": "SECID,ISIN"
    }
    try:
        r = requests.get(MOEX_BONDS_URL, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error(f"Ошибка загрузки цен облигаций: {e}")
        if not prices_path.exists():
            raise RuntimeError("Не удалось загрузить цены облигаций, и локальный файл отсутствует.")
        logger.warning("Используются устаревшие локальные цены.")
        return

    if "marketdata" not in payload or "data" not in payload["marketdata"]:
        raise RuntimeError("Некорректный ответ MOEX ISS: нет блока marketdata.data")

    md_cols = payload["marketdata"]["columns"]
    md_df = pd.DataFrame(payload["marketdata"]["data"], columns=md_cols)

    sec_cols = payload.get("securities", {}).get("columns", [])
    sec_df = pd.DataFrame(payload.get("securities", {}).get("data", []),
                          columns=sec_cols) if "securities" in payload else pd.DataFrame(columns=["SECID", "ISIN"])

    # Приведение типов числовых колонок
    for col in ["LCURRENTPRICE", "LAST", "LCLOSEPRICE", "YIELD", "DURATION", "VOLTODAY"]:
        if col in md_df.columns:
            md_df[col] = pd.to_numeric(md_df[col], errors="coerce")

    # Выбор цены (приоритет: LCURRENTPRICE -> LAST -> LCLOSEPRICE)
    md_df["price_pct"] = np.nan
    for col in ["LCURRENTPRICE", "LAST", "LCLOSEPRICE"]:
        if col in md_df.columns:
            mask = md_df["price_pct"].isna() & md_df[col].notna() & (md_df[col] > 0)
            md_df.loc[mask, "price_pct"] = md_df.loc[mask, col]

    # Слияние с ISIN
    if not sec_df.empty and "ISIN" in sec_df.columns:
        isin_map = sec_df[["SECID", "ISIN"]].drop_duplicates("SECID", keep="first")
        md_df = md_df.merge(isin_map, on="SECID", how="left")
    else:
        md_df["ISIN"] = np.nan

    # Конвертируем объём из штук в рубли
    nominal_default = 1000.0
    volume_qty = md_df["VOLTODAY"].fillna(0.0)
    volume_rub = volume_qty * (md_df["price_pct"].fillna(0.0) / 100.0 * nominal_default)

    today_str = datetime.now().strftime("%Y-%m-%d")

    res_df = pd.DataFrame({
        "date": today_str,
        "ISIN": md_df["ISIN"],
        "price_pct": md_df["price_pct"],
        "ytm_pct": md_df["YIELD"],
        "duration_years": md_df["DURATION"],
        "volume_rub": volume_rub
    })

    # Очистка от записей без ISIN и без цены
    res_df = res_df.dropna(subset=["ISIN", "price_pct"])
    res_df = res_df.drop_duplicates(subset=["ISIN"], keep="first").sort_values("ISIN").reset_index(drop=True)

    save_csv(res_df, prices_path)
    logger.info(f"Цены облигаций обновлены: {len(res_df)} записей")


# ============================================================================
# Проверка входных файлов
# ============================================================================

def check_required_files(data_dir: Path, logger: logging.Logger) -> None:
    """
    Проверяет наличие и актуальность критичных файлов.
    При отсутствии или устаревании автоматически загружает их с MOEX ISS.
    """
    fetch_and_save_bonds_universe(data_dir, logger)
    fetch_and_save_bond_prices(data_dir, logger)

    rate_path = data_dir / "risk_free_rate.csv"
    if not rate_path.exists():
        logger.warning("risk_free_rate.csv не найден. Убедитесь, что module1_stocks.py был запущен первым.")

    logger.info("Все необходимые данные для Module 2 проверены и готовы.")


# ============================================================================
# Основная функция модуля
# ============================================================================

def rank_bonds(date: str, config: dict, data_dir: Path,
               logger: logging.Logger, diagnostics: list):
    """
    Рейтинг облигаций: фильтрация по спреду, рейтингу, сроку и распределение по бакетам.
    В конце обновляет матрицу субпортфелей portfolio_units.csv
    (колонки "ОФЗ" и "Корпораты").
    """
    date_dt = pd.to_datetime(date)

    # Загрузка входных файлов
    ofz = pd.read_csv(data_dir / 'bonds_OFZ.csv', sep=';')
    corps = pd.read_csv(data_dir / 'bonds_CORPS.csv', sep=';')

    # T-17: Дубликаты ISIN между файлами
    common_isins = set(ofz['ISIN']).intersection(set(corps['ISIN']))
    if common_isins:
        logger.warning(f"isin_duplicate_between_files: {common_isins}")
        diagnostics.append({'code': 'isin_duplicate_between_files', 'details': str(common_isins)})
        corps = corps[~corps['ISIN'].isin(common_isins)]

    # Объединение в единый universe
    bonds = pd.concat([
        ofz.assign(bond_type='ofz', source_file='bonds_OFZ', issuer='RF'),
        corps.assign(bond_type='corporate', source_file='bonds_CORPS')
    ], ignore_index=True)

    # Загрузка ключевой ставки
    rate_path = data_dir / 'risk_free_rate.csv'
    if rate_path.exists():
        cbr = pd.read_csv(rate_path)
        cbr['date'] = pd.to_datetime(cbr['date'])
        cbr_rate = cbr[cbr['date'] <= date_dt].sort_values('date').iloc[-1]['rate_pct']
    else:
        logger.error("risk_free_rate.csv не найден. Невозможно рассчитать спред.")
        raise RuntimeError("risk_free_rate.csv не найден")

    # Загрузка цен облигаций
    prices_path = data_dir / 'bond_prices.csv'
    if prices_path.exists():
        prices = pd.read_csv(prices_path, sep=';')
        prices['date'] = pd.to_datetime(prices['date'])
        latest_prices = prices[prices['date'] <= date_dt].sort_values('date').groupby('ISIN').tail(1).set_index('ISIN')
    else:
        logger.warning("bond_prices.csv не найден. Все облигации будут excluded.")
        latest_prices = pd.DataFrame()

    # Рейтинг и фильтрация
    results = []
    rating_order = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7, 'BBB+': 8, 'BBB': 9}
    min_rating_val = rating_order.get(config['min_credit_rating'], 99)

    for _, row in bonds.iterrows():
        isin = row['ISIN']

        # Контроль: проверка на пустую дату погашения
        mat_date = pd.to_datetime(row.get('maturity_date'), errors='coerce')
        if pd.isna(mat_date):
            results.append({
                'date': date, 'ISIN': isin, 'name': row.get('name', ''),
                'issuer': row.get('issuer', 'RF'),
                'bond_type': row['bond_type'], 'source_file': row['source_file'],
                'maturity_date': row.get('maturity_date', ''), 'days_to_maturity': 0,
                'coupon_type': row.get('coupon_type', ''), 'coupon_rate_pct': row.get('coupon_rate_pct', 0.0),
                'price_pct': np.nan, 'ytm_gross_pct': np.nan, 'ytm_net_pct': np.nan,
                'spread_to_cbr_pct': np.nan, 'duration_years': np.nan,
                'credit_rating': row.get('credit_rating', 'AAA'),
                'bucket': 'excluded', 'eligible': False, 'exclusion_reason': 'no_maturity_date',
                'lot_size': int(row.get('lot_size', 1)) if pd.notna(row.get('lot_size')) else 1,
                'target_weight_sleeve_pct': 0.0, 'target_weight_portfolio_pct': 0.0
            })
            continue

        if isin not in latest_prices.index:
            results.append({
                'date': date, 'ISIN': isin, 'name': row.get('name', ''), 'issuer': row.get('issuer', 'RF'),
                'bond_type': row['bond_type'], 'source_file': row['source_file'],
                'maturity_date': row['maturity_date'], 'days_to_maturity': 0,
                'coupon_type': row.get('coupon_type', ''), 'coupon_rate_pct': row.get('coupon_rate_pct', 0.0),
                'price_pct': np.nan, 'ytm_gross_pct': np.nan, 'ytm_net_pct': np.nan,
                'spread_to_cbr_pct': np.nan, 'duration_years': np.nan,
                'credit_rating': row.get('credit_rating', 'AAA'),
                'bucket': 'excluded', 'eligible': False, 'exclusion_reason': 'no_price',
                'lot_size': int(row.get('lot_size', 1)) if pd.notna(row.get('lot_size')) else 1,
                'target_weight_sleeve_pct': 0.0, 'target_weight_portfolio_pct': 0.0
            })
            continue

        p_row = latest_prices.loc[isin]
        years_to_mat = (mat_date - date_dt).days / 365.0

        bucket = 'excluded'
        exclusion_reason = ''
        eligible = True

        # Фильтрация по сроку
        if row['bond_type'] == 'ofz':
            if config['ofz_core_min_years'] <= years_to_mat <= config['ofz_core_max_years']:
                bucket = 'core_ofz'
            else:
                exclusion_reason = 'maturity_out_of_range'
                eligible = False
        else:
            if years_to_mat <= config['short_corp_max_years']:
                current_rating_val = rating_order.get(str(row.get('credit_rating', 'AAA')), 99)
                if current_rating_val > min_rating_val:
                    eligible = False
                    exclusion_reason = 'low_rating'
                else:
                    bucket = 'short_corp'
            else:
                exclusion_reason = 'maturity_out_of_range'
                eligible = False

        # Проверка YTM
        if pd.isna(p_row['ytm_pct']):
            eligible = False
            exclusion_reason = 'no_ytm'

        # Проверка объёма
        if p_row['volume_rub'] < config['bond_min_volume_rub']:
            eligible = False
            exclusion_reason = 'low_volume'

        # Расчёт спреда
        ytm_gross = p_row['ytm_pct']
        spread = ytm_gross - cbr_rate

        if eligible and spread < config['bond_min_spread_pct']:
            eligible = False
            exclusion_reason = 'spread'

        lot_size = int(row.get('lot_size', 1)) if pd.notna(row.get('lot_size')) else 1
        if pd.isna(row.get('lot_size')):
            diagnostics.append({'code': 'lot_missing', 'details': f"ISIN {isin}"})

        results.append({
            'date': date, 'ISIN': isin, 'name': row['name'], 'issuer': row.get('issuer', 'RF'),
            'bond_type': row['bond_type'], 'source_file': row['source_file'],
            'maturity_date': row['maturity_date'], 'days_to_maturity': int(years_to_mat * 365),
            'coupon_type': row['coupon_type'], 'coupon_rate_pct': row['coupon_rate_pct'],
            'price_pct': p_row['price_pct'],
            'ytm_gross_pct': ytm_gross, 'ytm_net_pct': ytm_gross * (1 - config['tax_rate_pct']/100),
            'spread_to_cbr_pct': spread, 'duration_years': p_row['duration_years'],
            'credit_rating': row.get('credit_rating', 'AAA'), 'bucket': bucket,
            'eligible': eligible, 'exclusion_reason': exclusion_reason, 'lot_size': lot_size,
            'target_weight_sleeve_pct': 0.0, 'target_weight_portfolio_pct': 0.0
        })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Разделяем на ОФЗ и корпораты
        df_ofz = df_res[df_res['bond_type'] == 'ofz'].copy()
        df_corp = df_res[df_res['bond_type'] == 'corporate'].copy()

        # Сортировка внутри каждого типа по YTM desc
        df_ofz = df_ofz.sort_values('ytm_gross_pct', ascending=False).reset_index(drop=True)
        df_corp = df_corp.sort_values('ytm_gross_pct', ascending=False).reset_index(drop=True)

        # Присвоение рангов (отдельно для ОФЗ и корпоратов)
        df_ofz['rank'] = df_ofz.index + 1
        df_corp['rank'] = df_corp.index + 1

        # Распределение весов
        for b, df_b in [('core_ofz', df_ofz), ('short_corp', df_corp)]:
            mask = (df_b['bucket'] == b) & (df_b['eligible'])
            if mask.sum() > 0:
                target_share = config['ofz_core_share_pct'] if b == 'core_ofz' else config['short_corp_share_pct']
                weight_per_pos = target_share / mask.sum()
                cap = config['issuer_limit_pct']
                df_b.loc[mask, 'target_weight_sleeve_pct'] = min(weight_per_pos, cap)
                df_b.loc[mask, 'target_weight_portfolio_pct'] = df_b.loc[mask, 'target_weight_sleeve_pct'] * (config['bond_sleeve_pct'] / 100)
    else:
        df_ofz = pd.DataFrame()
        df_corp = pd.DataFrame()

    # Bond allocation summary
    alloc = []
    for b, df_b in [('core_ofz', df_ofz), ('short_corp', df_corp)]:
        mask = (df_b['bucket'] == b) & (df_b['eligible']) if not df_b.empty else pd.Series([False])
        if mask.sum() > 0:
            alloc.append({
                'date': date, 'bucket': b,
                'target_share_pct': config['ofz_core_share_pct'] if b == 'core_ofz' else config['short_corp_share_pct'],
                'n_positions': int(mask.sum()),
                'avg_ytm_gross_pct': df_b.loc[mask, 'ytm_gross_pct'].mean(),
                'avg_ytm_net_pct': df_b.loc[mask, 'ytm_net_pct'].mean(),
                'total_weight_portfolio_pct': df_b.loc[mask, 'target_weight_portfolio_pct'].sum()
            })

    # === Диагностика: подсчёт причин исключения ===
    if not df_res.empty:
        exclusion_stats = df_res[~df_res['eligible']].groupby('exclusion_reason').size().to_dict()
        logger.info("Диагностика исключений облигаций:")
        for reason, count in sorted(exclusion_stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")

        if 'spread_to_cbr_pct' in df_res.columns:
            valid_spreads = df_res['spread_to_cbr_pct'].dropna()
            if not valid_spreads.empty:
                logger.info(f"Спреды: min={valid_spreads.min():.2f}%, max={valid_spreads.max():.2f}%, mean={valid_spreads.mean():.2f}%")

    # ------------------------------------------------------------------
    # Обновление матрицы субпортфелей (колонки "ОФЗ" и "Корпораты")
    # ------------------------------------------------------------------
    units_path = data_dir / "portfolio_units.csv"
    try:
        matrix = load_portfolio_units(units_path, logger)

        # Собираем eligible облигации по бакетам
        ofz_values = {}
        corp_values = {}

        if not df_ofz.empty:
            eligible_ofz = df_ofz[(df_ofz['bucket'] == 'core_ofz') & (df_ofz['eligible'])]
            for _, row in eligible_ofz.iterrows():
                isin = str(row["ISIN"]).strip().upper()
                ofz_values[isin] = BOND_UNITS_VALUE

        if not df_corp.empty:
            eligible_corp = df_corp[(df_corp['bucket'] == 'short_corp') & (df_corp['eligible'])]
            for _, row in eligible_corp.iterrows():
                isin = str(row["ISIN"]).strip().upper()
                corp_values[isin] = BOND_UNITS_VALUE

        # Обновляем колонки матрицы (каждую отдельно, чтобы не затирать ручные правки)
        matrix = update_subportfolio_column(matrix, OFZ_SUBPORTFOLIO_COL, ofz_values, "bond", logger)
        matrix = update_subportfolio_column(matrix, CORP_SUBPORTFOLIO_COL, corp_values, "bond", logger)

        save_portfolio_units(matrix, units_path)
        logger.info(
            f"Матрица обновлена: {len(matrix)} строк, "
            f"колонок-субпортфелей: {len([c for c in matrix.columns if c not in ('SECID','asset_type')])}, "
            f"заполнено в '{OFZ_SUBPORTFOLIO_COL}': {len(ofz_values)}, "
            f"в '{CORP_SUBPORTFOLIO_COL}': {len(corp_values)}"
        )
    except Exception as e:
        logger.error(f"Не удалось обновить матрицу субпортфелей: {e}")
        diagnostics.append({"code": "portfolio_units_update_failed", "details": str(e)})

    return df_ofz, df_corp, pd.DataFrame(alloc)


# ============================================================================
# Точка входа для изолированного запуска
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """CLI для отладки module2_bonds.py."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Module 2: рейтинг облигаций (изолированный запуск)"
    )
    ap.add_argument("--date", default="",
                    help="Дата расчёта YYYY-MM-DD (по умолчанию — из config.yaml или today)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="Путь к config.yaml")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="Папка с входными CSV (data/raw)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Папка для выходных CSV (data/output)")
    args = ap.parse_args(argv)

    # Корневая директория проекта
    project_root = Path(__file__).resolve().parent.parent
    config_path = (Path(args.config) if Path(args.config).is_absolute()
                   else project_root / args.config)
    data_dir = (Path(args.data_dir) if Path(args.data_dir).is_absolute()
                else project_root / args.data_dir)
    output_dir = (Path(args.output_dir) if Path(args.output_dir).is_absolute()
                  else project_root / args.output_dir)

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
    logger.info(f"=== Module 2 start: date={date}, data_dir={data_dir}, output_dir={output_dir} ===")

    diagnostics: list[dict] = []

    try:
        # Проверка входных файлов
        check_required_files(data_dir, logger)

        # Выполнение рейтинга
        ofz_ranking, corp_ranking, bond_allocation = rank_bonds(
            date, config, data_dir, logger, diagnostics
        )

        # Сохранение результатов (два отдельных рейтинга)
        save_csv(ofz_ranking, output_dir / "ofz_ranking.csv")
        save_csv(corp_ranking, output_dir / "corp_ranking.csv")
        save_csv(bond_allocation, output_dir / "bond_allocation.csv")

        logger.info(f"ofz_ranking: {len(ofz_ranking)} строк")
        logger.info(f"corp_ranking: {len(corp_ranking)} строк")
        logger.info(f"bond_allocation: {len(bond_allocation)} строк")

        if not ofz_ranking.empty:
            eligible_ofz = ofz_ranking[ofz_ranking['eligible']]
            logger.info(f"Eligible OFZ: {len(eligible_ofz)}")

        if not corp_ranking.empty:
            eligible_corp = corp_ranking[corp_ranking['eligible']]
            logger.info(f"Eligible Corporate: {len(eligible_corp)}")

        # diagnostics
        if diagnostics:
            save_csv(pd.DataFrame(diagnostics), output_dir / "diagnostics.csv")
            logger.info(f"diagnostics: {len(diagnostics)} записей")

        logger.info("=== Module 2 finished successfully ===")
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