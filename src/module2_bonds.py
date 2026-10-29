#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 2. Рейтинг облигаций (core_ofz, long_ofz, corp).

Автономный запуск:
    python src/module2_bonds.py
    python src/module2_bonds.py --date 2026-10-24
"""
from __future__ import annotations

import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# ============================================================================
# Константы
# ============================================================================
MOEX_BONDS_URL = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"
MOEX_BONDS_HISTORY_URL = "https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities.json"
CBR_URL = "http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
HTTP_TIMEOUT = 30
MAX_AGE_BONDS_SEC = 86400  # 24 часа для справочника
MAX_AGE_BOND_PRICES = 3600  # 1 час для цен
MAX_AGE_CBR = 86400  # 24 часа для ставки ЦБ

# Матрица субпортфелей
CORE_OFZ_SUBPORTFOLIO_COL = "ОФЗ"
LONG_OFZ_SUBPORTFOLIO_COL = "Длинные ОФЗ"
CORP_SUBPORTFOLIO_COL = "Корпораты"

# ============================================================================
# Утилиты
# ============================================================================
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")


def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{date}.log"
    logger = logging.getLogger("module2")
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
        "bond_sleeve_pct": 35.0,
        "core_ofz_share_pct": 30.0,
        "long_ofz_share_pct": 10.0,
        "corp_share_pct": 60.0,
        "core_ofz_min_years": 2.5,
        "core_ofz_max_years": 3.0,
        "core_ofz_price_min_pct": 85.0,
        "core_ofz_price_max_pct": 105.0,
        "long_ofz_min_years": 8.0,
        "long_ofz_max_years": 12.0,
        "long_ofz_price_min_pct": 50.0,
        "long_ofz_price_max_pct": 105.0,
        "long_ofz_top_n": 2,
        "corp_ytm_min_spread_to_cbr": 5.0,
        "corp_ytm_max_multiplier": 3.0,
        "corp_price_min_pct": 60.0,
        "corp_price_max_pct": 105.0,
        "corp_min_months": 12,
        "corp_max_months": 40,
        "min_credit_rating": "A-",
        "issuer_limit_pct": 3.0,
        "bond_min_volume_rub": 1_000_000.0,
        "bond_avg_volume_days": 20,
        "bond_require_coupon": True,
        "bond_max_facevalue": 10000,
        "risk_free_rate_max_age_sec": 86400,
        "tax_rate_pct": 13.0,
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



def file_age(path: Path) -> float:
    return time.time() - path.stat().st_mtime if path.exists() else float('inf')


def is_fresh(path: Path, max_age: float) -> bool:
    return file_age(path) <= max_age

# ============================================================================
# Работа с матрицей субпортфелей
# ============================================================================
def load_portfolio_units(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Загружает матрицу субпортфелей. Если файла нет — возвращает пустой DataFrame."""
    if not path.exists():
        logger.info(f"Матрица {path.name} не найдена — будет создана")
        return pd.DataFrame(columns=["SECID", "asset_type", "ISIN"])
    
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    else:
        df = pd.read_csv(path, sep=";")
    
    if "SECID" not in df.columns:
        logger.warning(f"В {path.name} нет колонки SECID — создаю новую матрицу")
        return pd.DataFrame(columns=["SECID", "asset_type", "ISIN"])
    
    # Приводим числовые колонки к numeric
    for c in df.columns:
        if c not in ("SECID", "asset_type", "ISIN"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    return df

def update_subportfolio_column(
    matrix: pd.DataFrame,
    col_name: str,
    values: dict,  # {SECID_or_ISIN: units}
    asset_type: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Обновляет (или создаёт) колонку sub-портфеля в матрице.
    - Если SECID/ISIN уже есть — перезаписывает значение.
    - Если новый — добавляет строку с указанным asset_type.
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
                "asset_type": asset_type,
                col_name: units,
            }])
            df = pd.concat([df, new_row], ignore_index=True)
    
    # Заполняем NaN в остальных строках нулями
    for c in df.columns:
        if c not in ("SECID", "asset_type", "ISIN"):
            df[c] = df[c].fillna(0)
    
    # Удаляем строки с total_units = 0
    sub_cols = [c for c in df.columns if c not in ("SECID", "asset_type", "ISIN")]
    if sub_cols:
        df["_total"] = df[sub_cols].sum(axis=1)
        removed = int((df["_total"] == 0).sum())
        df = df[df["_total"] > 0].drop(columns=["_total"]).reset_index(drop=True)
        if removed:
            logger.info(f"Матрица: удалено {removed} строк с нулевыми units")
    
    # Сортировка
    df = df.sort_values(["asset_type", "SECID"], ascending=[False, True]).reset_index(drop=True)
    
    return df

def save_portfolio_units(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет матрицу с разделителем ; и целыми числами для units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in df.columns if c not in ("SECID", "asset_type", "ISIN")]
    out = df.copy()
    for c in num_cols:
        out[c] = out[c].round().astype("Int64")
    out.to_csv(path, index=False, sep=";", encoding="utf-8")

# def update_bond_portfolio_units(
#     core_ofz_ranking: pd.DataFrame,
#     long_ofz_ranking: pd.DataFrame,
#     corp_ranking: pd.DataFrame,
#     data_dir: Path,
#     logger: logging.Logger,
# ) -> None:
#     """
#     Обновляет матрицу субпортфелей целевыми весами облигаций из трёх рейтингов.
#     """
#     units_path = data_dir / "portfolio_units.csv"
    
#     try:
#         matrix = load_portfolio_units(units_path, logger)
        
#         # Собираем словари {ISIN: target_weight_sleeve_pct} для каждого бакета
#         def extract_weights(df: pd.DataFrame) -> dict:
#             if df.empty:
#                 return {}
#             eligible = df[df.get("eligible") == True]
#             if eligible.empty:
#                 return {}
#             weights = {}
#             for _, row in eligible.iterrows():
#                 isin = str(row.get("ISIN", "")).strip().upper()
#                 weight = float(row.get("target_weight_sleeve_pct", 0) or 0)
#                 if isin and weight > 0:
#                     weights[isin] = weight
#             return weights
        
#         core_ofz_weights = extract_weights(core_ofz_ranking)
#         long_ofz_weights = extract_weights(long_ofz_ranking)
#         corp_weights = extract_weights(corp_ranking)
        
#         # Обновляем три колонки
#         matrix = update_subportfolio_column(matrix, CORE_OFZ_SUBPORTFOLIO_COL, core_ofz_weights, "bond", logger)
#         matrix = update_subportfolio_column(matrix, LONG_OFZ_SUBPORTFOLIO_COL, long_ofz_weights, "bond", logger)
#         matrix = update_subportfolio_column(matrix, CORP_SUBPORTFOLIO_COL, corp_weights, "bond", logger)
        
#         save_portfolio_units(matrix, units_path)
        
#         logger.info(
#             f"Матрица обновлена: {len(matrix)} строк, "
#             f"колонок-субпортфелей: {len([c for c in matrix.columns if c not in ('SECID','asset_type')])}, "
#             f"заполнено: '{CORE_OFZ_SUBPORTFOLIO_COL}'={len(core_ofz_weights)}, "
#             f"'{LONG_OFZ_SUBPORTFOLIO_COL}'={len(long_ofz_weights)}, "
#             f"'{CORP_SUBPORTFOLIO_COL}'={len(corp_weights)}"
#         )
        
#     except Exception as e:
#         logger.error(f"Не удалось обновить матрицу субпортфелей: {e}")
#         raise
    
def update_bond_portfolio_units(core_ofz_ranking: pd.DataFrame,
                                long_ofz_ranking: pd.DataFrame,
                                corp_ranking: pd.DataFrame,
                                data_dir: Path,
                                logger: logging.Logger) -> None:
    """Обновляет матрицу субпортфелей целевыми весами облигаций.
    Ключ строки — SECID (SU.../BYM...), отдельная колонка ISIN.
    Поддерживает миграцию старых строк, где в SECID лежал ISIN."""
    units_path = data_dir / "portfolio_units.csv"
    try:
        matrix = load_portfolio_units(units_path, logger)
        if matrix.empty:
            matrix = pd.DataFrame(columns=["SECID", "asset_type"])
        matrix = matrix.copy()
        matrix["SECID"] = matrix["SECID"].astype(str).str.strip().str.upper()
        # if "ISIN" not in matrix.columns:
        #     matrix["ISIN"] = ""
        # matrix["ISIN"] = matrix["ISIN"].fillna("").astype(str).str.strip().str.upper()

        # 1. Маппинги из рейтингов (в них уже есть SECID + ISIN из справочников)
        # isin_to_secid: dict = {}
        # secid_to_isin: dict = {}
        # for ranking in (core_ofz_ranking, long_ofz_ranking, corp_ranking):
        #     if ranking.empty or "SECID" not in ranking.columns or "ISIN" not in ranking.columns:
        #         continue
        #     for _, r in ranking.iterrows():
        #         secid = str(r.get("SECID", "")).strip().upper()
                # isin = str(r.get("ISIN", "")).strip().upper()
                # if secid and isin:
                #     isin_to_secid[isin] = secid
                #     secid_to_isin[secid] = isin
        # logger.info(f"Матрица облигаций: маппинг SECID<->ISIN = {len(secid_to_isin)} бумаг")

        # 2. Миграция legacy-строк: в SECID лежит ISIN
        # migrated = 0
        # for idx in matrix.index:
        #     val = str(matrix.at[idx, "SECID"]).upper()
        #     if len(val) >= 10 and (val.startswith("RU") or val.startswith("XS")):
        #         real_secid = isin_to_secid.get(val, "")
        #         if real_secid:
        #             if not matrix.at[idx, "ISIN"]:
        #                 matrix.at[idx, "ISIN"] = val
        #             matrix.at[idx, "SECID"] = real_secid
        #             migrated += 1
        #         else:
        #             logger.warning(f"Legacy ISIN {val}: SECID не найден в справочниках/рейтингах")
        # if migrated:
        #     logger.info(f"Матрица облигаций: мигрировано {migrated} строк")

        # 3. Заполняем ISIN для всех известных облигаций
        # known = matrix["SECID"].isin(secid_to_isin)
        # matrix.loc[known, "ISIN"] = matrix.loc[known, "SECID"].map(secid_to_isin)

        # 4. Веса по SECID (не по ISIN!)
        def extract_weights(df: pd.DataFrame) -> dict:
            if df.empty or "eligible" not in df.columns or "SECID" not in df.columns:
                return {}
            eligible = df[df["eligible"] == True]  # noqa: E712
            weights = {}
            for _, r in eligible.iterrows():
                secid = str(r.get("SECID", "")).strip().upper()
                try:
                    weight = float(r.get("target_weight_sleeve_pct", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if secid and weight > 0:
                    weights[secid] = weight
            return weights

        def update_col(col_name: str, weights: dict) -> None:
            """Обнуляем колонку, затем проставляем веса по SECID; новые строки — с ISIN."""
            nonlocal matrix
            if col_name not in matrix.columns:
                matrix[col_name] = 0.0
            matrix[col_name] = pd.to_numeric(matrix[col_name], errors="coerce").fillna(0.0)
            matrix[col_name] = 0.0
            for secid, weight in weights.items():
                mask = matrix["SECID"] == secid
                if mask.any():
                    matrix.loc[mask, col_name] = weight
                    # if secid in secid_to_isin:
                    #     matrix.loc[mask, "ISIN"] = secid_to_isin[secid]
                else:
                    new_row = {c: 0 for c in matrix.columns}
                    new_row["SECID"] = secid
                    new_row["asset_type"] = "bond"
                    # new_row["ISIN"] = secid_to_isin.get(secid, "")
                    new_row[col_name] = weight
                    matrix = pd.concat([matrix, pd.DataFrame([new_row])], ignore_index=True)
            logger.info(f"Матрица: '{col_name}' заполнено {len(weights)} бумаг (ключ SECID)")

        update_col(CORE_OFZ_SUBPORTFOLIO_COL, extract_weights(core_ofz_ranking))
        update_col(LONG_OFZ_SUBPORTFOLIO_COL, extract_weights(long_ofz_ranking))
        update_col(CORP_SUBPORTFOLIO_COL, extract_weights(corp_ranking))

        # 5. Числовые колонки (ISIN — НЕ числовая!) и удаление пустых строк
        meta_cols = {"SECID", "asset_type", "ISIN"}
        sub_cols = [c for c in matrix.columns if c not in meta_cols]
        for c in sub_cols:
            matrix[c] = pd.to_numeric(matrix[c], errors="coerce").fillna(0.0)
        if sub_cols:
            keep = matrix[sub_cols].sum(axis=1) > 1e-9
            dropped = int((~keep).sum())
            if dropped:
                logger.info(f"Матрица: удалено {dropped} строк с нулевыми весами")
            matrix = matrix[keep].copy()
        matrix = matrix.sort_values("SECID").reset_index(drop=True)

        save_portfolio_units(matrix, units_path)
        logger.info(f"Матрица сохранена: {len(matrix)} строк, колонка ISIN заполнена")
    except Exception as e:
        logger.error(f"Ошибка обновления матрицы облигаций: {e}")    
    
    
# ============================================================================
# Загрузка ставки ЦБ
# ============================================================================
def fetch_cbr_rate() -> float:
    """Загружает текущую ключевую ставку ЦБ через SOAP."""
    today = datetime.now()
    d_from = today - timedelta(days=7)
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
        '<KeyRate xmlns="http://web.cbr.ru/">'
        f'<fromDate>{d_from:%Y-%m-%d}T00:00:00</fromDate>'
        f'<ToDate>{today:%Y-%m-%d}T00:00:00</ToDate>'
        '</KeyRate></soap:Body></soap:Envelope>'
    )
    headers = {
        'Content-Type': 'text/xml; charset=utf-8',
        'SOAPAction': '"http://web.cbr.ru/KeyRate"'
    }
    r = requests.post(CBR_URL, data=body.encode('utf-8'), headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.text)
    rates = []
    for el in root.iter():
        leaves = {c.tag.split("}")[-1]: (c.text or "").strip() for c in el if len(c) == 0}
        if len(leaves) >= 2:
            date = next((leaves[k] for k in ("DT", "Date", "date") if k in leaves), None)
            rate = next((leaves[k] for k in ("Rate", "BaseRate", "Value") if k in leaves), None)
            if date and rate:
                try:
                    rates.append((date[:10], float(rate.replace(",", "."))))
                except ValueError:
                    continue
    
    if not rates:
        raise ValueError("Не удалось получить ставку ЦБ")
    
    # Берём последнюю (самую свежую) ставку
    rates.sort(key=lambda x: x[0], reverse=True)
    return rates[0][1]


def load_risk_free_rate(data_dir: Path, config: dict, logger: logging.Logger) -> float:
    """Загружает ставку ЦБ из файла или загружает с ЦБ."""
    rate_path = data_dir / "risk_free_rate.csv"
    max_age = config.get("risk_free_rate_max_age_sec", MAX_AGE_CBR)
    
    if is_fresh(rate_path, max_age):
        df = pd.read_csv(rate_path)
        if not df.empty:
            rate = float(df.iloc[-1]["rate_pct"])
            logger.info(f"Ставка ЦБ (из кэша): {rate:.2f}%")
            return rate
    
    logger.info("Загрузка ставки ЦБ с cbr.ru...")
    try:
        rate = fetch_cbr_rate()
        # Сохраняем одну строку
        df = pd.DataFrame({"date": [datetime.now().strftime("%Y-%m-%d")], "rate_pct": [rate]})
        save_csv(df, rate_path)
        logger.info(f"Ставка ЦБ (загружена): {rate:.2f}%")
        return rate
    except Exception as e:
        logger.error(f"Ошибка загрузки ставки ЦБ: {e}")
        if rate_path.exists():
            df = pd.read_csv(rate_path)
            if not df.empty:
                rate = float(df.iloc[-1]["rate_pct"])
                logger.warning(f"Используется устаревшая ставка: {rate:.2f}%")
                return rate
        raise RuntimeError("Не удалось загрузить ставку ЦБ, и кэш отсутствует")

def _issuer_key(name: str) -> str:
    """
    Извлекает ключ эмитента из названия выпуска.
    SAMOLET P11 -> SAMOLET; ГАЗПРОМ 001Р-05 -> ГАЗПРОМ; ЛУКОЙЛ 10 -> ЛУКОЙЛ
    """
    import re
    s = str(name or "").strip().upper()
    # Отрезаем последний токен, содержащий номер выпуска (P11, 001P-05, 32, 1P1)
    s = re.sub(r"\s+[A-ZА-Я]*\d+[A-ZА-Я]*(?:-\d+)*$", "", s)
    return re.sub(r"[\s\-–]+$", "", s)

# ============================================================================
# Загрузка справочника облигаций
# ============================================================================
def fetch_and_save_bonds_universe(data_dir: Path, logger: logging.Logger) -> None:
    ofz_path = data_dir / "bonds_OFZ.csv"
    corps_path = data_dir / "bonds_CORPS.csv"
    
    if is_fresh(ofz_path, MAX_AGE_BONDS_SEC) and is_fresh(corps_path, MAX_AGE_BONDS_SEC):
        logger.info("Справочники облигаций актуальны.")
        return
    
    logger.info("Загрузка справочника облигаций с MOEX ISS...")
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "securities.columns": "SECID,ISIN,SHORTNAME,SECNAME,LATNAME,LOTSIZE,BOARDID,MATDATE,COUPONPERCENT,COUPONPERIOD,FACEVALUE"
    }
    try:
        r = requests.get(MOEX_BONDS_URL, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error(f"Ошибка загрузки справочника: {e}")
        if not ofz_path.exists() or not corps_path.exists():
            raise RuntimeError("Не удалось загрузить справочник, и локальные файлы отсутствуют")
        logger.warning("Используются устаревшие локальные файлы")
        return
    
    if "securities" not in payload or "data" not in payload["securities"]:
        raise RuntimeError("Некорректный ответ MOEX ISS")
    
    cols = payload["securities"]["columns"]
    df = pd.DataFrame(payload["securities"]["data"], columns=cols)
    
    df["LOTSIZE"] = pd.to_numeric(df["LOTSIZE"], errors="coerce").fillna(1).astype(int)
    df["FACEVALUE"] = pd.to_numeric(df["FACEVALUE"], errors="coerce").fillna(1000.0)
    df["COUPONPERCENT"] = pd.to_numeric(df["COUPONPERCENT"], errors="coerce").fillna(0.0)
    df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    # AMORTIZATION не отдаётся MOEX ISS в массовом эндпоинте — проставляем False по умолчанию
    df["AMORTIZATION"] = False
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Исключаем внебиржевые сделки (SPOB)
    df = df[df["BOARDID"] != "SPOB"].copy()
    logger.info(f"После исключения SPOB: {len(df)} бумаг")
    
    # Разделение на ОФЗ и корпораты
    # ОФЗ определяем по торговой доске (TQOB/TQOF), без обращения к NAME
    _name_col = "SHORTNAME" if "SHORTNAME" in df.columns else "LATNAME"
    is_ofz = (
        df["BOARDID"].isin(["TQOB", "TQOF"])
        | df[_name_col].fillna("").str.contains("ОФЗ|ОФЗ|OFZ", case=False, na=False)
    )
    
    df_ofz = df[is_ofz].copy()
    df_corps = df[~is_ofz].copy()
    
    # ОФЗ
    ofz_res = pd.DataFrame({
        "SECID": df_ofz["SECID"],
        "ISIN": df_ofz["ISIN"],
        "name": df_ofz["SHORTNAME"],
        "issuer": "RF",
        "maturity_date": df_ofz["MATDATE"],
        "coupon_type": "fixed",
        "coupon_rate_pct": df_ofz["COUPONPERCENT"],
        "coupon_period_days": df_ofz["COUPONPERIOD"],
        "nominal": df_ofz["FACEVALUE"],
        "amortization": df_ofz["AMORTIZATION"],
        "LOTSIZE": df_ofz["LOTSIZE"],
        "board": df_ofz["BOARDID"],
        "lot_as_of": today_str
    })
    
    # Корпораты
    corps_res = pd.DataFrame({
        "SECID": df_corps["SECID"],
        "ISIN": df_corps["ISIN"],
        "name": df_corps["SHORTNAME"],
        # Временный ключ эмитента: LATNAME (или SHORTNAME) без номера выпуска
        "issuer": df_corps["LATNAME"].fillna(df_corps["SHORTNAME"]).map(_issuer_key),
        "maturity_date": df_corps["MATDATE"],
        "coupon_type": "fixed",
        "coupon_rate_pct": df_corps["COUPONPERCENT"],
        "coupon_period_days": df_corps["COUPONPERIOD"],
        "nominal": df_corps["FACEVALUE"],
        "amortization": df_corps["AMORTIZATION"],
        "LOTSIZE": df_corps["LOTSIZE"],
        "board": df_corps["BOARDID"],
        "lot_as_of": today_str
    })
    
    ofz_res = ofz_res.drop_duplicates(subset=["ISIN"], keep="first").sort_values("ISIN").reset_index(drop=True)
    corps_res = corps_res.drop_duplicates(subset=["ISIN"], keep="first").sort_values("ISIN").reset_index(drop=True)
    
    save_csv(ofz_res, ofz_path)
    save_csv(corps_res, corps_path)
    logger.info(f"Справочники обновлены: ОФЗ={len(ofz_res)}, Корпораты={len(corps_res)}")


# ============================================================================
# Загрузка цен облигаций
# ============================================================================
def fetch_and_save_bond_prices(data_dir: Path, logger: logging.Logger) -> None:
    prices_path = data_dir / "bond_prices.csv"
    logger.info("Загрузка цен облигаций с MOEX ISS...")
    params = {
        "iss.meta": "off",
        "iss.only": "marketdata,securities",
        "marketdata.columns": "SECID,BOARDID,LCURRENTPRICE,LAST,LCLOSEPRICE,YIELD,DURATION,VOLTODAY",
        "securities.columns": "SECID,BOARDID,ISIN,ACCRUEDINT,FACEVALUE",
    }
    try:
        r = requests.get(MOEX_BONDS_URL, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error(f"Ошибка загрузки цен: {e}")
        if not prices_path.exists():
            raise RuntimeError("Не удалось загрузить цены, и локальный файл отсутствует")
        logger.warning("Используются устаревшие локальные цены")
        return

    if "marketdata" not in payload or "data" not in payload["marketdata"]:
        raise RuntimeError("Некорректный ответ MOEX ISS: нет блока marketdata.data")
    md_cols = payload["marketdata"]["columns"]
    md_df = pd.DataFrame(payload["marketdata"]["data"], columns=md_cols)

    sec_block = payload.get("securities")
    if sec_block and "data" in sec_block and "columns" in sec_block:
        sec_df = pd.DataFrame(sec_block["data"], columns=sec_block["columns"])
    else:
        logger.warning("В ответе ISS нет блока securities — НКД и ISIN будут пустыми")
        sec_df = pd.DataFrame(columns=["SECID", "BOARDID", "ISIN", "ACCRUEDINT", "FACEVALUE"])

    for col in ["LCURRENTPRICE", "LAST", "LCLOSEPRICE", "YIELD", "DURATION", "VOLTODAY"]:
        if col in md_df.columns:
            md_df[col] = pd.to_numeric(md_df[col], errors="coerce")

    if "ACCRUEDINT" in sec_df.columns:
        sec_df["ACCRUEDINT"] = pd.to_numeric(sec_df["ACCRUEDINT"], errors="coerce").fillna(0.0)
    if "FACEVALUE" in sec_df.columns:
        sec_df["FACEVALUE"] = pd.to_numeric(sec_df["FACEVALUE"], errors="coerce")

    md_df["price_pct"] = np.nan
    for col in ["LCURRENTPRICE", "LAST", "LCLOSEPRICE"]:
        if col in md_df.columns:
            mask = md_df["price_pct"].isna() & md_df[col].notna() & (md_df[col] > 0)
            md_df.loc[mask, "price_pct"] = md_df.loc[mask, col]

    md_df = md_df.dropna(subset=["SECID", "price_pct"])
    md_df = md_df.drop_duplicates(subset=["SECID"], keep="first")
    sec_df = sec_df.dropna(subset=["SECID"])
    sec_df = sec_df.drop_duplicates(subset=["SECID"], keep="first")

    merged = md_df.merge(sec_df[["SECID", "ISIN", "ACCRUEDINT"]], on="SECID", how="left")
    today_str = datetime.now().strftime("%Y-%m-%d")

    res_df = pd.DataFrame({
        "date": today_str,
        "SECID": merged["SECID"],
        "ISIN": merged["ISIN"],
        "price_pct": merged["price_pct"],
        "nkd_rub": merged["ACCRUEDINT"].fillna(0.0),
        "ytm_pct": merged["YIELD"],
        "duration_years": merged["DURATION"],
        "volume_rub": merged["VOLTODAY"].fillna(0.0)
    })

    if "BOARDID" in merged.columns:
        res_df = res_df[merged["BOARDID"] != "SPOB"].copy()
    res_df = res_df.dropna(subset=["ISIN"])
    res_df = res_df.drop_duplicates(subset=["ISIN"], keep="first").sort_values("ISIN").reset_index(drop=True)

    save_csv(res_df, prices_path)
    logger.info(f"Цены облигаций обновлены: {len(res_df)} записей")
    
    
def build_isin_to_secid_map(data_dir: Path, logger: logging.Logger) -> dict:
    """
    Строит маппинг ISIN -> SECID из справочников bonds_OFZ.csv и bonds_CORPS.csv.
    Используется для обогащения цен и рейтингов облигаций колонкой SECID.
    """
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
# Вспомогательные функции для Module 2
# ============================================================================

def fetch_bond_history_bulk(date_dt: datetime, days: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Загружает историю торгов за N дней для ВСЕХ облигаций одним запросом по основным площадкам.
    Возвращает DataFrame с колонками: SECID, avg_volume_rub, avg_daily_trades
    """
    date_from = (date_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    date_till = date_dt.strftime("%Y-%m-%d")
    
    boards = ["TQOB", "TQCB"]
    all_history = []
    
    for board in boards:
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/bonds/boards/{board}/securities.json"
        params = {
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "SECID,TRADEDATE,VALUE,NUMTRADES",
            "from": date_from,
            "till": date_till
        }
        
        logger.info(f"Загрузка истории торгов для площадки {board} за {days} дней...")
        start = 0
        while True:
            params["start"] = start
            try:
                r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                logger.warning(f"Ошибка загрузки истории {board} (start={start}): {e}")
                break
                
            if "history" not in payload or "data" not in payload["history"]:
                break
                
            cols = payload["history"]["columns"]
            data = payload["history"]["data"]
            if not data:
                break
                
            df_chunk = pd.DataFrame(data, columns=cols)
            all_history.append(df_chunk)
            
            # Проверка на пагинацию
            cursor = payload.get("history", {}).get("cursor", {})
            if cursor.get("INDEX", 0) + cursor.get("TOTAL", 0) <= cursor.get("START", 0) + len(data):
                break
            start = cursor.get("START", 0) + len(data)

    if not all_history:
        logger.warning("Не удалось загрузить историю торгов, возвращаем пустой DataFrame")
        return pd.DataFrame(columns=["SECID", "avg_volume_rub", "avg_daily_trades"])
        
    df_hist = pd.concat(all_history, ignore_index=True)
    
    df_hist["VALUE"] = pd.to_numeric(df_hist["VALUE"], errors="coerce").fillna(0.0)
    df_hist["NUMTRADES"] = pd.to_numeric(df_hist["NUMTRADES"], errors="coerce").fillna(0.0)
    
    # Агрегация: среднее значение за дни, когда были торги
    df_agg = df_hist.groupby("SECID").agg(
        avg_volume_rub=("VALUE", "mean"),
        avg_daily_trades=("NUMTRADES", "mean")
    ).reset_index()
    
    logger.info(f"История торгов загружена и агрегирована для {len(df_agg)} уникальных SECID")
    return df_agg


def apply_water_filling(df_b: pd.DataFrame, share: float, cap: float, bond_sleeve_pct: float) -> pd.DataFrame:
    """
    Корректное распределение весов (water-filling) с учётом лимита на позицию/эмитента (cap).
    Гарантирует, что сумма весов будет равна share, если cap * N >= share.
    """
    mask = df_b["eligible"] == True
    if not mask.any():
        return df_b
    
    eligible_df = df_b[mask].copy()
    eligible_df = eligible_df.sort_values("current_yield_pct", ascending=False).reset_index(drop=True)
    
    df_b.loc[mask, "target_weight_sleeve_pct"] = 0.0
    remaining_share = float(share)
    n_eligible = len(eligible_df)
    
    for idx, row in eligible_df.iterrows():
        if remaining_share <= 1e-6 or n_eligible <= 0:
            break
        
        # Целевой вес, если делить остаток поровну на все оставшиеся позиции
        target_weight = remaining_share / n_eligible
        
        # Ограничиваем cap-ом
        weight_to_assign = min(target_weight, cap)
        
        df_b.loc[row.name, "target_weight_sleeve_pct"] = weight_to_assign
        remaining_share -= weight_to_assign
        n_eligible -= 1
        
    # Расчёт портфельного веса
    df_b.loc[mask, "target_weight_portfolio_pct"] = df_b.loc[mask, "target_weight_sleeve_pct"] * (bond_sleeve_pct / 100.0)
    return df_b


# ============================================================================
# Основная функция рейтинга
# ============================================================================
def rank_bonds(date: str, config: dict, data_dir: Path, logger: logging.Logger, diagnostics: list):
    date_dt = pd.to_datetime(date)
    
    # 1. Загрузка ставки ЦБ
    risk_free_rate = load_risk_free_rate(data_dir, config, logger)
    
    # 2. Загрузка справочников и цен
    ofz = pd.read_csv(data_dir / "bonds_OFZ.csv", sep=";")
    corps = pd.read_csv(data_dir / "bonds_CORPS.csv", sep=";")
    prices = pd.read_csv(data_dir / "bond_prices.csv", sep=";")
    prices["date"] = pd.to_datetime(prices["date"])
    latest_prices = prices[prices["date"] <= date_dt].sort_values("date").groupby("SECID").tail(1).set_index("SECID")
    
    # Рейтинг-порядок
    rating_order = {"AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7, "BBB+": 8, "BBB": 9, "BBB-": 10}
    min_rating_val = rating_order.get(config["min_credit_rating"], 99)
    
    # ========================================================================
    # Обработка ОФЗ
    # ========================================================================
    ofz_results = []
    for _, row in ofz.iterrows():
        isin = row["ISIN"]
        mat_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        if pd.isna(mat_date):
            ofz_results.append({"date": date,"SECID": row.get("SECID", ""), "ISIN": isin, "name": row.get("name", ""), "issuer": "RF",
                "bond_type": "ofz", "maturity_date": "", "days_to_maturity": 0, "coupon_rate_pct": 0.0,
                "price_pct": np.nan, "nkd_rub": np.nan, "current_yield_pct": 0.0, "ytm_gross_pct": np.nan,
                "spread_to_cbr_pct": np.nan, "duration_years": np.nan, "bucket": "excluded", "eligible": False,
                "exclusion_reason": "no_maturity_date", "avg_daily_trades": 0.0, "LOTSIZE": int(row.get("LOTSIZE", 1)),
                "target_weight_sleeve_pct": 0.0, "target_weight_portfolio_pct": 0.0})
            continue
            
        years_to_mat = (mat_date - date_dt).days / 365.0
        
        if isin not in latest_prices.index:
            ofz_results.append({"date": date,"SECID": row.get("SECID", ""), "ISIN": isin, "name": row.get("name", ""), "issuer": "RF",
                "bond_type": "ofz", "maturity_date": mat_date.strftime("%Y-%m-%d"), "days_to_maturity": int(years_to_mat * 365),
                "coupon_rate_pct": row.get("coupon_rate_pct", 0.0), "price_pct": np.nan, "nkd_rub": np.nan,
                "current_yield_pct": 0.0, "ytm_gross_pct": np.nan, "spread_to_cbr_pct": np.nan, "duration_years": np.nan,
                "bucket": "excluded", "eligible": False, "exclusion_reason": "no_price", "avg_daily_trades": 0.0,
                "LOTSIZE": int(row.get("LOTSIZE", 1)), "target_weight_sleeve_pct": 0.0, "target_weight_portfolio_pct": 0.0})
            continue
            
        p_row = latest_prices.loc[isin]
        price = p_row["price_pct"]
        ytm = p_row["ytm_pct"]
        
        exclusion_reason = ""
        eligible = True
        
        if row.get("nominal", 1000) >= config["bond_max_facevalue"]:
            eligible, exclusion_reason = False, "facevalue_too_high"
            
        if eligible and config["bond_require_coupon"]:
            coupon_period = row.get("coupon_period_days", 0)
            coupon_rate = row.get("coupon_rate_pct", 0)
            if pd.isna(coupon_period) or coupon_period <= 0 or coupon_rate == 0:
                eligible, exclusion_reason = False, "no_coupon"
                
        if eligible and (pd.isna(ytm) or ytm < risk_free_rate):
            eligible, exclusion_reason = False, "ytm_below_risk_free"
            
        bucket = "excluded"
        if eligible:
            if config["core_ofz_min_years"] <= years_to_mat <= config["core_ofz_max_years"]:
                if not (config["core_ofz_price_min_pct"] <= price <= config["core_ofz_price_max_pct"]):
                    eligible, exclusion_reason = False, "price_out_of_range"
                else:
                    bucket = "core_ofz"
            elif config["long_ofz_min_years"] <= years_to_mat <= config["long_ofz_max_years"]:
                if not (config["long_ofz_price_min_pct"] <= price <= config["long_ofz_price_max_pct"]):
                    eligible, exclusion_reason = False, "price_out_of_range"
                else:
                    bucket = "long_ofz"
            else:
                eligible, exclusion_reason = False, "maturity_out_of_range"
                
        current_yield = (row.get("coupon_rate_pct", 0.0) / (price / 100)) if price > 0 else 0.0
        
        ofz_results.append({"date": date,"SECID": row.get("SECID", ""), "ISIN": isin, "name": row.get("name", ""), "issuer": "RF",
            "bond_type": "ofz", "maturity_date": mat_date.strftime("%Y-%m-%d"), "days_to_maturity": int(years_to_mat * 365),
            "coupon_rate_pct": row.get("coupon_rate_pct", 0.0), "price_pct": price, "nkd_rub": p_row.get("nkd_rub", 0.0),
            "current_yield_pct": current_yield, "ytm_gross_pct": ytm, "spread_to_cbr_pct": ytm - risk_free_rate if pd.notna(ytm) else np.nan,
            "duration_years": p_row.get("duration_years", np.nan), "bucket": bucket, "eligible": eligible,
            "exclusion_reason": exclusion_reason, "avg_daily_trades": 0.0, "LOTSIZE": int(row.get("LOTSIZE", 1)),
            "target_weight_sleeve_pct": 0.0, "target_weight_portfolio_pct": 0.0})
            
    df_ofz_all = pd.DataFrame(ofz_results)
    logger.info("ОФЗ: фильтр по обороту не применяется (безрисковый инструмент)")
    
    df_core_ofz = df_ofz_all[df_ofz_all["bucket"] == "core_ofz"].copy()
    df_long_ofz = df_ofz_all[df_ofz_all["bucket"] == "long_ofz"].copy()
    
    df_core_ofz = df_core_ofz.sort_values("current_yield_pct", ascending=False).reset_index(drop=True)
    df_core_ofz["rank"] = df_core_ofz.index + 1
    
    df_long_ofz = df_long_ofz.sort_values("current_yield_pct", ascending=False).reset_index(drop=True)
    df_long_ofz["rank"] = df_long_ofz.index + 1
    long_ofz_top_n = config.get("long_ofz_top_n", 2)
    df_long_ofz = df_long_ofz.head(long_ofz_top_n)
    
    # Применяем water-filling для ОФЗ
    df_core_ofz = apply_water_filling(df_core_ofz, config["core_ofz_share_pct"], config["issuer_limit_pct"], config["bond_sleeve_pct"])
    df_long_ofz = apply_water_filling(df_long_ofz, config["long_ofz_share_pct"], config["issuer_limit_pct"], config["bond_sleeve_pct"])

    # ========================================================================
    # Обработка корпоратов
    # ========================================================================
    corp_results = []
    for _, row in corps.iterrows():
        isin = row["ISIN"]
        mat_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        if pd.isna(mat_date):
            corp_results.append({"date": date,"SECID": row.get("SECID", ""), "ISIN": isin, "name": row.get("name", ""), "issuer": row.get("issuer", ""),
                "bond_type": "corp", "maturity_date": "", "days_to_maturity": 0, "coupon_rate_pct": 0.0,
                "price_pct": np.nan, "nkd_rub": np.nan, "current_yield_pct": 0.0, "ytm_gross_pct": np.nan,
                "spread_to_cbr_pct": np.nan, "duration_years": np.nan, "bucket": "excluded", "eligible": False,
                "exclusion_reason": "no_maturity_date", "avg_daily_trades": 0.0, "LOTSIZE": int(row.get("LOTSIZE", 1)),
                "target_weight_sleeve_pct": 0.0, "target_weight_portfolio_pct": 0.0})
            continue
            
        months_to_mat = (mat_date - date_dt).days / 30.44
        
        if isin not in latest_prices.index:
            corp_results.append({"date": date,"SECID": row.get("SECID", ""), "ISIN": isin, "name": row.get("name", ""), "issuer": row.get("issuer", ""),
                "bond_type": "corp", "maturity_date": mat_date.strftime("%Y-%m-%d"), "days_to_maturity": int(months_to_mat * 30.44),
                "coupon_rate_pct": row.get("coupon_rate_pct", 0.0), "price_pct": np.nan, "nkd_rub": np.nan,
                "current_yield_pct": 0.0, "ytm_gross_pct": np.nan, "spread_to_cbr_pct": np.nan, "duration_years": np.nan,
                "bucket": "excluded", "eligible": False, "exclusion_reason": "no_price", "avg_daily_trades": 0.0,
                "LOTSIZE": int(row.get("LOTSIZE", 1)), "target_weight_sleeve_pct": 0.0, "target_weight_portfolio_pct": 0.0})
            continue
            
        p_row = latest_prices.loc[isin]
        price = p_row["price_pct"]
        ytm = p_row["ytm_pct"]
        
        exclusion_reason = ""
        eligible = True
        
        if row.get("nominal", 1000) >= config["bond_max_facevalue"]:
            eligible, exclusion_reason = False, "facevalue_too_high"
            
        if eligible and config["bond_require_coupon"]:
            coupon_period = row.get("coupon_period_days", 0)
            coupon_rate = row.get("coupon_rate_pct", 0)
            if pd.isna(coupon_period) or coupon_period <= 0 or coupon_rate == 0:
                eligible, exclusion_reason = False, "no_coupon"
                
        # if eligible and not (config["corp_min_months"] <= months_to_mat <= config["corp_max_months"]):
        #     eligible, exclusion_reason = False, "maturity_out_of_range"
            
        # Фильтр по цене отключён по вашему запросу (или можно раскомментировать при необходимости)
        # if eligible and not (config["corp_price_min_pct"] <= price <= config["corp_price_max_pct"]):
        #     eligible, exclusion_reason = False, "price_out_of_range"
            
        ytm_min = risk_free_rate + config["corp_ytm_min_spread_to_cbr"]
        ytm_max = risk_free_rate * config["corp_ytm_max_multiplier"]
        if eligible and (pd.isna(ytm) or not (ytm_min <= ytm <= ytm_max)):
            eligible, exclusion_reason = False, "ytm_out_of_range"
            
        # if eligible:
        #     rating_val = rating_order.get(str(row.get("credit_rating", "Unknown")), 99)
        #     if rating_val > min_rating_val:
        #         eligible, exclusion_reason = False, "low_rating"
                
        current_yield = (row.get("coupon_rate_pct", 0.0) / (price / 100)) if price > 0 else 0.0
        
        corp_results.append({"date": date,"SECID": row.get("SECID", ""), "ISIN": isin, "name": row.get("name", ""), "issuer": row.get("issuer", ""),
            "bond_type": "corp", "maturity_date": mat_date.strftime("%Y-%m-%d"), "days_to_maturity": int(months_to_mat * 30.44),
            "coupon_rate_pct": row.get("coupon_rate_pct", 0.0), "price_pct": price, "nkd_rub": p_row.get("nkd_rub", 0.0),
            "current_yield_pct": current_yield, "ytm_gross_pct": ytm, "spread_to_cbr_pct": ytm - risk_free_rate if pd.notna(ytm) else np.nan,
            "duration_years": p_row.get("duration_years", np.nan), "bucket": "corp" if eligible else "excluded", 
            "eligible": eligible, "exclusion_reason": exclusion_reason, "avg_daily_trades": 0.0, 
            "LOTSIZE": int(row.get("LOTSIZE", 1)), "target_weight_sleeve_pct": 0.0, "target_weight_portfolio_pct": 0.0})
            
    df_corp_all = pd.DataFrame(corp_results)
    
    # 3. Загрузка истории оборотов ОДНИМ запросом (bulk)
    df_hist_agg = fetch_bond_history_bulk(date_dt, config["bond_avg_volume_days"], logger)
    
    # 4. Merge истории с корпоратами по SECID (нужно добавить SECID в df_corp_all для merge)
    # Для этого создадим маппинг ISIN -> SECID из справочника corps
    isin_to_secid = dict(zip(corps["ISIN"], corps["SECID"]))
    df_corp_all["SECID"] = df_corp_all["ISIN"].map(isin_to_secid)
    
    df_corp_all = df_corp_all.merge(df_hist_agg, on="SECID", how="left")
    # Гарантируем наличие колонок после merge
    if "avg_volume_rub" not in df_corp_all.columns:
        df_corp_all["avg_volume_rub"] = 0.0
    if "avg_daily_trades" not in df_corp_all.columns:
        df_corp_all["avg_daily_trades"] = 0.0
    df_corp_all["avg_volume_rub"] = df_corp_all["avg_volume_rub"].fillna(0.0)
    df_corp_all["avg_daily_trades"] = df_corp_all["avg_daily_trades"].fillna(0.0)
    
    # 5. Фильтрация по объёму
    min_vol = config["bond_min_volume_rub"]
    mask_low_vol = (df_corp_all["eligible"] == True) & (df_corp_all["avg_volume_rub"] < min_vol)
    if mask_low_vol.any():
        df_corp_all.loc[mask_low_vol, "eligible"] = False
        df_corp_all.loc[mask_low_vol, "exclusion_reason"] = "low_volume"
        df_corp_all.loc[mask_low_vol, "bucket"] = "excluded"
        logger.info(f"Корпораты: отсеяно по low_volume: {mask_low_vol.sum()} бумаг")
        
    # 6. Удаление дублей эмитентов
    df_corp_eligible = df_corp_all[(df_corp_all["bucket"] == "corp") & (df_corp_all["eligible"])].copy()
    df_corp_eligible["issuer_norm"] = df_corp_eligible["issuer"].str.lower().str.strip()
    
    # Оставляем лучший по YTM
    df_corp_unique = df_corp_eligible.sort_values("ytm_gross_pct", ascending=False).drop_duplicates(subset=["issuer_norm"], keep="first")
    
    # Помечаем дубли как excluded
    duplicate_isins = set(df_corp_eligible["ISIN"]) - set(df_corp_unique["ISIN"])
    for idx, row in df_corp_all.iterrows():
        if row["ISIN"] in duplicate_isins:
            df_corp_all.loc[idx, "eligible"] = False
            df_corp_all.loc[idx, "exclusion_reason"] = "issuer_duplicate"
            df_corp_all.loc[idx, "bucket"] = "excluded"
            
    df_corp = df_corp_all[(df_corp_all["bucket"] == "corp") & (df_corp_all["eligible"])].copy()
    df_corp = df_corp.sort_values("current_yield_pct", ascending=False).reset_index(drop=True)
    df_corp["rank"] = df_corp.index + 1
    
    # 7. Применяем water-filling для корпоратов
    df_corp = apply_water_filling(df_corp, config["corp_share_pct"], config["issuer_limit_pct"], config["bond_sleeve_pct"])
    
    # ========================================================================
    # Диагностика и сводка
    # ========================================================================
    alloc = []
    for b, df_b in [("core_ofz", df_core_ofz), ("long_ofz", df_long_ofz), ("corp", df_corp)]:
        mask = df_b["eligible"] if not df_b.empty else pd.Series([False])
        if mask.sum() > 0:
            alloc.append({
                "date": date, "bucket": b,
                "target_share_pct": config[f"{b}_share_pct"],
                "n_positions": int(mask.sum()),
                "avg_ytm_gross_pct": df_b.loc[mask, "ytm_gross_pct"].mean(),
                "avg_ytm_net_pct": df_b.loc[mask, "ytm_gross_pct"].mean() * (1 - config["tax_rate_pct"] / 100),
                "total_weight_portfolio_pct": df_b.loc[mask, "target_weight_portfolio_pct"].sum()
            })
            
    filter_stats = []
    ofz_total = len(df_ofz_all)
    ofz_reasons = df_ofz_all[df_ofz_all["eligible"] == False]["exclusion_reason"].value_counts().to_dict()
    filter_stats.append({"bucket": "ofz_total", "count": ofz_total, "reason": "all"})
    for reason, count in ofz_reasons.items():
        filter_stats.append({"bucket": "ofz_excluded", "count": count, "reason": reason})
    filter_stats.append({"bucket": "ofz_volume_check", "count": 0, "reason": "skipped_for_ofz"})
    filter_stats.append({"bucket": "core_ofz_final", "count": int((df_core_ofz["eligible"]).sum()), "reason": "passed"})
    filter_stats.append({"bucket": "long_ofz_final", "count": int((df_long_ofz["eligible"]).sum()), "reason": "passed"})
    
    corp_total = len(df_corp_all)
    corp_reasons = df_corp_all[df_corp_all["eligible"] == False]["exclusion_reason"].value_counts().to_dict()
    filter_stats.append({"bucket": "corp_total", "count": corp_total, "reason": "all"})
    for reason, count in corp_reasons.items():
        filter_stats.append({"bucket": "corp_excluded", "count": count, "reason": reason})
    filter_stats.append({"bucket": "corp_final", "count": int((df_corp["eligible"]).sum()), "reason": "passed"})
    
    df_filter_stats = pd.DataFrame(filter_stats)
    
    # Логирование сводки
    logger.info("=" * 60)
    logger.info("СТАТИСТИКА ФИЛЬТРОВ:")
    logger.info(f"  ОФЗ всего: {ofz_total}")
    for reason, count in ofz_reasons.items():
        logger.info(f"    - {reason}: {count}")
    logger.info(f"  core_ofz прошло: {int((df_core_ofz['eligible']).sum())}")
    logger.info(f"  long_ofz прошло: {int((df_long_ofz['eligible']).sum())}")
    logger.info("  ОФЗ: фильтр по обороту пропущен")
    logger.info(f"  Корпораты всего: {corp_total}")
    for reason, count in corp_reasons.items():
        logger.info(f"    - {reason}: {count}")
    logger.info(f"  corp прошло: {int((df_corp['eligible']).sum())}")
    logger.info("=" * 60)
    
    # 8. Обновление матрицы субпортфелей
    try:
        update_bond_portfolio_units(df_core_ofz, df_long_ofz, df_corp, data_dir, logger)
    except Exception as e:
        logger.error(f"Ошибка обновления матрицы: {e}")
        diagnostics.append({"code": "portfolio_units_update_failed", "details": str(e)})
        
    return df_core_ofz, df_long_ofz, df_corp, pd.DataFrame(alloc), df_filter_stats, df_ofz_all, df_corp_all



def _count_reasons(df: pd.DataFrame, reason_col: str = "exclusion_reason") -> dict:
    """Считает количество бумаг по каждой причине исключения."""
    if df.empty:
        return {}
    excluded = df[df["eligible"] == False]
    if excluded.empty:
        return {}
    return excluded[reason_col].value_counts().to_dict()

# def enrich_with_secid(df: pd.DataFrame, isin_map: dict) -> pd.DataFrame:
#     if df.empty or "ISIN" not in df.columns:
#         return df
#     out = df.copy()
#     out["ISIN"] = out["ISIN"].astype(str).str.strip().str.upper()
#     out["SECID"] = out["ISIN"].map(isin_map).fillna("")
#     # Вставляем SECID сразу после ISIN
#     cols = list(out.columns)
#     if "SECID" in cols:
#         isin_idx = cols.index("ISIN")
#         cols.remove("SECID")
#         cols.insert(isin_idx + 1, "SECID")
#         out = out[cols]
#     return out
        
# ============================================================================
# Основная функция рейтинга
# ============================================================================


# ============================================================================
# Точка входа
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    import argparse
    
    ap = argparse.ArgumentParser(description="Module 2: рейтинг облигаций")
    ap.add_argument("--date", default="", help="Дата расчёта YYYY-MM-DD")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Путь к config.yaml")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Папка с входными CSV")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Папка для выходных CSV")
    args = ap.parse_args(argv)
    
    project_root = Path(__file__).resolve().parent.parent
    config_path = (Path(args.config) if Path(args.config).is_absolute() else project_root / args.config)
    data_dir = (Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir)
    output_dir = (Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir)
    
    config = load_config(config_path)
    date_from_config = config.get("date", "today")
    if not date_from_config or str(date_from_config).lower() in ("today", "auto", ""):
        date = args.date or datetime.now().strftime("%Y-%m-%d")
    else:
        date = args.date or str(date_from_config)
    
    logger = setup_logging(date, output_dir)
    logger.info(f"=== Module 2 start: date={date} ===")
    
    diagnostics: list[dict] = []
    
    try:
        # Загрузка данных
        fetch_and_save_bonds_universe(data_dir, logger)
        fetch_and_save_bond_prices(data_dir, logger)
        
        # Рейтинг
        # Рейтинг (теперь возвращает 5 значений)
        core_ofz_ranking, long_ofz_ranking, corp_ranking, bond_allocation, filter_stats,df_ofz_all,df_corp_all = rank_bonds(date, config, data_dir, logger, diagnostics)
        
        
        # Сохранение
        save_csv(core_ofz_ranking, output_dir / "core_ofz_ranking.csv")
        save_csv(long_ofz_ranking, output_dir / "long_ofz_ranking.csv")
        save_csv(corp_ranking, output_dir / "corp_ranking.csv")
        save_csv(bond_allocation, output_dir / "bond_allocation.csv")
        save_csv(filter_stats, output_dir / "bond_filter_stats.csv") 
        save_csv(df_ofz_all, output_dir / "ofz_all_with_reasons.csv")
        save_csv(df_corp_all, output_dir / "corp_all_with_reasons.csv")
        # Обновление матрицы субпортфелей
        # try:
        #     update_bond_portfolio_units(
        #         core_ofz_ranking, long_ofz_ranking, corp_ranking,
        #         data_dir, logger
        #     )
        # except Exception as e:
        #     logger.error(f"Ошибка обновления матрицы: {e}")
        #     diagnostics.append({"code": "portfolio_units_update_failed", "details": str(e)})
                
        logger.info(f"core_ofz_ranking: {len(core_ofz_ranking)} строк")
        logger.info(f"long_ofz_ranking: {len(long_ofz_ranking)} строк")
        logger.info(f"corp_ranking: {len(corp_ranking)} строк")
        
        if diagnostics:
            save_csv(pd.DataFrame(diagnostics), output_dir / "diagnostics.csv")
        
        logger.info("=== Module 2 finished ===")
        return 0
    
    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())