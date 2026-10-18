from __future__ import annotations

import logging
import math
import os
import sys
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
BCS_PARAMS = {"isActual": "true", "limit": "500", "order": "2", "sorting": "0", "isForeign": "false"}
MOEX_PRICES_URL = "http://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
CBR_URL = "http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
HTTP_TIMEOUT = 30


# ============================================================================
# Встроенные утилиты (для автономного запуска, без common.py)
# ============================================================================

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")

REQUIRED_RAW_FILES = [
    "stocks_universe.csv",
    "divs_data.csv",           # может отсутствовать — будет создан из БКС
    "stock_prices.csv",        # может отсутствовать — будет создан из MOEX
    "risk_free_rate.csv",      # может отсутствовать — будет создан из ЦБ
]

# Файлы, отсутствие которых критично (не могут быть получены из внешних источников)
CRITICAL_RAW_FILES = [
    "stocks_universe.csv",
]


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
        "date": datetime.now().strftime("%Y-%m-%d"),
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
    """Сохраняет DataFrame в CSV с разделителем ; и заданной точностью float."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8",
              float_format=float_format, date_format="%Y-%m-%d")


def check_required_files(data_dir: Path, logger: logging.Logger) -> None:
    """
    Проверяет наличие критичных входных файлов.
    - Если stocks_universe.csv отсутствует, загружает его с MOEX ISS.
    - Если stock_fundamentals.csv отсутствует, создаёт пустой файл с требуемой схемой.
    Остальные файлы (divs_data, stock_prices, risk_free_rate) получаются из внешних источников.
    """
    # 1. Проверка справочника акций
    universe_path = data_dir / "stocks_universe.csv"
    if not universe_path.exists():
        logger.warning("stocks_universe.csv не найден. Загружаю актуальный справочник с MOEX ISS...")
        fetch_and_save_stocks_universe(data_dir, logger)

    # 2. Проверка и создание stock_fundamentals.csv
    fund_path = data_dir / "stock_fundamentals.csv"
    if not fund_path.exists():
        logger.info("stock_fundamentals.csv не найден. Создаю пустой файл с требуемой схемой...")
        
        # Создаем пустой DataFrame с точными именами колонок из ТЗ
        empty_fund = pd.DataFrame(columns=[
            "SECID", "report_date", "period", "eps_ttm", "eps_prev_ttm",
            "eps_3y_ago", "net_income_ttm", "shares_outstanding",
            "dividend_policy_url", "notes"
        ])
        
        # Явное приведение типов для корректного сохранения и последующего чтения
        empty_fund = empty_fund.astype({
            "SECID": str,
            "report_date": str,       # При чтении будет преобразован в datetime через pd.to_datetime
            "period": str,
            "eps_ttm": float,
            "eps_prev_ttm": float,
            "eps_3y_ago": float,
            "net_income_ttm": float,
            "shares_outstanding": float,
            "dividend_policy_url": str,
            "notes": str
        })
        
        save_csv(empty_fund, fund_path)
        logger.info(f"Пустой {fund_path.name} успешно создан.")

    # 3. Информационное логирование об остальных опциональных файлах
    optional = [
        "divs_data.csv", 
        "stock_prices.csv", 
        "risk_free_rate.csv", 
        "special_dividends_registry.csv"
    ]
    for f in optional:
        if not (data_dir / f).exists():
            logger.info(f"Опциональный файл отсутствует: {f} (будет получен из внешнего источника или пропущен)")
                        
            
# ============================================================================
# Загрузка данных из внешних источников (по аналогии с dividend_rating.py)
# ============================================================================

def fetch_bcs_calendar() -> dict:
    """Загружает актуальный календарь дивидендов с БКС."""
    r = requests.get(BCS_URL, params=BCS_PARAMS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Некорректный ответ БКС")
    return payload


def bcs_to_divs_data(payload: dict) -> pd.DataFrame:
    """
    Преобразует ответ БКС в формат divs_data.csv.
    В БКС: securCode -> SECID, recordDate -> close_date, size -> div_sum.
    """
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
                'SECID': str(secid).strip(),
                'close_date': pd.to_datetime(str(record_date)[:10]),
                'div_sum': float(div_sum)
            })
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Суммирование по дате (если БКС отдаёт дубли) и дедупликация
    df = df.groupby(['SECID', 'close_date'], as_index=False)['div_sum'].sum()
    return df


def update_divs_data(divs_path: Path, bcs_df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """
    Обновляет divs_data.csv:
    - из старого файла оставляем только записи <= today (история фактов);
    - будущие записи из old удаляем (могли быть устаревшими прогнозами);
    - добавляем свежие данные из БКС;
    - сохраняем результат с разделителем ';' согласно ТЗ §0.4.
    """
    if divs_path.exists():
        old = pd.read_csv(divs_path, sep=';')
        old['close_date'] = pd.to_datetime(old['close_date'])
        old['div_sum'] = pd.to_numeric(old['div_sum'], errors='coerce')
        old = old.dropna(subset=['SECID', 'close_date', 'div_sum'])
        # Оставляем только факты (прошедшие даты)
        old = old[old['close_date'] <= today]
    else:
        old = pd.DataFrame(columns=['SECID', 'close_date', 'div_sum'])

    merged = pd.concat([old, bcs_df], ignore_index=True)
    # Дедупликация: если одна и та же дата — берём новую (из БКС)
    merged = merged.sort_values('close_date').drop_duplicates(
        subset=['SECID', 'close_date'], keep='last'
    )
    merged = merged.sort_values(['SECID', 'close_date']).reset_index(drop=True)
    
    # Сохраняем с единым стандартом проекта
    save_csv(merged, divs_path, sep=";", float_format="%.9f")
    
    return merged


def fetch_moex_prices() -> pd.DataFrame:
    """
    Загружает текущие цены акций с ISS Мосбиржи (TQBR).
    Приоритет: LCURRENTPRICE -> LAST -> LCLOSEPRICE.
    """
    params = {
        "iss.meta": "off",
        "iss.only": "marketdata,securities",
        "marketdata.columns": "SECID,LCURRENTPRICE,LAST,LCLOSEPRICE",
        "securities.columns": "SECID,ISIN"
    }
    r = requests.get(MOEX_PRICES_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()

    if "marketdata" not in payload or "data" not in payload["marketdata"]:
        raise ValueError("В ответе ISS нет блока marketdata")

    md = pd.DataFrame(payload["marketdata"]["data"],
                      columns=payload["marketdata"]["columns"])

    for col in ['LCURRENTPRICE', 'LAST', 'LCLOSEPRICE']:
        if col in md.columns:
            md[col] = pd.to_numeric(md[col], errors='coerce')

    # Выбираем лучшую доступную цену
    md['close'] = np.nan
    for col in ['LCURRENTPRICE', 'LAST', 'LCLOSEPRICE']:
        if col in md.columns:
            mask = md['close'].isna() & md[col].notna() & (md[col] > 0)
            md.loc[mask, 'close'] = md.loc[mask, col]

    md = md[['SECID', 'close']].dropna(subset=['close'])
    md['date'] = datetime.now().strftime('%Y-%m-%d')
    md['volume_rub'] = 0.0
    return md[['date', 'SECID', 'close', 'volume_rub']]


def fetch_cbr_rate() -> pd.DataFrame:
    """
    Загружает ключевую ставку ЦБ через SOAP-интерфейс.
    Возвращает DataFrame с колонками date, rate_pct.
    """
    today = datetime.now()
    d_from = today - timedelta(days=370)
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
    r = requests.post(CBR_URL, data=body.encode('utf-8'),
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
                rows.append({'date': date[:10],
                             'rate_pct': float(rate.replace(",", "."))})
            except ValueError:
                continue
    if not rows:
        raise ValueError("В ответе ЦБ не найдено пар дата/ставка")
    return pd.DataFrame(rows).drop_duplicates('date').sort_values('date')


def fetch_and_save_stocks_universe(data_dir: Path, logger: logging.Logger) -> None:
    """
    Загружает актуальный справочник акций с Мосбиржи (ISS) и сохраняет в stocks_universe.csv.
    Если для одного SECID есть несколько досок (BOARDID), выбирается та, у которой выше
    уровень листинга (LISTLEVEL), а при равенстве — первая по алфавиту.
    """
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
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

    # Приведение типов
    df["LOTSIZE"] = pd.to_numeric(df["LOTSIZE"], errors="coerce").fillna(1).astype(int)
    df["LISTLEVEL"] = pd.to_numeric(df["LISTLEVEL"], errors="coerce").fillna(-1)

    # Сортировка для выбора приоритетной доски: выше уровень листинга, затем алфавитный порядок BOARDID
    df = df.sort_values(by=["LISTLEVEL", "BOARDID"], ascending=[False, True])

    # Оставляем по одной записи на SECID (приоритетная доска)
    df_unique = df.drop_duplicates(subset=["SECID"], keep="first").copy()

    # Формирование итоговых колонок согласно ТЗ (§1.2)
    # Примечание: MOEX ISS в массовом эндпоинте не отдаёт чистое название юрлица (issuer),
    # поэтому используем SHORTNAME. Поле sector справочное и не является фильтром.
    result = pd.DataFrame({
        "SECID": df_unique["SECID"],
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

def generate_forecast_dividends(divs_df: pd.DataFrame, today: pd.Timestamp, match_days: int = 45) -> pd.DataFrame:
    """
    Генерирует прогнозные дивиденды на основе выплат за последние 365 дней.
    Точная копия логики build_events из dividend_rating.py:
    - берём фактические выплаты за последние 365 дней;
    - сдвигаем дату на +365 дней, сумма остаётся той же;
    - если объявленный дивиденд уже есть в пределах ±match_days, прогноз удаляется;
    - оставшиеся прогнозы добавляются с dps_source='forecast'.
    """
    if divs_df.empty:
        res = divs_df.copy()
        if "dps_source" not in res.columns:
            res["dps_source"] = "declared"
        return res

    df = divs_df.copy()
    if "dps_source" not in df.columns:
        df["dps_source"] = "declared"

    # 1. Фактические выплаты за последние 365 дней
    hist = df[
        (df["close_date"] <= today) &
        (df["close_date"] > today - pd.Timedelta(days=365)) &
        (df["div_sum"] > 0)
    ].copy()

    if hist.empty:
        return df

    # 2. Прогноз: сдвиг даты на +365 дней, та же сумма
    fc = hist[["SECID", "close_date", "div_sum"]].copy()
    fc["close_date"] = fc["close_date"] + pd.Timedelta(days=365)
    fc = fc[fc["close_date"] > today].reset_index(drop=True)

    if fc.empty:
        return df

    # 3. Исключаем прогнозы, если объявленный дивиденд уже есть вблизи этой даты (± match_days)
    #    Используем подход из build_events: сохраняем индекс fc, чтобы удалять по нему
    declared = df[df["close_date"] > today][["SECID", "close_date"]].copy()

    if not declared.empty and not fc.empty:
        m = fc.reset_index().merge(
            declared.rename(columns={"close_date": "dec_date"}),
            on="SECID"
        )
        # Находим индексы fc, которые покрыты объявленными дивидендами
        covered_idx = m.loc[
            (m["close_date"] - m["dec_date"]).dt.days.abs() <= match_days,
            "index"
        ].unique()
        fc = fc.drop(index=covered_idx)

    fc["dps_source"] = "forecast"

    # 4. Объединяем исходные данные с прогнозом
    result = pd.concat([df, fc], ignore_index=True)

    # 5. Дедупликация: если на одну дату есть и declared, и forecast, оставляем declared
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
    check_required_files(data_dir, logger)

    # ------------------------------------------------------------------
    # Шаг 1. Обновление данных из внешних источников
    # ------------------------------------------------------------------

    # 1.1 Календарь БКС -> divs_data.csv
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
    save_csv(divs_df, divs_path, sep=";", float_format="%.9f")
    logger.info(f"divs_data.csv updated: {len(divs_df)} records, "
                f"{divs_df['SECID'].nunique()} tickers")

    # 1.2 Текущие цены с MOEX ISS -> stock_prices.csv
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

    # 1.3 Ключевая ставка ЦБ -> risk_free_rate.csv
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
    fundamentals = (pd.read_csv(fund_path, sep=';')
                    if fund_path.exists() else pd.DataFrame())

    # Приводим даты
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    latest_prices = (prices_df[prices_df['date'] <= date_dt]
                     .sort_values('date')
                     .groupby('SECID').tail(1)
                     .set_index('SECID'))

    divs_df['close_date'] = pd.to_datetime(divs_df['close_date'])
    
    # Генерация прогнозов на основе истории за 365 дней (аналог build_events)
    divs_df = generate_forecast_dividends(divs_df, date_dt, match_days=45)
    logger.info(f"divs_data после генерации прогнозов: {len(divs_df)} записей, "
                f"прогнозных: {(divs_df['dps_source'] == 'forecast').sum()}")

    # Список SECID, полностью исключённых реестром спецдивидендов
    bad_registry_secids = set()
    if not registry.empty and 'classification' in registry.columns:
        bad_registry_secids = set(
            registry[registry['classification'].isin(['special', 'uncertain'])]['SECID'].unique()
        )

    
    # ------------------------------------------------------------------
    # Шаг 3. Расчёт рейтинга: обходим только SECID с дивидендами
    # ------------------------------------------------------------------
    
    # 3.0 Собираем SECID, у которых есть будущие дивиденды (declared или forecast)
    future_divs = divs_df[divs_df['close_date'] > date_dt]
    active_secids = future_divs['SECID'].unique().tolist()
    logger.info(f"Активных SECID с будущими дивидендами: {len(active_secids)}")
    
    # # 3.0.1 Аудит: сколько тикеров из universe не имеют дивидендов
    # universe_secids = set(universe['SECID'])
    # without_divs = universe_secids - set(active_secids)
    # if without_divs:
    #     logger.info(f"В universe {len(without_divs)} тикеров без дивидендов (пропущены)")
    #     diagnostics.append({
    #         'code': 'universe_without_dividends',
    #         'details': f"count={len(without_divs)}"
    #     })
    
    # # 3.0.2 Проверка: есть ли SECID в divs_data, которых нет в universe
    # missing_in_universe = set(active_secids) - universe_secids
    # if missing_in_universe:
    #     logger.warning(f"SECID в дивидендах, но нет в universe: {len(missing_in_universe)}")
    #     diagnostics.append({
    #         'code': 'secid_not_in_universe',
    #         'details': f"tickers={','.join(sorted(missing_in_universe)[:20])}"
    #     })
    
    # 3.0.3 Создаём быстрый доступ к данным universe по SECID
    universe_dict = universe.set_index('SECID').to_dict('index')
    
    results = []
    excluded = []
    
    # 3.1 Обходим ТОЛЬКО SECID с будущими дивидендами
    for secid in active_secids:
        # Получаем данные из universe (если есть)
        uni_row = universe_dict.get(secid, {})
        lot_size = int(uni_row.get('lot_size', 1)) if uni_row else 1
        
        # 3.2 Проверка цены
        if secid not in latest_prices.index:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_price', 'details': 'No recent price'})
            continue
        price = latest_prices.loc[secid, 'close']
        if pd.isna(price) or price <= 0:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_price', 'details': 'Price NaN or <= 0'})
            continue
        
        # 3.3 Исключение по реестру спецдивидендов
        if secid in bad_registry_secids:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'registry_missing',
                             'details': 'Excluded by special_dividends_registry'})
            continue
        
        # 3.4 Поиск ближайшего дивиденда (теперь гарантированно есть)
        secid_divs = divs_df[divs_df['SECID'] == secid].copy()
        
        # Приоритет: declared (1) раньше forecast (2)
        secid_divs["source_priority"] = secid_divs["dps_source"].map({"declared": 1, "forecast": 2}).fillna(3)
        secid_divs = secid_divs.sort_values(["close_date", "source_priority"])
        
        secid_divs = secid_divs.rename(columns={
            'close_date': 'record_date',
            'div_sum': 'amount_per_share'
        })
        
        future_secid_divs = secid_divs[secid_divs['record_date'] > date_dt]
        
        # Этот блок теперь не нужен, т.к. мы обходим только SECID с дивидендами
        # Но оставим на случай edge cases
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
        
        # 3.5 Расчёт метрик
        ay_pct = ((dps / price) / max(days_to_record, config['d_min_days'])
                  * 365 * 100)
        dy_gross_pct = (dps / price) * 100
        
        # 3.6 Фундаментальные показатели
        fund_row = (fundamentals[fundamentals['SECID'] == secid]
                    .sort_values('report_date'))
        eps_ttm = np.nan
        payout_fwd = 0.0
        eps_yoy = np.nan
        eps_3y = np.nan
        if not fund_row.empty:
            last_fund = fund_row.iloc[-1]
            eps_ttm = last_fund.get('eps_ttm', np.nan)
            payout_fwd = last_fund.get('payout_fwd_pct', 0.0) or 0.0
            eps_yoy = last_fund.get('eps_growth_yoy_pct', np.nan)
            eps_3y = last_fund.get('eps_growth_3y_pct', np.nan)
        
        # 3.7 Стоп-лист
        stop_flag = False
        stop_reasons = []
        if dy_gross_pct < config['min_dy_gross_pct']:
            stop_flag = True
            stop_reasons.append("low_dy")
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
            'dps_source': dps_source,
           # 'name': uni_row.get('name', ''),
            # 'issuer': uni_row.get('issuer', ''),
            # 'sector': uni_row.get('sector', 'Unknown'),
            'price': price,
            'dps_reg_next': dps,
            'record_date': next_div['record_date'].strftime('%Y-%m-%d'),
            'days_to_record': days_to_record,
            'ay_pct': ay_pct,
            'dy_gross_pct': dy_gross_pct,
            'payout_fwd_pct': payout_fwd,
            'eps_growth_yoy_pct': eps_yoy,
            'eps_growth_3y_pct': eps_3y,
            'stop_flag': stop_flag,
            'stop_reason': ";".join(stop_reasons),
            'lot_size': lot_size
        })

    # ------------------------------------------------------------------
    # Шаг 4. Сортировка и нормировка весов
    # ------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # ------------------------------------------------------------------
        # Шаг 3.7. Исключение дубликатов классов (обычка vs преф)
        # Для пары SECID / SECID+P оставляем бумагу с большей AY;
        # при равенстве AY — с большей DY_gross (CY).
        # ------------------------------------------------------------------
        secs = set(df_res["SECID"])
        drop = []
        for s in list(secs):
            p = s + "P"
            if p in secs:
                a = df_res[df_res["SECID"] == s].iloc[0]
                b = df_res[df_res["SECID"] == p].iloc[0]
                # Сравниваем пары (ay_pct, dy_gross_pct)
                drop.append(p if (a["ay_pct"], a["dy_gross_pct"]) >= (b["ay_pct"], b["dy_gross_pct"]) else s)

        if drop:
            logger.info("обычка и префы: убраны %s", ", ".join(sorted(drop)))
            # Переносим исключённые бумаги в список excluded
            for secid in drop:
                excluded.append({
                    'date': date,
                    'SECID': secid,
                    'reason': 'duplicate_class',
                    'details': 'Обычка/преф: исключена бумага с меньшей AY/DY'
                })
            df_res = df_res[~df_res["SECID"].isin(drop)].copy().reset_index(drop=True)

            # Диагностика
            diagnostics.append({
                'code': 'duplicate_class_removed',
                'details': "; ".join(sorted(drop))
            })

        # Теперь сортируем и присваиваем ранги только оставшимся уникальным эмитентам
        # Приоритет сортировки:
        #   1) ay_pct — по убыванию (главный критерий рейтинга)
        #   2) dps_source — declared (1) раньше forecast (2) при равном AY
        df_res = df_res.copy()
        df_res['_src_priority'] = df_res['dps_source'].map({'declared': 1, 'forecast': 2}).fillna(3)
        df_res = df_res.sort_values(
            ['_src_priority','ay_pct' ],
            ascending=[True,False ]
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

    df_exc = pd.DataFrame(excluded)
    return df_res, df_exc

# ============================================================================
# Точка входа для изолированного запуска
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """CLI для отладки module1_stocks.py.

    Примеры:
        python -m src.module1_stocks
        python src/module1_stocks.py
        python src/module1_stocks.py --date 2026-10-17
        python src/module1_stocks.py --config config.yaml --data-dir data/raw
    """
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

    # Корневая директория проекта (от неё строятся относительные пути по умолчанию)
    project_root = Path(__file__).resolve().parent.parent
    config_path = (Path(args.config) if Path(args.config).is_absolute()
                   else project_root / args.config)
    data_dir = (Path(args.data_dir) if Path(args.data_dir).is_absolute()
                else project_root / args.data_dir)
    output_dir = (Path(args.output_dir) if Path(args.output_dir).is_absolute()
                  else project_root / args.output_dir)

    # Загружаем конфиг, чтобы узнать дату (если не задана через CLI)
    config = load_config(config_path)
    date = args.date or config.get("date") or datetime.now().strftime("%Y-%m-%d")

    logger = setup_logging(date, output_dir)
    if args.update_universe:
        logger.info("Принудительное обновление справочника акций...")
        fetch_and_save_stocks_universe(data_dir, logger)
    logger.info(f"=== Module 1 start: date={date}, data_dir={data_dir}, output_dir={output_dir} ===")

    diagnostics: list[dict] = []

    try:
        stock_ranking, stock_excluded = rank_stocks(date, config, data_dir, logger, diagnostics)

        save_csv(stock_ranking, output_dir / "stock_ranking.csv")
        save_csv(stock_excluded, output_dir / "stock_excluded.csv")

        logger.info(f"stock_ranking: {len(stock_ranking)} строк")
        logger.info(f"stock_excluded: {len(stock_excluded)} строк")
        if not stock_ranking.empty:
            top5 = stock_ranking.head(5)[["rank", "SECID", "ay_pct", "dy_gross_pct", "stop_flag"]]
            logger.info(f"Top-5:\n{top5.to_string(index=False)}")

        # diagnostics
        if diagnostics:
            save_csv(pd.DataFrame(diagnostics), output_dir / "diagnostics.csv")
            logger.info(f"diagnostics: {len(diagnostics)} записей")

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