import pandas as pd
import numpy as np
from pathlib import Path
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# ============================================================================
# Константы источников данных
# ============================================================================
BCS_URL = "https://be.broker.ru/bcsexpress-partners-gateway/express-divcalendar/api/v2/dividends"
BCS_PARAMS = {"isActual": "true", "limit": "500", "order": "2", "sorting": "0", "isForeign": "false"}
MOEX_PRICES_URL = "http://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
CBR_URL = "http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
HTTP_TIMEOUT = 30

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
    - добавляем свежие данные из БКС (и факты, и будущие анонсы).
    """
    if divs_path.exists():
        old = pd.read_csv(divs_path)
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


# ============================================================================
# Основная функция модуля
# ============================================================================

def rank_stocks(date: str, config: dict, data_dir: Path,
                logger: logging.Logger, diagnostics: list):
    date_dt = pd.to_datetime(date)

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
    # Сохраняем divs_data.csv с разделителем ',' (как в исходном формате)
    divs_df.to_csv(divs_path, index=False, encoding='utf-8')
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

    # Список SECID, полностью исключённых реестром спецдивидендов
    bad_registry_secids = set()
    if not registry.empty and 'classification' in registry.columns:
        bad_registry_secids = set(
            registry[registry['classification'].isin(['special', 'uncertain'])]['SECID'].unique()
        )

    # ------------------------------------------------------------------
    # Шаг 3. Расчёт рейтинга по каждой бумаге
    # ------------------------------------------------------------------
    results = []
    excluded = []

    for _, row in universe.iterrows():
        secid = row['SECID']
        lot_size = int(row['lot_size']) if pd.notna(row.get('lot_size')) else 1

        # 3.1 Проверка цены
        if secid not in latest_prices.index:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_price', 'details': 'No recent price'})
            continue
        price = latest_prices.loc[secid, 'close']
        if pd.isna(price) or price <= 0:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_price', 'details': 'Price NaN or <= 0'})
            continue

        # 3.2 Исключение по реестру спецдивидендов
        if secid in bad_registry_secids:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'registry_missing',
                             'details': 'Excluded by special_dividends_registry'})
            continue

        # 3.3 Поиск ближайшего дивиденда из divs_data.csv
        secid_divs = divs_df[divs_df['SECID'] == secid].copy()
        secid_divs = secid_divs.rename(columns={
            'close_date': 'record_date',
            'div_sum': 'amount_per_share'
        })

        future_divs = secid_divs[secid_divs['record_date'] > date_dt]
        if future_divs.empty:
            # Fallback по политике (§2.8)
            if not config.get('registry_strict_mode', False):
                fund_row = fundamentals[fundamentals['SECID'] == secid]
                if (not fund_row.empty
                        and 'dividend_policy_url' in fund_row.columns
                        and pd.notna(fund_row['dividend_policy_url'].iloc[-1])):
                    logger.warning(f"policy_fallback for {secid}")
                    diagnostics.append({'code': 'policy_fallback',
                                        'details': f"SECID {secid}"})
                    # В режиме warn_and_uncertain всё равно исключаем,
                    # т.к. в divs_data нет будущего дивиденда
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'no_dividend',
                             'details': 'No future dividends in divs_data'})
            continue

        future_divs = future_divs.sort_values('record_date')
        next_div = future_divs.iloc[0]

        days_to_record = (next_div['record_date'] - date_dt).days
        if days_to_record < config['d_min_days']:
            excluded.append({'date': date, 'SECID': secid,
                             'reason': 'too_close_to_record',
                             'details': f'd={days_to_record}'})
            continue

        dps = next_div['amount_per_share']

        # 3.4 Расчёт метрик
        ay_pct = ((dps / price) / max(days_to_record, config['d_min_days'])
                  * 365 * 100)
        dy_gross_pct = (dps / price) * 100

        # 3.5 Фундаментальные показатели
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

        # 3.6 Стоп-лист
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
            'name': row.get('name', ''),
            'issuer': row.get('issuer', ''),
            'sector': row.get('sector', ''),
            'price': price,
            'dps_reg_next': dps,
            'dps_source': 'declared',  # в divs_data все — объявленные
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
        df_res = df_res.sort_values('ay_pct', ascending=False).reset_index(drop=True)
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