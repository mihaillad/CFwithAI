#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 4. HTML-отчёт по рейтингам и портфелю.
Входы : stock_ranking.csv, ofz_ranking.csv, corp_ranking.csv (из output_dir)
        portfolio_details.csv (из output_dir, формируется module3)
        stocks_universe.csv (из raw_dir — для ISIN и логотипов)
Выход : report_<date>.html
"""
from __future__ import annotations

import html
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ============================================================================
# Константы
# ============================================================================
DEFAULT_INPUT_DIR = Path("data/output")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_CONFIG_PATH = Path("config.yaml")
LOGO_URL = "https://mybroker.storage.bcs.ru/FinInstrumentLogo/{isin}.png"

STOCK_COLUMNS = [
    ("rank", "№", "int"),
    ("SECID", "Тикер", "ticker"),
    ("price", "Цена, ₽", "num"),
    ("dps_reg_next", "DPS, ₽", "num"),
    ("dps_source", "Источник", "txt"),
    ("record_date", "Отсечка", "date"),
    ("days_to_record", "Дней", "int"),
    ("ay_pct", "AY, %", "pct"),
    ("dy_gross_pct", "DY, %", "pct"),
    ("stop_flag", "Стоп", "bool"),
    ("target_weight_portfolio_pct", "Вес портф., %", "pct"),
    ("lot_size", "Лот", "int"),
]

BOND_COLUMNS = [
    ("rank", "№", "int"),
    ("ISIN", "ISIN", "txt"),
    ("name", "Название", "txt"),
    ("maturity_date", "Погашение", "date"),
    ("days_to_maturity", "Дней", "int"),
    ("coupon_rate_pct", "Купон, %", "pct"),
    ("price_pct", "Цена, %", "num"),
    ("ytm_gross_pct", "YTM gross, %", "pct"),
    ("spread_to_cbr_pct", "Спред, %", "pct"),
    ("duration_years", "Дюрация", "num"),
    ("credit_rating", "Рейтинг", "txt"),
    ("eligible", "Допущена", "bool"),
    ("exclusion_reason", "Исключена из-за", "txt"),
    ("target_weight_portfolio_pct", "Вес портф., %", "pct"),
]

PORTFOLIO_COLUMNS = [
    ("SECID_or_ISIN", "Тикер/ISIN", "ticker"),
    ("asset_type", "Тип", "txt"),
    ("quantity", "Кол-во", "num"),
    ("current_value", "Сумма тек. ₽", "money"),
    ("target_value", "Сумма план. ₽", "money"),
    ("deviation_value", "Откл. ₽", "money"),
    ("current_weight_pct", "Доля текущ., %", "pct"),
    ("target_weight_pct", "Доля план., %", "pct"),
    ("cost_basis", "Себестоимость, ₽", "money"),
    ("accounting_profit", "Прибыль уч., ₽", "money"),
    ("accounting_profit_pct", "Прибыль уч., %", "pct"),
    ("years_held", "Лет вл.", "num"),
    ("dividend_sum", "Див. ₽", "money"),
    ("dividend_pct", "Див. %", "pct"),
]

HEAT_COLUMNS = {"ay_pct", "ytm_gross_pct"}


# ============================================================================
# Утилиты
# ============================================================================
def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("module4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(output_dir / f"report_{date}.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_config(config_path: Path) -> dict:
    defaults = {"date": "today"}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            defaults.update(yaml.safe_load(f) or {})
    return defaults


def read_csv_flex(path: Path) -> pd.DataFrame:
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep=";")


def load_table(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Файл не найден, секция пропущена: {path.name}")
        return pd.DataFrame()
    df = read_csv_flex(path)
    logger.info(f"Загружено {path.name}: {len(df)} строк, {len(df.columns)} колонок")
    return df


def load_isin_map(raw_dir: Path, logger: logging.Logger) -> dict:
    """Загружает SECID -> ISIN из stocks_universe.csv."""
    universe_path = raw_dir / "stocks_universe.csv"
    if not universe_path.exists():
        logger.warning(f"stocks_universe.csv не найден в {raw_dir} — логотипы для акций недоступны")
        return {}
    try:
        u = read_csv_flex(universe_path)
        if "SECID" in u.columns and "ISIN" in u.columns:
            m = dict(zip(u["SECID"].astype(str), u["ISIN"].astype(str)))
            logger.info(f"ISIN-мапа загружена: {len(m)} тикеров")
            return m
    except Exception as e:
        logger.warning(f"Не удалось загрузить ISIN-мапу: {e}")
    return {}


# ============================================================================
# Форматирование
# ============================================================================
def _is_nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def fmt_cell(value, kind: str) -> tuple[str, str]:
    if _is_nan(value):
        return "", ""
    if kind == "bool":
        flag = str(value).strip().lower() in ("true", "1", "1.0", "yes", "да")
        return ("Да" if flag else "Нет"), ("1" if flag else "0")
    if kind in ("num", "pct", "int"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return html.escape(str(value)), html.escape(str(value))
        if kind == "int":
            return f"{int(f):,}".replace(",", " "), f"{f:.6f}"
        s = f"{f:,.2f}".replace(",", " ")
        return (s + " %") if kind == "pct" else s, f"{f:.6f}"
    s = str(value)
    return html.escape(s), html.escape(s)


def fmt_money(v) -> str:
    try:
        f = float(v)
        return f"{f:,.2f} ₽".replace(",", " ")
    except (TypeError, ValueError):
        return str(v) if not _is_nan(v) else ""


def fmt_cell_ext(value, kind: str) -> tuple[str, str]:
    if _is_nan(value):
        return "", ""
    if kind == "money":
        try:
            f = float(value)
            s = f"{f:,.2f}".replace(",", " ")
            return s, f"{f:.2f}"
        except (TypeError, ValueError):
            return html.escape(str(value)), html.escape(str(value))
    return fmt_cell(value, kind)


def heat_style(value, vmax: float, vmin: float) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if vmax <= vmin:
        return ""
    t = max(0.0, min(1.0, (f - vmin) / (vmax - vmin)))
    return f"background:rgba(46,160,67,{0.08 + 0.42 * t:.3f});"


def ticker_cell(secid, isin) -> str:
    """Ячейка с логотипом и тикером (по аналогии с dividend_rating.py)."""
    secid_s = html.escape(str(secid))
    if not isin or _is_nan(isin) or str(isin).strip() == "":
        return secid_s
    isin_s = html.escape(str(isin))
    src = LOGO_URL.format(isin=isin_s)
    return (f'<span class="tk">'
            f'<img class="logo" src="{src}" alt="" width="20" height="20" '
            f'loading="lazy" onerror="this.style.display=\'none\'">'
            f'{secid_s}</span>')


# ============================================================================
# Рендер таблиц
# ============================================================================
def render_table(df: pd.DataFrame, columns: list, title: str, table_id: str,
                 logger: logging.Logger, eligible_col: str | None = None,
                 isin_map: dict | None = None) -> str:
    if df.empty:
        return f'<section id="{table_id}" class="tabpane"><h2>{html.escape(title)}</h2>' \
               f'<p class="empty">Нет данных</p></section>'

    cols = [(c, h, k) for c, h, k in columns if c in df.columns]
    missing = [c for c, _, _ in columns if c not in df.columns]
    if missing:
        logger.info(f"{title}: отсутствуют колонки {missing}")

    # Счётчики в заголовке
    total = len(df)
    eligible_n = total
    if eligible_col and eligible_col in df.columns:
        eligible_n = int(df[eligible_col].astype(str).str.strip().str.lower()
                         .isin(["true", "1"]).sum())
    header = f"{html.escape(title)} <span class=\"cnt\">{eligible_n}/{total}</span>"

    # Диапазоны для тепловой раскраски
    ranges = {}
    for c, _, _ in cols:
        if c in HEAT_COLUMNS:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if not s.empty:
                ranges[c] = (float(s.min()), float(s.max()))

    head = "".join(
        f'<th data-kind="{k}" onclick="sortTable(this)">{html.escape(h)}'
        f'<span class="arr"></span></th>'
        for _, h, k in cols
    )

    rows_html = []
    for _, row in df.iterrows():
        cls = []
        stop = str(row.get("stop_flag", "")).strip().lower() in ("true", "1")
        elig = row.get(eligible_col, "") if eligible_col else ""
        elig_ok = str(elig).strip().lower() in ("true", "1")
        if stop:
            cls.append("row-stop")
        if elig_ok:
            cls.append("row-ok")
        elif eligible_col and eligible_col in df.columns and not _is_nan(elig):
            cls.append("row-out")

        tds = []
        for c, _, k in cols:
            val = row.get(c)
            style = ""
            if c in ranges and not _is_nan(val):
                style = heat_style(val, *ranges[c])

            if k == "ticker":
                # Тикер с логотипом
                secid = val
                isin = None
                if isin_map:
                    # Для акций — SECID; для облигаций — ISIN
                    if c == "SECID":
                        isin = isin_map.get(str(secid), "")
                    elif c == "ISIN":
                        isin = secid
                disp = ticker_cell(secid, isin)
                raw = html.escape(str(secid))
                tds.append(f'<td data-v="{raw}">{disp}</td>')
            else:
                disp, raw = fmt_cell(val, k)
                num = k in ("num", "pct", "int")
                td_cls = "num" + (" heat" if style else "")
                tds.append(f'<td class="{td_cls}" style="{style}" data-v="{raw}">{disp}</td>')

        rows_html.append(f'<tr class="{" ".join(cls)}">{"".join(tds)}</tr>')

    return f"""
<section id="{table_id}" class="tabpane">
  <h2>{header}</h2>
  <input class="filter" type="text" placeholder="Фильтр…" oninput="filterRows(this,'{table_id}')">
  <div class="twrap">
    <table id="tbl_{table_id}">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</section>"""


def render_portfolio(df: pd.DataFrame, isin_map: dict, logger: logging.Logger,
                     default_sort: str = "deviation_value",
                     default_asc: bool = True) -> str:
    if df.empty:
        return '<section id="pane_portfolio" class="tabpane"><h2>Портфель</h2>' \
               '<p class="empty">Нет данных</p></section>'

    # Предустановленная сортировка по default_sort (NaN — в конец)
    if default_sort in df.columns:
        df = df.copy()
        df = df.sort_values(
            default_sort,
            ascending=default_asc,
            na_position="last",
            key=lambda s: pd.to_numeric(s, errors="coerce") if s.name == default_sort else s
        ).reset_index(drop=True)

    cols = [(c, h, k) for c, h, k in PORTFOLIO_COLUMNS if c in df.columns]
    
    head = "".join(
        f'<th data-kind="{k}" onclick="sortTable(this)">{html.escape(h)}'
        f'<span class="arr"></span></th>'
        for _, h, k in cols
    )

    rows_html = []
    for _, row in df.iterrows():
        secid = row["SECID_or_ISIN"]
        asset_type = str(row.get("asset_type", ""))
        # Для облигаций ISIN = SECID_or_ISIN; для акций — из мапы
        isin = isin_map.get(str(secid), "") if asset_type == "stock" else secid

        tds = []
        for c, _, k in cols:
            val = row.get(c)
            style = ""
            # Подсветка отклонений и прибыли
            if c in ("deviation_value", "accounting_profit", "accounting_profit_pct"):
                try:
                    f = float(val)
                    if f > 0:
                        style = "color:#1a7f37;font-weight:600;"
                    elif f < 0:
                        style = "color:#cf222e;font-weight:600;"
                except (TypeError, ValueError):
                    pass

            if k == "ticker":
                disp = ticker_cell(secid, isin)
                raw = html.escape(str(secid))
                tds.append(f'<td data-v="{raw}">{disp}</td>')
            else:
                disp, raw = fmt_cell_ext(val, k)
                num = k in ("num", "pct", "int", "money")
                td_cls = "num" if num else ""
                tds.append(f'<td class="{td_cls}" style="{style}" data-v="{raw}">{disp}</td>')

        rows_html.append(f'<tr>{"".join(tds)}</tr>')

    # Итоги по портфелю
    total_value = pd.to_numeric(df.get("current_value", pd.Series()), errors="coerce").sum()
    total_cost = pd.to_numeric(df.get("cost_basis", pd.Series()), errors="coerce").sum()
    total_profit = total_value - total_cost
    total_profit_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0
    total_div = pd.to_numeric(df.get("dividend_sum", pd.Series()), errors="coerce").sum()
    profit_color = "#1a7f37" if total_profit >= 0 else "#cf222e"

    summary_html = f"""
    <div class="pf-summary">
      <div><b>Стоимость:</b> {fmt_money(total_value)}</div>
      <div><b>Себестоимость:</b> {fmt_money(total_cost)}</div>
      <div><b>Прибыль:</b> <span style="color:{profit_color}">{fmt_money(total_profit)} ({total_profit_pct:+.2f}%)</span></div>
      <div><b>Ожидаемые дивиденды:</b> {fmt_money(total_div)}</div>
    </div>
    """

    return f"""
<section id="pane_portfolio" class="tabpane">
  <h2>Портфель <span class="cnt">({len(df)} позиций)</span></h2>
  {summary_html}
  <div class="twrap">
    <table id="tbl_pane_portfolio">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</section>"""


# ============================================================================
# HTML-каркас
# ============================================================================
CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;
     background:#f6f8fa;color:#24292f}
header{background:#24292f;color:#fff;padding:12px 24px}
header .sub{opacity:.85;font-size:13px}
main{padding:20px 24px;max-width:1700px;margin:0 auto}
nav{display:flex;gap:6px;margin-bottom:14px;border-bottom:1px solid #d0d7de}
nav button{border:1px solid #d0d7de;border-bottom:none;background:#eaeef2;
  padding:8px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
nav button.active{background:#fff;font-weight:600}
.tabpane{display:none;background:#fff;border:1px solid #d0d7de;
  border-radius:0 8px 8px 8px;padding:16px}
.tabpane.active{display:block}
.tabpane h2{margin:0 0 10px;font-size:16px}
.cnt{color:#57606a;font-weight:400;font-size:13px;margin-left:6px}
.filter{width:280px;padding:6px 10px;margin-bottom:10px;border:1px solid #d0d7de;
  border-radius:6px;font-size:13px}
.twrap{overflow:auto;max-height:72vh;border:1px solid #d0d7de;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid #eaeef2;white-space:nowrap;text-align:left}
th{position:sticky;top:0;background:#f6f8fa;cursor:pointer;user-select:none;
   font-weight:600;z-index:1}
th:hover{background:#eaeef2}
th .arr{margin-left:4px;opacity:.5;font-size:10px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:#f0f6ff}
.row-stop{color:#8b1a1a;background:#fff5f5}
.row-ok td:first-child{box-shadow:inset 3px 0 0 #2ea043}
.row-out{opacity:.55}
.empty{color:#57606a}
.tk{display:inline-flex;align-items:center;gap:6px}
.logo{width:20px;height:20px;flex:none;border-radius:50%;object-fit:contain;
      background:#fff;border:1px solid #d0d7de;vertical-align:middle}
.pf-summary{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;
            padding:12px 16px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px}
.pf-summary div{font-size:14px}
@media print{.filter,nav{display:none}.tabpane{display:block!important;page-break-before:always}}
"""

JS = """
function showTab(id,btn){
  document.querySelectorAll('.tabpane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}
function filterRows(inp,pid){
  const q=inp.value.toLowerCase();
  document.querySelectorAll('#tbl_'+pid+' tbody tr').forEach(tr=>{
    tr.style.display = tr.textContent.toLowerCase().includes(q)?'':'none';
  });
}
function sortTable(th){
  const tbl=th.closest('table'), tb=tbl.tBodies[0];
  const idx=[...th.parentNode.children].indexOf(th);
  const kind=th.dataset.kind, asc=th.dataset.asc!=='1';
  tbl.querySelectorAll('th').forEach(h=>{h.dataset.asc='';h.querySelector('.arr').textContent='';});
  th.dataset.asc=asc?'1':'0';
  th.querySelector('.arr').textContent=asc?'▲':'▼';
  const rows=[...tb.rows];
  rows.sort((a,b)=>{
    const va=a.cells[idx].dataset.v??'', vb=b.cells[idx].dataset.v??'';
    let r;
    if(['num','pct','int','money'].includes(kind)){
      const na=parseFloat(String(va).replace(/\\s/g,'').replace('₽','')),
            nb=parseFloat(String(vb).replace(/\\s/g,'').replace('₽',''));
      r=(isNaN(na)?-Infinity:na)-(isNaN(nb)?-Infinity:nb);
    } else r=va.localeCompare(vb,'ru');
    return asc?r:-r;
  });
  rows.forEach(r=>tb.appendChild(r));
}
document.addEventListener('DOMContentLoaded',()=>{
  const b=document.querySelector('nav button'); if(b) b.click();
});
"""


def build_html(date: str, panes: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт {date}</title><style>{CSS}</style></head>
<body>
<header><div class="sub">Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div></header>
<main>
<nav>
  <button onclick="showTab('pane_stocks',this)">Акции</button>
  <button onclick="showTab('pane_ofz',this)">ОФЗ</button>
  <button onclick="showTab('pane_corp',this)">Корпораты</button>
  <button onclick="showTab('pane_portfolio',this)">Портфель</button>
</nav>
{panes}
</main>
<script>{JS}</script>
</body></html>"""


# ============================================================================
# Основная функция
# ============================================================================
def build_report(date: str, input_dir: Path, output_dir: Path,
                 logger: logging.Logger) -> Path:
    stocks = load_table(input_dir / "stock_ranking.csv", logger)
    ofz = load_table(input_dir / "ofz_ranking.csv", logger)
    corp = load_table(input_dir / "corp_ranking.csv", logger)
    portfolio = load_table(output_dir / "portfolio_details.csv", logger)

    # Фоллбэк для старого bond_ranking.csv
    if ofz.empty and corp.empty:
        legacy = input_dir / "bond_ranking.csv"
        if legacy.exists():
            df = load_table(legacy, logger)
            if not df.empty and "bond_type" in df.columns:
                ofz = df[df["bond_type"] == "ofz"].reset_index(drop=True)
                corp = df[df["bond_type"] == "corporate"].reset_index(drop=True)

    # ISIN-мапа из stocks_universe (raw_dir = input_dir.parent / "raw")
    raw_dir = input_dir.parent / "raw"
    isin_map = load_isin_map(raw_dir, logger)

    panes = (
        render_table(stocks, STOCK_COLUMNS, "Акции", "pane_stocks", logger,
                     eligible_col="eligible_for_buy", isin_map=isin_map)
        + render_table(ofz, BOND_COLUMNS, "ОФЗ", "pane_ofz", logger,
                       eligible_col="eligible", isin_map=isin_map)
        + render_table(corp, BOND_COLUMNS, "Корпораты", "pane_corp", logger,
                       eligible_col="eligible", isin_map=isin_map)
        + render_portfolio(portfolio, isin_map, logger)
    )

    out = output_dir / f"report_{date}.html"
    out.write_text(build_html(date, panes), encoding="utf-8")
    logger.info(f"HTML-отчёт сохранён: {out}")
    return out


# ============================================================================
# Точка входа
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Module 4: HTML-отчёт")
    ap.add_argument("--date", default="")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--open", action="store_true", help="Открыть отчёт в браузере")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p
    config_path, input_dir, output_dir = (
        resolve(args.config), resolve(args.input_dir), resolve(args.output_dir)
    )

    cfg = load_config(config_path)
    d = cfg.get("date", "today")
    date = args.date or (
        datetime.now().strftime("%Y-%m-%d")
        if not d or str(d).lower() in ("today", "auto", "") else str(d)
    )

    logger = setup_logging(date, output_dir)
    logger.info(f"=== Module 4 start: date={date} ===")
    try:
        out = build_report(date, input_dir, output_dir, logger)
        if args.open:
            if sys.platform == "win32":
                import os
                try:
                    os.startfile(str(out))
                except Exception:
                    webbrowser.open(out.as_uri())
            else:
                webbrowser.open(out.as_uri())
        logger.info("=== Module 4 finished ===")
        return 0
    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())