#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 4. HTML-отчёт по рейтингам.

Собирает из data/output/*.csv один самодостаточный HTML-файл
(inline CSS + JS, без внешних зависимостей, открывается офлайн).

Входы : stock_ranking.csv, ofz_ranking.csv, corp_ranking.csv
Выход : report_<date>.html

Запуск:
    python src/module4_report.py
    python -m src.module4_report --date 2026-10-18 --open
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

# (колонка, заголовок, тип)  тип: txt | num | int | pct | bool | date
STOCK_COLUMNS = [
    ("rank", "№", "int"),
    ("SECID", "Тикер", "txt"),
    ("name", "Название", "txt"),
    ("price", "Цена, ₽", "num"),
    ("dps_reg_next", "DPS, ₽", "num"),
    ("dps_source", "Источник", "txt"),
    ("record_date", "Отсечка", "date"),
    ("days_to_record", "Дней", "int"),
    ("ay_pct", "AY, %", "pct"),
    ("dy_gross_pct", "DY, %", "pct"),
    ("payout_fwd_pct", "Payout, %", "pct"),
    ("eps_growth_yoy_pct", "EPS YoY, %", "pct"),
    ("stop_flag", "Стоп", "bool"),
    ("stop_reason", "Причина стопа", "txt"),
    ("target_weight_portfolio_pct", "Вес портф., %", "pct"),
    ("lot_size", "Лот", "int"),
]

BOND_COLUMNS = [
    ("rank", "№", "int"),
    ("ISIN", "ISIN", "txt"),
    ("name", "Название", "txt"),
    ("issuer", "Эмитент", "txt"),
    ("maturity_date", "Погашение", "date"),
    ("days_to_maturity", "Дней", "int"),
    ("coupon_type", "Купон", "txt"),
    ("coupon_rate_pct", "Купон, %", "pct"),
    ("price_pct", "Цена, %", "num"),
    ("nkd_rub", "НКД, ₽", "num"),
    ("ytm_gross_pct", "YTM gross, %", "pct"),
    ("ytm_net_pct", "YTM net, %", "pct"),
    ("spread_to_cbr_pct", "Спред, %", "pct"),
    ("duration_years", "Дюрация", "num"),
    ("credit_rating", "Рейтинг", "txt"),
    ("bucket", "Бакет", "txt"),
    ("eligible", "Допущена", "bool"),
    ("exclusion_reason", "Исключена из-за", "txt"),
    ("target_weight_portfolio_pct", "Вес портф., %", "pct"),
    ("lot_size", "Лот", "int"),
]

# Колонки для тепловой раскраски
HEAT_COLUMNS = {"ay_pct", "ytm_gross_pct"}


# ============================================================================
# Утилиты
# ============================================================================
def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    """Лог отчёта. Отдельное имя, чтобы не перезаписать run_<date>.log модулей 1-2."""
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
    """Читает CSV с автоопределением разделителя (';' стандарт проекта, ',' — фоллбэк)."""
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


# ============================================================================
# Форматирование ячеек
# ============================================================================
def _is_nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def fmt_cell(value, kind: str) -> tuple[str, str]:
    """Возвращает (отображаемый_текст, сырое_значение_для_сортировки)."""
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


def heat_style(value, vmax: float, vmin: float) -> str:
    """Фоновая подсветка 0..1 от минимума к максимуму колонки."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if vmax <= vmin:
        return ""
    t = max(0.0, min(1.0, (f - vmin) / (vmax - vmin)))
    # зелёный градиент
    return f"background:rgba(46,160,67,{0.08 + 0.42 * t:.3f});"


# ============================================================================
# Рендер таблицы
# ============================================================================
def render_table(df: pd.DataFrame, columns: list[tuple[str, str, str]],
                 title: str, table_id: str, logger: logging.Logger) -> str:
    if df.empty:
        return f'<section id="{table_id}" class="tabpane"><h2>{html.escape(title)}</h2>' \
               f'<p class="empty">Нет данных</p></section>'

    cols = [(c, h, k) for c, h, k in columns if c in df.columns]
    missing = [c for c, _, _ in columns if c not in df.columns]
    if missing:
        logger.info(f"{title}: отсутствуют колонки {missing} — пропущены")

    # диапазоны для тепловой раскраски
    ranges: dict[str, tuple[float, float]] = {}
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
        # класс строки
        cls = []
        stop = str(row.get("stop_flag", "")).strip().lower() in ("true", "1")
        elig = row.get("eligible", row.get("eligible_for_buy", ""))
        elig_ok = str(elig).strip().lower() in ("true", "1")
        if stop:
            cls.append("row-stop")
        if elig_ok:
            cls.append("row-ok")
        elif "eligible" in df.columns and not _is_nan(elig):
            cls.append("row-out")

        tds = []
        for c, _, k in cols:
            disp, raw = fmt_cell(row.get(c), k)
            style = ""
            if c in ranges and not _is_nan(row.get(c)):
                style = heat_style(row.get(c), *ranges[c])
            num = k in ("num", "pct", "int")
            td_cls = "num" + (" heat" if style else "")
            tds.append(f'<td class="{td_cls}" style="{style}" data-v="{raw}">{disp}</td>')

        rows_html.append(f'<tr class="{" ".join(cls)}">{"".join(tds)}</tr>')

    return f"""
<section id="{table_id}" class="tabpane">
  <h2>{html.escape(title)} <span class="cnt">({len(df)})</span></h2>
  <input class="filter" type="text" placeholder="Фильтр…" oninput="filterRows(this,'{table_id}')">
  <div class="twrap">
    <table id="tbl_{table_id}">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</section>"""


# ============================================================================
# Сводка
# ============================================================================
def render_summary(sections: list[tuple[str, pd.DataFrame, str, str]]) -> str:
    cards = []
    for title, df, key_col, id_ in sections:
        if df.empty:
            continue
        n = len(df)
        elig_col = "eligible_for_buy" if "eligible_for_buy" in df.columns else "eligible"
        n_ok = 0
        if elig_col in df.columns:
            n_ok = int(df[elig_col].astype(str).str.strip().str.lower().isin(["true", "1"]).sum())
        avg = ""
        if key_col in df.columns:
            s = pd.to_numeric(df[key_col], errors="coerce").dropna()
            if not s.empty:
                avg = f"{s.mean():.2f}%"
        top = ""
        if key_col in df.columns and "rank" in df.columns:
            d = df.dropna(subset=[key_col]).sort_values("rank").head(1)
            if not d.empty:
                code = d.iloc[0].get("SECID", d.iloc[0].get("ISIN", ""))
                top = f"{code}: {float(d.iloc[0][key_col]):.2f}%"
        cards.append(
            f'<div class="card"><div class="card-t">{html.escape(title)}</div>'
            f'<div class="card-v">{avg or "—"}</div>'
            f'<div class="card-s">всего {n} · допущено {n_ok}<br>топ-1: {html.escape(str(top))}</div></div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>' if cards else ""


# ============================================================================
# Сборка HTML
# ============================================================================
CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;
     background:#f6f8fa;color:#24292f}
header{background:#24292f;color:#fff;padding:16px 24px}
header h1{margin:0;font-size:19px}
header .sub{opacity:.7;font-size:12px;margin-top:4px}
main{padding:20px 24px;max-width:1600px;margin:0 auto}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:12px 16px;min-width:220px}
.card-t{font-size:12px;color:#57606a;text-transform:uppercase;letter-spacing:.4px}
.card-v{font-size:24px;font-weight:600;margin:4px 0}
.card-s{font-size:12px;color:#57606a}
nav{display:flex;gap:6px;margin-bottom:14px;border-bottom:1px solid #d0d7de}
nav button{border:1px solid #d0d7de;border-bottom:none;background:#eaeef2;
  padding:8px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
nav button.active{background:#fff;font-weight:600}
.tabpane{display:none;background:#fff;border:1px solid #d0d7de;
  border-radius:0 8px 8px 8px;padding:16px}
.tabpane.active{display:block}
.tabpane h2{margin:0 0 10px;font-size:16px}
.cnt{color:#57606a;font-weight:400;font-size:13px}
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
    if(['num','pct','int'].includes(kind)){
      const na=parseFloat(va), nb=parseFloat(vb);
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


def build_html(date: str, panes: str, summary: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Рейтинги {date}</title><style>{CSS}</style></head>
<body>
<header><h1>Инвестиционные рейтинги — {date}</h1>
<div class="sub">Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div></header>
<main>
{summary}
<nav>
  <button onclick="showTab('pane_stocks',this)">Акции</button>
  <button onclick="showTab('pane_ofz',this)">ОФЗ</button>
  <button onclick="showTab('pane_corp',this)">Корпораты</button>
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

    # Фоллбэк: старый единый bond_ranking.csv
    if ofz.empty and corp.empty:
        legacy = input_dir / "bond_ranking.csv"
        if legacy.exists():
            logger.info("ofz/corp рейтинги пусты, разбираю legacy bond_ranking.csv")
            df = load_table(legacy, logger)
            if not df.empty and "bond_type" in df.columns:
                ofz = df[df["bond_type"] == "ofz"].reset_index(drop=True)
                corp = df[df["bond_type"] == "corporate"].reset_index(drop=True)

    panes = (
        render_table(stocks, STOCK_COLUMNS, "Акции (рейтинг по AY)", "pane_stocks", logger)
        + render_table(ofz, BOND_COLUMNS, "ОФЗ (ядро 2,5–3 года)", "pane_ofz", logger)
        + render_table(corp, BOND_COLUMNS, "Короткие корпораты (до 1 года)", "pane_corp", logger)
    )
    summary = render_summary([
        ("Акции · средний AY", stocks, "ay_pct", "pane_stocks"),
        ("ОФЗ · средний YTM", ofz, "ytm_gross_pct", "pane_ofz"),
        ("Корпораты · средний YTM", corp, "ytm_gross_pct", "pane_corp"),
    ])

    out = output_dir / f"report_{date}.html"
    out.write_text(build_html(date, panes, summary), encoding="utf-8")
    logger.info(f"HTML-отчёт сохранён: {out}")
    return out


# ============================================================================
# Точка входа
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Module 4: HTML-отчёт по рейтингам")
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
            webbrowser.open(out.as_uri())
        logger.info("=== Module 4 finished ===")
        return 0
    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())