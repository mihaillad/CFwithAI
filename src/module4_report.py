#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль 4. HTML-отчёт по рейтингам и портфелю.

Собирает из data/output/*.csv и data/raw/portfolio_units.csv один самодостаточный HTML-файл.
Входы:
  - stock_ranking.csv, ofz_ranking.csv, corp_ranking.csv (из output_dir)
  - portfolio_details.csv (из output_dir, формируется module3)
  - portfolio_units.csv (из raw_dir — матрица субпортфелей)
  - stocks_universe.csv (из raw_dir — для ISIN и логотипов)
Выход: report_<date>.html

Запуск:
    python src/module4_report.py
    python -m src.module4_report --date 2026-10-23 --open
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
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CONFIG_PATH = Path("config.yaml")

LOGO_URL = "https://mybroker.storage.bcs.ru/FinInstrumentLogo/{isin}.png"

# Авто-субпортфели (заполняются module1/module2)
AUTO_SUBPORTFOLIOS = {"Дивиденды", "ОФЗ", "Длинные ОФЗ", "Корпораты"}

# Колонки рейтингов
STOCK_COLUMNS = [
    ("rank", "№", "int"),
    ("SECID", "Тикер", "ticker"),
    ("name", "Название", "txt"),
    ("price", "Цена, ₽", "num"),
    ("dps_reg_next", "DPS, ₽", "num"),
    ("dps_source", "Источник", "txt"),
    ("record_date", "Отсечка", "date"),
    ("days_to_record", "Дней", "int"),
    ("ay_pct", "AY, %", "pct"),
    ("dy_gross_pct", "DY, %", "pct"),
    ("stop_flag", "Стоп", "bool"),
    ("target_weight_portfolio_pct", "Вес портф., %", "pct"),
    ("LOTSIZE", "Лот", "int"),
]

BOND_COLUMNS = [
    ("rank", "№", "int"),
    ("ISIN", "ISIN", "txt"),
    ("name", "Название", "txt"),
    ("issuer", "Эмитент", "txt"),
    ("maturity_date", "Погашение", "date"),
    ("days_to_maturity", "Дней", "int"),
    ("coupon_rate_pct", "Купон, %", "pct"),
    ("price_pct", "Цена, %", "num"),
    ("ytm_gross_pct", "YTM gross, %", "pct"),
    ("spread_to_cbr_pct", "Спред, %", "pct"),
    ("duration_years", "Дюрация", "num"),
    ("credit_rating", "Рейтинг", "txt"),
    ("eligible", "Допущена", "bool"),
    ("target_weight_portfolio_pct", "Вес портф., %", "pct"),
]

# Колонки портфеля (добавлены action и reason)
PORTFOLIO_COLUMNS = [
    ("SECID", "Тикер / ISIN", "ticker"),
    ("name", "Название", "txt"),
    ("quantity", "Кол-во", "num"),
    ("current_value", "∑ тек., ₽", "money"),
    ("target_value", "∑ план., ₽", "money"),
    ("deviation_value", "Отклонение, ₽", "money"),
    ("current_weight_pct", "Доля тек., %", "pct"),
    ("target_weight_pct", "Доля план., %", "pct"),
    ("cost_basis", "С/сть, ₽", "money"),
    ("accounting_profit", "Прибыль уч., ₽", "money"),
    ("accounting_profit_pct", "Прибыль уч., %", "pct"),
    ("years_held", "Лет", "num"),
    ("dividend_sum", "Див., ₽", "money"),
    ("dividend_pct", "Див., %", "pct"),
    ("action", "Действие", "action"),
    ("reason", "Причина", "txt"),
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
    universe_path = raw_dir / "stocks_universe.csv"
    if not universe_path.exists():
        logger.warning(f"stocks_universe.csv не найден в {raw_dir}")
        return {}
    try:
        u = read_csv_flex(universe_path)
        if "SECID" in u.columns and "ISIN" in u.columns:
            m = dict(zip(u["SECID"].astype(str).str.upper(), u["ISIN"].astype(str)))
            logger.info(f"ISIN-мапа загружена: {len(m)} тикеров")
            return m
    except Exception as e:
        logger.warning(f"Не удалось загрузить ISIN-мапу: {e}")
    return {}

def load_name_map(raw_dir: Path, logger: logging.Logger) -> dict:
    """Собирает маппинг SECID/ISIN -> name из stocks_universe, bonds_OFZ, bonds_CORPS."""
    name_map = {}
    # Акции: SECID -> name
    u_path = raw_dir / "stocks_universe.csv"
    if u_path.exists():
        try:
            u = read_csv_flex(u_path)
            if "SECID" in u.columns and "name" in u.columns:
                for secid, name in zip(u["SECID"], u["name"]):
                    name_map[str(secid).strip().upper()] = str(name)
        except Exception as e:
            logger.warning(f"Не удалось загрузить имена акций: {e}")
    # Облигации: ISIN -> name
    for fn in ("bonds_OFZ.csv", "bonds_CORPS.csv"):
        p = raw_dir / fn
        if p.exists():
            try:
                d = read_csv_flex(p)
                if "ISIN" in d.columns and "name" in d.columns:
                    for isin, name in zip(d["ISIN"], d["name"]):
                        name_map[str(isin).strip().upper()] = str(name)
            except Exception as e:
                logger.warning(f"Не удалось загрузить имена из {fn}: {e}")
    logger.info(f"Маппинг имён загружен: {len(name_map)} бумаг")
    return name_map

# ============================================================================
# Загрузка матрицы и расчёт сводки по субпортфелям
# ============================================================================
def load_portfolio_units(raw_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    path = raw_dir / "portfolio_units.csv"
    if not path.exists():
        logger.warning(f"Матрица {path.name} не найдена — сводка по субпортфелям недоступна")
        return pd.DataFrame()
    df = read_csv_flex(path)
    if df.empty or "SECID" not in df.columns:
        return pd.DataFrame()
    df["SECID"] = df["SECID"].astype(str).str.strip().str.upper()
    meta_cols = {"SECID", "asset_type"}
    for c in df.columns:
        if c not in meta_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Свертка по SECID и asset_type с суммированием всех остальных полей
    df = ( df.groupby(["SECID", "asset_type"], as_index=False) .sum(numeric_only=True) )
    df.to_csv(path, sep=";", index=False, encoding="utf-8")

    logger.info(f"Матрица субпортфелей загружена: {len(df)} бумаг, {len(df.columns) - len(meta_cols)} субпортфелей")
    return df


def compute_subportfolio_summary(matrix: pd.DataFrame, portfolio_details: pd.DataFrame,
                                  total_value: float, logger: logging.Logger) -> pd.DataFrame:
    """Считает сводку по субпортфелям: кол-во бумаг, units, доля, стоимость."""
    if matrix.empty:
        return pd.DataFrame()

    meta_cols = {"SECID", "asset_type"}
    sub_cols = [c for c in matrix.columns if c not in meta_cols]
    if not sub_cols:
        return pd.DataFrame()

    total_units_all = matrix[sub_cols].sum().sum()
    if total_units_all == 0:
        logger.warning("Сумма units в матрице = 0")
        return pd.DataFrame()

    # Подготовка portfolio_details
    details = portfolio_details.copy()
    if not details.empty and "SECID" in details.columns:
        details["SECID"] = details["SECID"].astype(str).str.strip().str.upper()
    else:
        details = pd.DataFrame(columns=["SECID", "current_value"])

    rows = []
    for col in sub_cols:
        sub_matrix = matrix[["SECID", col]].copy()
        sub_matrix.columns = ["SECID", "units"]
        sub_matrix = sub_matrix[sub_matrix["units"] > 0]

        if sub_matrix.empty:
            continue

        merged = sub_matrix.merge(details[["SECID", "current_value"]], on="SECID", how="left")
        merged["current_value"] = merged["current_value"].fillna(0)

        total_units_in_sub = merged["units"].sum()
        # Пропорциональное распределение стоимости
        merged["weighted_value"] = merged["current_value"] * merged["units"] / total_units_in_sub

        cost = merged["weighted_value"].sum()
        n_papers = len(merged)
        sum_units = int(total_units_in_sub)
        share_pct = sum_units / total_units_all * 100
        portfolio_share_pct = cost / total_value * 100 if total_value > 0 else 0

        sub_type = "auto" if col in AUTO_SUBPORTFOLIOS else "manual"

        rows.append({
            "subportfolio": col,
            "type": sub_type,
            "n_papers": n_papers,
            "sum_units": sum_units,
            "share_pct": round(share_pct, 2),
            "cost_rub": round(cost, 2),
            "portfolio_share_pct": round(portfolio_share_pct, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("share_pct", ascending=False).reset_index(drop=True)
    logger.info(f"Сводка по субпортфелям: {len(df)} субпортфелей")
    return df


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
    if kind == "action":
        s = str(value).strip().upper()
        colors = {"SELL": "#cf222e", "BUY": "#1a7f37", "FREEZE": "#8a5a00", "HOLD": "#57606a", "NO_ACTION": "#57606a"}
        color = colors.get(s, "#24292f")
        return f'<span style="color:{color};font-weight:600">{html.escape(s)}</span>', html.escape(s)
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
                 isin_map: dict | None = None) -> tuple[str, str]:
    """Возвращает (HTML панели без h2, label для вкладки со счётчиком)."""
    if df.empty:
        return (f'<section id="{table_id}" class="tabpane">'
                f'<p class="empty">Нет данных</p></section>',
                html.escape(title))

    cols = [(c, h, k) for c, h, k in columns if c in df.columns]
    missing = [c for c, _, _ in columns if c not in df.columns]
    if missing:
        logger.info(f"{title}: отсутствуют колонки {missing}")

    # Счётчики
    total = len(df)
    eligible_n = total
    if eligible_col and eligible_col in df.columns:
        eligible_n = int(df[eligible_col].astype(str).str.strip().str.lower()
                         .isin(["true", "1"]).sum())
    tab_label = f"{html.escape(title)} {eligible_n}/{total}"

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
                secid = val
                isin = None
                if isin_map:
                    if c == "SECID":
                        isin = isin_map.get(str(secid).upper(), "")
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

    panel_html = f"""
<section id="{table_id}" class="tabpane">
  <input class="filter" type="text" placeholder="Фильтр…" oninput="filterRows(this,'{table_id}')">
  <div class="twrap">
    <table id="tbl_{table_id}">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</section>"""
    return panel_html, tab_label


def render_subportfolio_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""

    head = "".join([
        "<th>Субпортфель</th>",
        "<th>Тип</th>",
        "<th>Бумаг</th>",
        "<th>Units</th>",
        "<th>Доля, %</th>",
        "<th>Стоимость, ₽</th>",
        "<th>Доля портфеля, %</th>",
    ])

    rows_html = []
    for _, row in summary.iterrows():
        type_badge = '<span class="badge auto">auto</span>' if row["type"] == "auto" else '<span class="badge manual">manual</span>'
        cells = [
            f'<td><b>{html.escape(str(row["subportfolio"]))}</b></td>',
            f'<td>{type_badge}</td>',
            f'<td class="num">{row["n_papers"]}</td>',
            f'<td class="num">{row["sum_units"]}</td>',
            f'<td class="num">{row["share_pct"]:.2f}</td>',
            f'<td class="num">{fmt_money(row["cost_rub"])}</td>',
            f'<td class="num">{row["portfolio_share_pct"]:.2f}</td>',
        ]
        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    return f"""
<div class="summary-block">
  <h3>Сводка по субпортфелям</h3>
  <div class="twrap">
    <table class="summary-table">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</div>
"""

def render_summary_tab(summary: pd.DataFrame, total_value: float, total_cost: float,
                       total_profit: float, total_profit_pct: float, total_div: float) -> tuple[str, str]:
    """Отдельная вкладка 'Сводно': общие итоги + сводка по субпортфелям."""
    profit_color = "#1a7f37" if total_profit >= 0 else "#cf222e"
    pf_summary_html = f"""
    <div class="pf-summary">
    <div><b>Стоимость:</b> {fmt_money(total_value)}</div>
    <div><b>Себестоимость:</b> {fmt_money(total_cost)}</div>
    <div><b>Прибыль:</b> <span style="color:{profit_color}">{fmt_money(total_profit)} ({total_profit_pct:+.2f}%)</span></div>
    <div><b>Ожидаемые дивиденды:</b> {fmt_money(total_div)}</div>
    </div>
    """
    sub_summary_html = render_subportfolio_summary(summary)
    panel_html = f"""
    <section id="pane_summary" class="tabpane">
    {pf_summary_html}
    {sub_summary_html}
    </section>"""
    return panel_html, "Сводно"

def render_portfolio(df: pd.DataFrame, isin_map: dict, name_map: dict,
                     logger: logging.Logger) -> tuple[str, str]:
    """Возвращает (HTML панели без h2, label для вкладки со счётчиком)."""
    if df.empty:
        return ('<section id="pane_portfolio" class="tabpane">'
                '<p class="empty">Нет данных</p></section>',
                'Портфель')

    # Добавляем колонку name из маппинга
    df = df.copy()
    df["name"] = df["SECID"].astype(str).str.strip().str.upper().map(name_map).fillna("")

    cols = [(c, h, k) for c, h, k in PORTFOLIO_COLUMNS if c in df.columns]
    head = "".join(
        f'<th data-kind="{k}" onclick="sortTable(this)">{html.escape(h)}'
        f'<span class="arr"></span></th>'
        for _, h, k in cols
    )
    rows_html = []
    for _, row in df.iterrows():
        secid = row["SECID"]
        asset_type = str(row.get("asset_type", ""))
        isin = isin_map.get(str(secid).upper(), "") if asset_type == "stock" else secid

        action = str(row.get("action", "")).strip().upper()
        cls = []
        if action == "SELL":
            cls.append("row-sell")
        elif action == "FREEZE":
            cls.append("row-freeze")

        target_w = row.get("target_weight_pct", 0)
        try:
            target_w_f = float(target_w)
        except (TypeError, ValueError):
            target_w_f = 0.0
        if target_w_f <= 0:
            cls.append("row-out-of-plan")

        tds = []
        for c, _, k in cols:
            val = row.get(c)
            style = ""
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
                num = k in ("num", "pct", "int", "money", "action")
                td_cls = "num" if num else ""
                tds.append(f'<td class="{td_cls}" style="{style}" data-v="{raw}">{disp}</td>')
        rows_html.append(f'<tr class="{" ".join(cls)}">{"".join(tds)}</tr>')

    # Переключатель "показать позиции вне плана"
    toggle_html = """
    <div class="toggle-block">
    <label>
    <input type="checkbox" id="toggleOutOfPlan" onchange="toggleOutOfPlanRows(this, 'pane_portfolio')">
    Показать позиции вне плана (target_weight = 0)
    </label>
    </div>
    """
    tab_label = f"Портфель {len(df)}"
    panel_html = f"""
    <section id="pane_portfolio" class="tabpane">
    {toggle_html}
    <div class="twrap">
    <table id="tbl_pane_portfolio">
    <thead><tr>{head}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
    </table>
    </div>
    </section>"""
    return panel_html, tab_label

# ============================================================================
# HTML-каркас
# ============================================================================
CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;
     background:#f6f8fa;color:#24292f}
header{background:#24292f;color:#fff;padding:12px 24px}
header .sub{opacity:.65;font-size:11px;font-weight:300;letter-spacing:.3px}
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
.row-sell{background:#ffebe9}
.row-freeze{background:#fff8c5}
.row-out-of-plan{opacity:.45;font-style:italic}
.empty{color:#57606a}
.tk{display:inline-flex;align-items:center;gap:6px}
.logo{width:20px;height:20px;flex:none;border-radius:50%;object-fit:contain;
      background:#fff;border:1px solid #d0d7de;vertical-align:middle}
.pf-summary{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;
            padding:12px 16px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px}
.pf-summary div{font-size:14px}
.summary-block{margin-bottom:14px}
.summary-block h3{margin:0 0 8px;font-size:14px;color:#57606a}
.summary-table{font-size:13px}
.summary-table th{background:#eaeef2}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge.auto{background:#ddf4ff;color:#0969da}
.badge.manual{background:#fff8c5;color:#8a5a00}
.toggle-block{margin-bottom:10px;padding:8px 12px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px}
.toggle-block label{cursor:pointer;font-size:13px}
.toggle-block input{margin-right:6px}
@media print{.filter,nav,.toggle-block{display:none}.tabpane{display:block!important;page-break-before:always}}
nav button{white-space:nowrap}
nav button{font-size:13px}
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
      const na=parseFloat(String(va).replace(/\\s/g,'').replace('₽','').replace('%','')),
            nb=parseFloat(String(vb).replace(/\\s/g,'').replace('₽','').replace('%',''));
      r=(isNaN(na)?-Infinity:na)-(isNaN(nb)?-Infinity:nb);
    } else r=va.localeCompare(vb,'ru');
    return asc?r:-r;
  });
  rows.forEach(r=>tb.appendChild(r));
}
function toggleOutOfPlanRows(checkbox, pid){
  const rows = document.querySelectorAll('#tbl_'+pid+' tbody tr');
  rows.forEach(row => {
    if (checkbox.checked) {
      row.style.display = '';
    } else {
      if (row.classList.contains('row-out-of-plan')) {
        row.style.display = 'none';
      } else {
        row.style.display = '';
      }
    }
  });
}
document.addEventListener('DOMContentLoaded',()=>{
  const b=document.querySelector('nav button'); if(b) b.click();
  // Скрываем позиции вне плана по умолчанию
  const cb = document.getElementById('toggleOutOfPlan');
  if (cb) toggleOutOfPlanRows(cb, 'pane_portfolio');
  // Сортировка портфеля по deviation_value (колонка 5, 0-based)
  const tbl = document.getElementById('tbl_pane_portfolio');
  if (tbl) {
    const ths = tbl.tHead.rows[0].cells;
    if (ths[5]) sortTable(ths[5]);
  }
});
"""


def build_html(date: str, panes: str, tab_labels: list[str], tab_ids: list[str]) -> str:
    nav_buttons = "".join(
        f'<button onclick="showTab(\'{tid}\',this)">{label}</button>'
        for tid, label in zip(tab_ids, tab_labels)
    )
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт {date}</title><style>{CSS}</style></head>
<body>
<header><div class="sub">Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div></header>
<main>
<nav>
  {nav_buttons}
</nav>
{panes}
</main>
<script>{JS}</script>
</body></html>"""


# ============================================================================
# Основная функция
# ============================================================================
def build_report(date: str, input_dir: Path, output_dir: Path, raw_dir: Path,
                 logger: logging.Logger) -> Path:
    stocks = load_table(input_dir / "stock_ranking.csv", logger)
    corp = load_table(input_dir / "corp_ranking.csv", logger)
    portfolio = load_table(output_dir / "portfolio_details.csv", logger)
    # Загрузка двух рейтингов ОФЗ и объединение
    core_ofz = load_table(input_dir / "core_ofz_ranking.csv", logger)
    long_ofz = load_table(input_dir / "long_ofz_ranking.csv", logger)

    # Объединение в один DataFrame
    ofz_parts = []
    if not core_ofz.empty:
        ofz_parts.append(core_ofz)
    if not long_ofz.empty:
        ofz_parts.append(long_ofz)

    if ofz_parts:
        ofz = pd.concat(ofz_parts, ignore_index=True)
        # Сортировка по current_yield_pct DESC (как в Module 2)
        if "current_yield_pct" in ofz.columns:
            ofz["current_yield_pct"] = pd.to_numeric(ofz["current_yield_pct"], errors="coerce")
            ofz = ofz.sort_values("current_yield_pct", ascending=False).reset_index(drop=True)
        # Перенумерация rank
        ofz["rank"] = range(1, len(ofz) + 1)
        logger.info(f"ОФЗ объединены: core_ofz={len(core_ofz)}, long_ofz={len(long_ofz)}, итого={len(ofz)}")
    else:
        ofz = pd.DataFrame()
    
    
    # Фоллбэк для старого bond_ranking.csv
    if ofz.empty and corp.empty:
        legacy = input_dir / "bond_ranking.csv"
        if legacy.exists():
            df = load_table(legacy, logger)
            if not df.empty and "bond_type" in df.columns:
                ofz = df[df["bond_type"] == "ofz"].reset_index(drop=True)
                corp = df[df["bond_type"] == "corporate"].reset_index(drop=True)

    # ISIN-мапа и Name-мапа
    isin_map = load_isin_map(raw_dir, logger)
    name_map = load_name_map(raw_dir, logger)

    # Матрица субпортфелей
    matrix = load_portfolio_units(raw_dir, logger)

    # Итоги портфеля (для вкладки "Сводно")
    total_value = 0.0
    total_cost = 0.0
    total_div = 0.0
    if not portfolio.empty:
        total_value = pd.to_numeric(portfolio.get("current_value", pd.Series()), errors="coerce").sum()
        total_cost = pd.to_numeric(portfolio.get("cost_basis", pd.Series()), errors="coerce").sum()
        total_div = pd.to_numeric(portfolio.get("dividend_sum", pd.Series()), errors="coerce").sum()
    total_profit = total_value - total_cost
    total_profit_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0

    summary = compute_subportfolio_summary(matrix, portfolio, total_value, logger) if not matrix.empty else pd.DataFrame()

    # Рендер вкладок
    portfolio_panel, portfolio_label = render_portfolio(portfolio, isin_map, name_map, logger)
    summary_panel, summary_label = render_summary_tab(
        summary, total_value, total_cost, total_profit, total_profit_pct, total_div)
    stocks_panel, stocks_label = render_table(
        stocks, STOCK_COLUMNS, "Акции", "pane_stocks", logger,
        eligible_col="eligible_for_buy", isin_map=isin_map)
    ofz_panel, ofz_label = render_table(
        ofz, BOND_COLUMNS, "ОФЗ", "pane_ofz", logger,
        eligible_col="eligible", isin_map=isin_map)
    corp_panel, corp_label = render_table(
        corp, BOND_COLUMNS, "Корпораты", "pane_corp", logger,
        eligible_col="eligible", isin_map=isin_map)

    # Порядок: Портфель, Сводно, Акции, ОФЗ, Корпораты
    panes = portfolio_panel + summary_panel + stocks_panel + ofz_panel + corp_panel
    tab_labels = [portfolio_label, summary_label, stocks_label, ofz_label, corp_label]
    tab_ids = ["pane_portfolio", "pane_summary", "pane_stocks", "pane_ofz", "pane_corp"]



    out = output_dir / f"report_{date}.html"
    out.write_text(build_html(date, panes, tab_labels, tab_ids), encoding="utf-8")
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
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--open", action="store_true", help="Открыть отчёт в браузере")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p
    config_path = resolve(args.config)
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    raw_dir = resolve(args.raw_dir)

    cfg = load_config(config_path)
    d = cfg.get("date", "today")
    date = args.date or (
        datetime.now().strftime("%Y-%m-%d")
        if not d or str(d).lower() in ("today", "auto", "") else str(d)
    )

    logger = setup_logging(date, output_dir)
    logger.info(f"=== Module 4 start: date={date} ===")
    try:
        out = build_report(date, input_dir, output_dir, raw_dir, logger)
        # if args.open:
            # if sys.platform == "win32":
                # import os
                # try:
                    # os.startfile(str(out))
                # except Exception:
                    # webbrowser.open(out.as_uri())
            # else:
                # webbrowser.open(out.as_uri())
        logger.info("=== Module 4 finished ===")
        return 0
    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())