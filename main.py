#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оркестратор проекта.
Последовательно запускает модули 1-4 и открывает итоговый HTML-отчёт.

Запуск:
    python main.py
    python main.py --date 2026-10-18
    python main.py --open-report      # открыть HTML в браузере (по умолчанию включено)
    python main.py --no-open-report   # не открывать HTML
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

# Пути относительно корня проекта
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Модули в порядке выполнения
MODULES = [
    ("Module 1: Stocks",    "src/module1_stocks.py"),
    ("Module 2: Bonds",     "src/module2_bonds.py"),
    ("Module 3: Portfolio", "src/module3_portfolio.py"),
    ("Module 4: Report",    "src/module4_report.py"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Оркестратор инвестиционной аналитики")
    ap.add_argument("--date", default="",
                    help="Дата расчёта YYYY-MM-DD (по умолчанию — из config.yaml или today)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="Путь к config.yaml")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="Папка с входными CSV")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Папка с выходными CSV")
    ap.add_argument("--open-report", action="store_true", default=True,
                    help="Открыть HTML-отчёт в браузере (по умолчанию)")
    ap.add_argument("--no-open-report", action="store_true",
                    help="Не открывать HTML-отчёт")
    return ap.parse_args()


def run_module(name: str, script: str, common_args: list[str]) -> tuple[int, float]:
    """Запускает модуль как отдельный процесс. Возвращает (returncode, elapsed_sec)."""
    cmd = [sys.executable, script, *common_args]
    print(f"\n{'=' * 70}\n▶ {name}\n  cmd: {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0
        status = "✓ OK" if result.returncode == 0 else f"✗ FAILED (code {result.returncode})"
        print(f"\n{status}  [{elapsed:.1f}s]")
        return result.returncode, elapsed
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script}", file=sys.stderr)
        return 2, time.time() - t0
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1, time.time() - t0


def main() -> int:
    args = parse_args()

    # Собираем общие аргументы для всех модулей
    common_args = []
    if args.date:
        common_args += ["--date", args.date]
    common_args += ["--config", args.config]
    common_args += ["--data-dir", args.data_dir]
    common_args += ["--output-dir", args.output_dir]

    print(f"🚀 Оркестратор запущен: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"   date       = {args.date or '(из config / today)'}")
    print(f"   config     = {args.config}")
    print(f"   data_dir   = {args.data_dir}")
    print(f"   output_dir = {args.output_dir}")

    t_start = time.time()
    results: list[tuple[str, int, float]] = []

    # Последовательный запуск модулей 1-4
    for name, script in MODULES:
        # Для module4_report добавляем флаг --open, если нужно
        # Формируем аргументы для каждого модуля
        module_args = list(common_args)
        
        # Для module4_report: подменяем --data-dir на --input-dir и добавляем --raw-dir
        if script.endswith("module4_report.py"):
            new_args = []
            skip_next = False
            for i, arg in enumerate(module_args):
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--data-dir":
                    new_args.append("--input-dir")
                    new_args.append(args.output_dir)
                    skip_next = True
                else:
                    new_args.append(arg)
            # Добавляем --raw-dir
            new_args += ["--raw-dir", args.data_dir]
            module_args = new_args
            if args.open_report and not args.no_open_report:
                module_args.append("--open")
                
                                
        rc, elapsed = run_module(name, script, module_args)
        results.append((name, rc, elapsed))

        # Если критичный модуль (1-3) упал — прерываем цепочку
        if rc != 0 and not script.endswith("module4_report.py"):
            print(f"\n⛔ Критическая ошибка в {name}. Прерывание.")
            break

    # Итоговая сводка
    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print("📊 ИТОГ")
    print(f"{'=' * 70}")
    for name, rc, elapsed in results:
        mark = "✓" if rc == 0 else "✗"
        print(f"  {mark}  {name:<25} {elapsed:>6.1f}s  (code {rc})")
    print(f"{'─' * 70}")
    print(f"  Общее время: {total:.1f}s")

    # Открываем HTML-отчёт в браузере
    if args.open_report and not args.no_open_report:
        # Находим самый свежий report_*.html
        html_files = sorted(DEFAULT_OUTPUT_DIR.glob("report_*.html"), key=lambda p: p.stat().st_mtime)
        if html_files:
            latest = html_files[-1]
            print(f"\n🌐 Открываю отчёт: {latest}")
            
            # На Windows используем os.startfile() — надёжнее для путей с кириллицей
            if sys.platform == "win32":
                import os
                try:
                    os.startfile(str(latest))
                except Exception as e:
                    print(f"⚠ os.startfile() не сработал: {e}. Пробую webbrowser.open()...")
                    webbrowser.open(latest.as_uri())
            else:
                webbrowser.open(latest.as_uri())
        else:
            print(f"\n⚠ HTML-отчёт не найден в {DEFAULT_OUTPUT_DIR}")
            
            
    # Возвращаем код ошибки, если хоть один модуль упал
    failed = [r for r in results if r[1] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())