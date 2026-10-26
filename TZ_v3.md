# Техническое задание. Инвестиционная аналитика v3.0
Дата: 2026-10-25

## 0. Общие положения

### 0.1 Назначение
Автоматизация четырёх контуров:
1. Рейтинг дивидендных акций Мосбиржи с риск-фильтрами.
2. Рейтинг облигаций (ОФЗ core, ОФЗ long, корпораты) с динамическими фильтрами.
3. Оценка текущего портфеля и рекомендации по ребалансировке.
4. HTML-отчёт с интерактивными таблицами.

### 0.2 Архитектура
project/
├── config.yaml
├── DECISIONS.md
├── main.py
├── src/
│ ├── common.py
│ ├── module1_stocks.py
│ ├── module2_bonds.py
│ ├── module3_portfolio.py
│ └── module4_report.py
├── data/
│ ├── raw/
│ │ ├── divs_data.csv
│ │ ├── stocks_universe.csv
│ │ ├── stock_prices.csv
│ │ ├── stock_fundamentals.csv
│ │ ├── special_dividends_registry.csv
│ │ ├── bonds_OFZ.csv
│ │ ├── bonds_CORPS.csv
│ │ ├── bond_prices.csv
│ │ ├── risk_free_rate.csv
│ │ ├── deals.csv
│ │ ├── portfolio_holdings.csv
│ │ ├── cash_balance.csv
│ │ └── portfolio_units.csv
│ └── output/
│ ├── stock_ranking.csv
│ ├── stock_excluded.csv
│ ├── core_ofz_ranking.csv
│ ├── long_ofz_ranking.csv
│ ├── corp_ranking.csv
│ ├── bond_allocation.csv
│ ├── bond_filter_stats.csv
│ ├── ofz_all_with_reasons.csv
│ ├── corp_all_with_reasons.csv
│ ├── portfolio_snapshot.csv
│ ├── portfolio_details.csv
│ ├── rebalance_recommendations.csv
│ ├── outsiders.csv
│ ├── diagnostics.csv
│ ├── index.html
│ └── report_YYYY-MM-DD.html
└── tests/


### 0.3 Стек
Python 3.11+, pandas, numpy, PyYAML, requests, pytest. БД нет.

### 0.4 Соглашения
- Даты: `YYYY-MM-DD`
- Деньги: рубли, `float`, 2–4 знака; дивиденды — до 9 знаков
- Ставки/доли: в процентах, суффикс `_pct`
- CSV: UTF-8, разделитель `;`, десятичная точка
- Идемпотентность: повторный запуск за ту же дату перезаписывает output и лог
- Логи: `output/run_YYYY-MM-DD.log`
- Ошибки строк не роняют прогон — строка помечается `exclusion_reason`
- Поле лотности: `LOTSIZE` (везде)

### 0.5 Оркестратор `main.py`
1. Загрузить `config.yaml`, валидировать ключи.
2. `module1.rank_stocks(date)` → `stock_ranking.csv`, `stock_excluded.csv`.
3. `module2.rank_bonds(date)` → `core_ofz_ranking.csv`, `long_ofz_ranking.csv`, `corp_ranking.csv`, `bond_allocation.csv`, `bond_filter_stats.csv`.
4. `module3.evaluate_portfolio(date, ...)` → `portfolio_snapshot.csv`, `portfolio_details.csv`, `rebalance_recommendations.csv`, `outsiders.csv`.
5. `module4.render_html(date, ...)` → `index.html`, `report_YYYY-MM-DD.html`.
6. Сформировать `diagnostics.csv`.

---

## 1. Входные таблицы (data/raw)

### 1.1 `config.yaml`
Все ключи с комментариями на русском. Основные группы:
- Дата расчёта (`date: "today"`)
- Акции: `top_n_stocks`, `weight_top1_pct`, `weight_step_pct`, `stock_sleeve_pct`, `d_min_days`, `min_dy_gross_pct`, `payout_warn_pct`, `payout_max_pct`, `registry_missing_policy`, `registry_strict_mode`
- Облигации общие: `bond_sleeve_pct`, `bond_min_volume_rub`, `bond_avg_volume_days`, `bond_require_coupon`, `bond_max_facevalue`, `risk_free_rate_max_age_sec`, `tax_rate_pct`
- Доли внутри bond_sleeve: `core_ofz_share_pct`, `long_ofz_share_pct`, `corp_share_pct`
- ОФЗ core: `core_ofz_min_years`, `core_ofz_max_years`, `core_ofz_price_min_pct`, `core_ofz_price_max_pct`
- ОФЗ long: `long_ofz_min_years`, `long_ofz_max_years`, `long_ofz_price_min_pct`, `long_ofz_price_max_pct`, `long_ofz_top_n`
- Корпораты: `corp_ytm_min_spread_to_cbr`, `corp_ytm_max_multiplier`, `corp_price_min_pct`, `corp_price_max_pct`, `corp_min_months`, `corp_max_months`, `min_credit_rating`, `issuer_limit_pct`
- Портфель: `top_n_freeze`, `profit_take_pct`, `max_hold_years`, `frozen_limit_pct`, `cooldown_days`, `flow_limit_pct_week`, `min_deficit_pct`, `cash_buffer_pct`
- Технические: `lot_source`, `nkd_field`

### 1.2 `stocks_universe.csv`
Автозагрузка с MOEX ISS. Поля: `SECID`, `issuer`, `name`, `sector`, `LOTSIZE`, `board`, `lot_as_of`.

### 1.3 `stock_prices.csv`
Автозагрузка с MOEX ISS. Поля: `date`, `SECID`, `close`, `volume_rub`.

### 1.4 `divs_data.csv`
Автозагрузка с БКС + локальная история. Поля: `SECID`, `close_date`, `div_sum` (9 знаков).

### 1.5 `stock_fundamentals.csv`
Поля: `SECID`, `report_date`, `period`, `eps_ttm`, `eps_prev_ttm`, `eps_3y_ago`, `net_income_ttm`, `shares_outstanding`, `dividend_policy_url`, `notes`.

### 1.6 `special_dividends_registry.csv`
Поля: `SECID`, `declaration_date`, `amount_per_share`, `classification`, `reason`, `evidence`, `classifier`, `review_date`.

### 1.7 `bonds_OFZ.csv`
Автозагрузка с MOEX ISS. Поля: `ISIN`, `name`, `issuer` (RF), `maturity_date`, `offer_date`, `coupon_type`, `coupon_rate_pct`, `coupon_period_days`, `nominal`, `amortization`, `LOTSIZE`, `board`, `lot_as_of`.

### 1.8 `bonds_CORPS.csv`
Автозагрузка с MOEX ISS. Поля: `ISIN`, `name`, `issuer` (нормализованный из LATNAME), `issuer_group`, `maturity_date`, `offer_date`, `coupon_type`, `coupon_rate_pct`, `coupon_period_days`, `nominal`, `amortization`, `credit_rating`, `rating_agency`, `LOTSIZE`, `board`, `lot_as_of`.

### 1.9 `bond_prices.csv`
Автозагрузка с MOEX ISS. Поля: `date`, `ISIN`, `price_pct`, `nkd_rub`, `ytm_pct`, `duration_years`, `volume_rub`.

### 1.10 `risk_free_rate.csv`
Одна строка: `date`, `rate_pct`. Автозагрузка с ЦБ, кэш 24 часа.

### 1.11 `portfolio_holdings.csv`
Поля: `asset_type`, `SECID_or_ISIN`, `quantity`, `avg_price`, `first_purchase_date`, `last_purchase_date`.

### 1.12 `deals.csv`
Автозагрузка. Поля: `SECID`, `COUNT`, `SUM`, `TYPE`, `DATE`.

### 1.13 `cash_balance.csv`
Поля: `date`, `currency`, `amount`.

### 1.14 `portfolio_units.csv`
Матрица субпортфелей. Длинный формат: `SECID`, `asset_type`, `bucket`, `weight_pct`.

---

## 2. Модуль 1. Рейтинг акций

### 2.1 Алгоритм
1. Автозагрузка справочника акций, цен, календаря БКС, ставки ЦБ.
2. Обновление `divs_data.csv` из БКС (upsert, удаление устаревших прогнозов).
3. Генерация прогнозов: выплата за последние 365 дней → дата +365, сумма та же. Прогноз гасится объявленным дивидендом в окне ±45 дней.
4. Обход ТОЛЬКО SECID с будущими дивидендами.
5. Для каждого SECID:
   - Проверка цены, реестра спецдивидендов.
   - Выбор ближайшего дивиденда (приоритет declared > forecast).
   - Расчёт `ay_pct`, `dy_gross_pct`, фундаментальных показателей.
   - Проверка стоп-листа.
6. Фильтрация пар обычка/преф (остаётся лучшая по AY/CY).
7. Сортировка: `ay_pct` desc, `dps_source` (declared > forecast).
8. Нормировка весов для top-N.

### 2.2 Выход: `stock_ranking.csv`
Поля: `date`, `rank`, `SECID`, `name`, `issuer`, `sector`, `price`, `dps_reg_next`, `dps_source`, `record_date`, `days_to_record`, `ay_pct`, `dy_gross_pct`, `payout_fwd_pct`, `eps_growth_yoy_pct`, `eps_growth_3y_pct`, `stop_flag`, `stop_reason`, `target_weight_sleeve_pct`, `target_weight_portfolio_pct`, `LOTSIZE`, `eligible_for_buy`, `notes`.

### 2.3 Выход: `stock_excluded.csv`
Поля: `date`, `SECID`, `reason`, `details`.

---

## 3. Модуль 2. Рейтинг облигаций

### 3.1 Алгоритм
1. Автозагрузка справочников (ОФЗ/корп), цен, ставки ЦБ.
2. Первичная фильтрация по быстрым критериям (срок, цена, YTM, рейтинг, номинал, купон).
3. Загрузка истории оборотов для оставшихся корпоратов (20 дней).
4. Финальная фильтрация по среднему обороту (только корп, ОФЗ без этого фильтра).
5. Удаление дублей эмитентов (группировка по нормализованному issuer, лучший по YTM).
6. Распределение по бакетам: `core_ofz`, `long_ofz`, `corp`.
7. Сортировка внутри бакетов по `current_yield_pct` desc.
8. Расчёт целевых весов с учётом `issuer_limit_pct`.
9. Формирование диагностики.

### 3.2 Фильтры

#### ОФЗ (общие)
- Рейтинг: AAA
- YTM ≥ risk_free_rate
- FACEVALUE < 10000
- COUPONPERIOD > 0
- Без фильтра по обороту

#### core_ofz
- Срок: 2.5–3.0 года
- Цена: 85–105%

#### long_ofz
- Срок: 8–12 лет
- Цена: 50–105%
- Top-2 по YTM

#### corp
- Срок: 12–40 месяцев
- Цена: 60–105%
- YTM: [cbr + 5%] … [cbr * 3]
- Рейтинг: ≥ A-
- Спред к ЦБ: ≥ 2%
- Средний оборот: ≥ `bond_min_volume_rub`
- Удаление дублей эмитентов

### 3.3 Выходы
- `core_ofz_ranking.csv`
- `long_ofz_ranking.csv`
- `corp_ranking.csv`
- `bond_allocation.csv` (сводка по бакетам)
- `bond_filter_stats.csv` (статистика по фильтрам)
- `ofz_all_with_reasons.csv` (все ОФЗ с причинами исключения)
- `corp_all_with_reasons.csv` (все корп с причинами исключения)

### 3.4 Схема рейтинга облигации
Поля: `date`, `rank`, `ISIN`, `name`, `issuer`, `bond_type`, `maturity_date`, `days_to_maturity`, `coupon_rate_pct`, `price_pct`, `nkd_rub`, `current_yield_pct`, `ytm_gross_pct`, `ytm_net_pct`, `spread_to_cbr_pct`, `duration_years`, `credit_rating`, `bucket`, `eligible`, `exclusion_reason`, `avg_daily_trades`, `LOTSIZE`, `target_weight_sleeve_pct`, `target_weight_portfolio_pct`.

---

## 4. Модуль 3. Оценка портфеля

### 4.1 Алгоритм
1. Загрузка матрицы субпортфелей.
2. Загрузка рейтингов акций и облигаций (core_ofz, long_ofz, corp).
3. Расчёт целевых весов (для облигаций — приоритет у рейтинга).
4. Загрузка holdings, цен, метаданных, стоп-листа, кэша.
5. Оценка позиций (stock_value, bond_value с НКД, cash_value).
6. Расчёт дефицитов.
7. Формирование рекомендаций (SELL/BUY/HOLD/NO_ACTION) с учётом ограничений.
8. Формирование outsiders.

### 4.2 Выходы
- `portfolio_snapshot.csv` (с разбивкой по бакетам облигаций)
- `portfolio_details.csv` (с полем `bucket`)
- `rebalance_recommendations.csv` (с полем `bucket`)
- `outsiders.csv`

---

## 5. Модуль 4. HTML-отчёт

### 5.1 Функциональность
- Три вкладки: Акции, ОФЗ, Корпораты.
- Сортировка колонок кликом.
- Цветные чипы для статусов.
- Карточки-сводка сверху.
- Тёмная тема по системным настройкам.
- Архивный снимок `report_YYYY-MM-DD.html`.

---

## 6. Тесты (pytest)

### Базовые
- T-01. `bond_spread_basis=gross`: фильтр по `ytm_pct - cbr`.
- T-02. `bond_value = qty * (price_pct/100 * nominal + nkd)`.
- T-03. `nkd_rub < 0` — принимается, `nkd_negative`.
- T-04. Лотность: дефицит 5000, цена 250, лот 10 → `qty_delta=10`, `qty_delta_lots=1`.
- T-05. Лотность берётся из universe-файлов.
- T-06. `top_n_freeze=8`: rank 8 → FREEZE, rank 9 → SELL.
- T-07. Fallback по политике дивидендов.
- T-08. Отсутствие реестра → `registry_missing`.
- T-09. `registry_strict_mode=true` → fallback отключён.
- T-10. Веса 16 бумаг нормируются в 100%.
- T-11. AY при `d < d_min_days` → excluded.
- T-12. Спецдивиденд не попадает в `dps_reg_next`.
- T-13. Спред < 2% → excluded.
- T-14. 3-летний выход не срабатывает для топ-8.
- T-15. `frozen_limit` breach → SELL.
- T-16. Бумага без свежих цен → `no_price`.
- T-17. Дубликат ISIN между OFZ и CORPS → приоритет OFZ.
- T-18. Лог за дату перезаписывается.
- T-19. `stocks_universe.csv` без `is_active` обрабатывается корректно.
- T-20. Несколько board для одного SECID → выбран с максимальным оборотом.

### Новые (v3.0)
- T-21. YTM корп в динамическом диапазоне: при cbr=14% → [19%, 42%].
- T-22. Дубли эмитентов: из 3 выпусков "Самолет" остаётся 1.
- T-23. SPOB исключается из справочника и цен.
- T-24. `current_yield_pct = coupon_rate / (price/100)`.
- T-25. Прогноз дивиденда гасится объявленным в окне ±45 дней.
- T-26. long_ofz ограничен top-2.
- T-27. ОФЗ без фильтра по обороту.
- T-28. Бескупонные (COUPONPERIOD=0) исключаются.
- T-29. FACEVALUE ≥ 10000 исключается.

---

## 7. Зафиксированные решения

См. `DECISIONS.md`. Все архитектурные, фильтрационные и технические решения задокументированы там.

---

## 8. Нефункциональные требования

- Производительность: полный прогон < 5 сек (без сетевых запросов при актуальном кэше).
- Валидация: `common.validate_schema(df, schema)` перед расчётом.
- Логирование: INFO — счётчики; WARNING — breaches; ERROR — исключения с traceback.
- Воспроизводимость: `date` из config, иначе today.
- Аудит: в каждом выходном CSV — `date`; в `diagnostics.csv` — список входных файлов с mtime и sha256.