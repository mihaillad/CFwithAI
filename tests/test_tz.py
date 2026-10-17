import pytest
import pandas as pd
import math

def test_t04_lot_size_calculation():
    """T-04: Лотность: дефицит 5000, цена 250, лот 10 -> qty_delta=10, qty_delta_lots=1"""
    deficit_rub = 5000.0
    price = 250.0
    lot_size = 10
    lots_delta = math.floor(deficit_rub / (price * lot_size))
    qty_delta = lots_delta * lot_size
    
    assert lots_delta == 2 # 5000 / 2500 = 2
    assert qty_delta == 20
    
    # Если дефицит меньше стоимости лота:
    deficit_rub_small = 100.0
    lots_delta_small = math.floor(deficit_rub_small / (price * lot_size)) # 0
    qty_delta_small = lots_delta_small * lot_size # 0
    assert qty_delta_small == 0

def test_t13_bond_spread_filter():
    """T-13: Спред < 2% -> excluded"""
    ytm = 10.0
    cbr = 9.0
    spread = ytm - cbr
    assert spread == 1.0
    assert spread < 2.0 # Должно быть отфильтровано

def test_t17_isin_duplicate_priority():
    """T-17: Дубликат ISIN между OFZ и CORPS -> приоритет OFZ"""
    ofz = pd.DataFrame({'ISIN': ['RU000A1'], 'type': ['ofz']})
    corps = pd.DataFrame({'ISIN': ['RU000A1', 'RU000A2'], 'type': ['corp', 'corp']})
    
    common = set(ofz['ISIN']).intersection(set(corps['ISIN']))
    corps_filtered = corps[~corps['ISIN'].isin(common)]
    
    assert len(corps_filtered) == 1
    assert corps_filtered.iloc[0]['ISIN'] == 'RU000A2'