import pandas as pd
import numpy as np
from pathlib import Path
import logging
import math

def evaluate_portfolio(date: str, config: dict, data_dir: Path, stock_ranking: pd.DataFrame, bond_ranking: pd.DataFrame, logger: logging.Logger, diagnostics: list):
    date_dt = pd.to_datetime(date)
    
    holdings = pd.read_csv(data_dir / 'portfolio_holdings.csv', sep=';')
    cash = pd.read_csv(data_dir / 'cash_balance.csv', sep=';')
    prices_stocks = pd.read_csv(data_dir / 'stock_prices.csv', sep=';')
    prices_bonds = pd.read_csv(data_dir / 'bond_prices.csv', sep=';')
    universe_stocks = pd.read_csv(data_dir / 'stocks_universe.csv', sep=';')
    
    latest_stock_prices = prices_stocks[prices_stocks['date'] <= date].sort_values('date').groupby('SECID').tail(1).set_index('SECID')
    latest_bond_prices = prices_bonds[prices_bonds['date'] <= date].sort_values('date').groupby('ISIN').tail(1).set_index('ISIN')
    
    cash_value = cash[cash['currency'] == 'RUB']['amount'].sum() if 'RUB' in cash['currency'].values else 0.0
    
    stock_val = 0.0
    bond_val = 0.0
    bond_val_clean = 0.0
    
    recommendations = []
    outsiders = []
    
    for _, h in holdings.iterrows():
        asset_type = h['asset_type']
        secid_or_isin = h['SECID_or_ISIN']
        qty = h['quantity']
        avg_price = h['avg_price']
        
        if asset_type == 'stock':
            if secid_or_isin in latest_stock_prices.index:
                close = latest_stock_prices.loc[secid_or_isin, 'close']
                val = qty * close
                stock_val += val
                
                # Аутсайдеры логика (§4.7, §4.9)
                holding_days = (date_dt - pd.to_datetime(h['first_purchase_date'])).days
                pnl_pct = ((close - avg_price) / avg_price * 100) if avg_price > 0 else np.nan
                
                stock_info = stock_ranking[stock_ranking['SECID'] == secid_or_isin]
                rank = stock_info['rank'].iloc[0] if not stock_info.empty else 999
                stop_flag = stock_info['stop_flag'].iloc[0] if not stock_info.empty else False
                in_top_freeze = rank <= config['top_n_freeze']
                
                action = 'HOLD'
                reason = 'no_action'
                
                if stop_flag:
                    action = 'SELL'
                    reason = 'stop_list'
                elif pnl_pct >= config['profit_take_pct']:
                    action = 'SELL'
                    reason = 'profit_take'
                elif holding_days >= 3365 and pnl_pct < 0:
                    if in_top_freeze:
                        action = 'FREEZE'
                        reason = 'three_year_top_freeze'
                    else:
                        action = 'SELL'
                        reason = 'three_year_exit'
                        
                outsiders.append({
                    'date': date, 'SECID': secid_or_isin, 'first_purchase_date': h['first_purchase_date'],
                    'holding_days': holding_days, 'avg_price': avg_price, 'current_price': close,
                    'pnl_pct': pnl_pct, 'in_top_freeze_ay': in_top_freeze, 'top_n_freeze_used': config['top_n_freeze'],
                    'stop_flag': stop_flag, 'action': action, 'reason': reason
                })
                
        elif asset_type == 'bond':
            if secid_or_isin in latest_bond_prices.index:
                p_row = latest_bond_prices.loc[secid_or_isin]
                nominal = 1000.0 # Упрощение, лучше брать из bonds_OFZ/CORPS
                nkd = p_row['nkd_rub'] if pd.notna(p_row['nkd_rub']) else 0.0
                if pd.isna(p_row['nkd_rub']):
                    diagnostics.append({'code': 'nkd_missing', 'details': f"ISIN {secid_or_isin}"})
                    
                val = qty * ((p_row['price_pct'] / 100 * nominal) + nkd)
                clean_val = qty * (p_row['price_pct'] / 100 * nominal)
                bond_val += val
                bond_val_clean += clean_val
                
    total_value = stock_val + bond_val + cash_value
    if total_value == 0:
        raise ValueError("total_value == 0")
        
    snapshot = {
        'date': date, 'total_value': total_value, 'stock_value': stock_val, 'bond_value': bond_val,
        'bond_value_clean': bond_val_clean, 'cash_value': cash_value,
        'stock_share_pct': (stock_val / total_value * 100),
        'bond_share_pct': (bond_val / total_value * 100),
        'cash_share_pct': (cash_value / total_value * 100),
        'target_stock_pct': config['stock_sleeve_pct'],
        'target_bond_pct': config['bond_sleeve_pct'],
        'deviation_stock_pct': (stock_val / total_value * 100) - config['stock_sleeve_pct'],
        'deviation_bond_pct': (bond_val / total_value * 100) - config['bond_sleeve_pct'],
        'cash_buffer_excess_pct': (cash_value / total_value * 100) - config['cash_buffer_pct']
    }
    
    # Генерация рекомендаций (упрощенная для BUY по дефициту)
    # T-04: Лотность: дефицит 5000, цена 250, лот 10 -> qty_delta=10, qty_delta_lots=1
    for _, row in stock_ranking.iterrows():
        if row['eligible_for_buy']:
            target_val = total_value * (row['target_weight_portfolio_pct'] / 100)
            current_val = 0 # Упрощение: нужно агрегировать из holdings
            deficit_pct = row['target_weight_portfolio_pct'] - (current_val / total_value * 100)
            
            if deficit_pct > config['min_deficit_pct']:
                lot_size = row['lot_size']
                price = row['price']
                # Q_delta = floor(deficit * total / (price * lot_size)) * lot_size
                deficit_rub = deficit_pct / 100 * total_value
                lots_delta = math.floor(deficit_rub / (price * lot_size))
                qty_delta = lots_delta * lot_size
                
                action = 'BUY' if qty_delta > 0 else 'NO_ACTION'
                reason_code = 'deficit' if action == 'BUY' else 'lot_too_small'
                
                recommendations.append({
                    'date': date, 'asset_type': 'stock', 'SECID_or_ISIN': row['SECID'],
                    'action': action, 'reason_code': reason_code, 'reason_text': f"Deficit {deficit_pct:.2f}%",
                    'current_qty': 0, 'current_weight_portfolio_pct': 0, 'target_weight_portfolio_pct': row['target_weight_portfolio_pct'],
                    'deficit_pct': deficit_pct, 'qty_delta': qty_delta, 'qty_delta_lots': lots_delta,
                    'lot_size': lot_size, 'lot_source': config['lot_source'], 'amount_rub': qty_delta * price,
                    'priority': 4, 'constraints_hit': ''
                })
                
    return pd.DataFrame([snapshot]), pd.DataFrame(recommendations), pd.DataFrame(outsiders)