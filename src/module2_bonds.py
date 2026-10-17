import pandas as pd
import numpy as np
from pathlib import Path
import logging

def rank_bonds(date: str, config: dict, data_dir: Path, logger: logging.Logger, diagnostics: list):
    date_dt = pd.to_datetime(date)
    ofz = pd.read_csv(data_dir / 'bonds_OFZ.csv', sep=';')
    corps = pd.read_csv(data_dir / 'bonds_CORPS.csv', sep=';')
    prices = pd.read_csv(data_dir / 'bond_prices.csv', sep=';')
    cbr = pd.read_csv(data_dir / 'risk_free_rate.csv')  # разделитель ','
    
    # T-17: Дубликаты ISIN
    common_isins = set(ofz['ISIN']).intersection(set(corps['ISIN']))
    if common_isins:
        logger.warning(f"isin_duplicate_between_files: {common_isins}")
        diagnostics.append({'code': 'isin_duplicate_between_files', 'details': str(common_isins)})
        corps = corps[~corps['ISIN'].isin(common_isins)]
        
    bonds = pd.concat([
        ofz.assign(bond_type='ofz', source_file='bonds_OFZ', issuer='RF'),
        corps.assign(bond_type='corporate', source_file='bonds_CORPS')
    ], ignore_index=True)
    
    cbr['date'] = pd.to_datetime(cbr['date'])
    cbr_rate = cbr[cbr['date'] <= date_dt].sort_values('date').iloc[-1]['rate_pct']
    
    prices['date'] = pd.to_datetime(prices['date'])
    latest_prices = prices[prices['date'] <= date_dt].sort_values('date').groupby('ISIN').tail(1).set_index('ISIN')
    
    results = []
    rating_order = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7, 'BBB+': 8, 'BBB': 9}
    min_rating_val = rating_order.get(config['min_credit_rating'], 99)
    
    for _, row in bonds.iterrows():
        isin = row['ISIN']
        if isin not in latest_prices.index:
            continue
            
        p_row = latest_prices.loc[isin]
        years_to_mat = (pd.to_datetime(row['maturity_date']) - date_dt).days / 365.0
        
        bucket = 'excluded'
        exclusion_reason = ''
        eligible = True
        
        if row['bond_type'] == 'ofz':
            if config['ofz_core_min_years'] <= years_to_mat <= config['ofz_core_max_years']:
                bucket = 'core_ofz'
            else:
                exclusion_reason = 'maturity_out_of_range'
                eligible = False
        else:
            if years_to_mat <= config['short_corp_max_years']:
                current_rating_val = rating_order.get(str(row['credit_rating']), 99)
                if current_rating_val > min_rating_val:
                    eligible = False
                    exclusion_reason = 'low_rating'
                else:
                    bucket = 'short_corp'
            else:
                exclusion_reason = 'maturity_out_of_range'
                eligible = False
                
        if pd.isna(p_row['ytm_pct']):
            eligible = False
            exclusion_reason = 'no_ytm'
            
        if p_row['volume_rub'] < config['bond_min_volume_rub']:
            eligible = False
            exclusion_reason = 'low_volume'
            
        ytm_gross = p_row['ytm_pct']
        spread = ytm_gross - cbr_rate
        
        if eligible and spread < config['bond_min_spread_pct']:
            eligible = False
            exclusion_reason = 'spread'
            
        lot_size = int(row['lot_size']) if pd.notna(row['lot_size']) else 1
        if pd.isna(row['lot_size']):
            diagnostics.append({'code': 'lot_missing', 'details': f"ISIN {isin}"})
            
        results.append({
            'date': date, 'ISIN': isin, 'name': row['name'], 'issuer': row.get('issuer', 'RF'),
            'bond_type': row['bond_type'], 'source_file': row['source_file'],
            'maturity_date': row['maturity_date'], 'days_to_maturity': int(years_to_mat * 365),
            'coupon_type': row['coupon_type'], 'coupon_rate_pct': row['coupon_rate_pct'],
            'price_pct': p_row['price_pct'], 'nkd_rub': p_row['nkd_rub'],
            'ytm_gross_pct': ytm_gross, 'ytm_net_pct': ytm_gross * (1 - config['tax_rate_pct']/100),
            'spread_to_cbr_pct': spread, 'duration_years': p_row['duration_years'],
            'credit_rating': row.get('credit_rating', 'AAA'), 'bucket': bucket,
            'eligible': eligible, 'exclusion_reason': exclusion_reason, 'lot_size': lot_size
        })
        
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Сортировка внутри бакетов по YTM desc
        df_res = df_res.sort_values(['bucket', 'ytm_gross_pct'], ascending=[True, False])
        
        # Распределение весов (упрощенное равное распределение с cap)
        for b in ['core_ofz', 'short_corp']:
            mask = (df_res['bucket'] == b) & (df_res['eligible'])
            if mask.sum() > 0:
                target_share = config['ofz_core_share_pct'] if b == 'core_ofz' else config['short_corp_share_pct']
                weight_per_pos = target_share / mask.sum()
                cap = config['issuer_limit_pct']
                df_res.loc[mask, 'target_weight_sleeve_pct'] = min(weight_per_pos, cap)
                df_res.loc[mask, 'target_weight_portfolio_pct'] = df_res.loc[mask, 'target_weight_sleeve_pct'] * (config['bond_sleeve_pct'] / 100)
                
    # Bond allocation summary
    alloc = []
    for b in ['core_ofz', 'short_corp']:
        mask = (df_res['bucket'] == b) & (df_res['eligible'])
        if mask.sum() > 0:
            alloc.append({
                'date': date, 'bucket': b, 'target_share_pct': config['ofz_core_share_pct'] if b=='core_ofz' else config['short_corp_share_pct'],
                'n_positions': int(mask.sum()), 'avg_ytm_gross_pct': df_res.loc[mask, 'ytm_gross_pct'].mean(),
                'avg_ytm_net_pct': df_res.loc[mask, 'ytm_net_pct'].mean(),
                'total_weight_portfolio_pct': df_res.loc[mask, 'target_weight_portfolio_pct'].sum()
            })
            
    return df_res, pd.DataFrame(alloc)