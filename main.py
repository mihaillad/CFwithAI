import sys
from pathlib import Path
import pandas as pd

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent / 'src'))

from common import load_config, setup_logging, save_csv, get_file_hash
from module1_stocks import rank_stocks
from module2_bonds import rank_bonds
from module3_portfolio import evaluate_portfolio

def main():
    project_dir = Path(__file__).parent
    config_path = project_dir / 'config.yaml'
    data_dir = project_dir / 'data' / 'raw'
    output_dir = project_dir / 'data' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(config_path)
    date = config.get('date')
    
    logger = setup_logging(date, output_dir)
    logger.info(f"Starting run for date: {date}")
    
    diagnostics = []
    
    # Аудит входных файлов (§5)
    for f in data_dir.glob('*.csv'):
        diagnostics.append({
            'file': f.name,
            'mtime': f.stat().st_mtime,
            'sha256': get_file_hash(f)
        })
        
    try:
        logger.info("Running Module 1: Stocks")
        stock_ranking, stock_excluded = rank_stocks(date, config, data_dir, logger, diagnostics)
        save_csv(stock_ranking, output_dir / 'stock_ranking.csv')
        save_csv(stock_excluded, output_dir / 'stock_excluded.csv')
        
        logger.info("Running Module 2: Bonds")
        bond_ranking, bond_allocation = rank_bonds(date, config, data_dir, logger, diagnostics)
        save_csv(bond_ranking, output_dir / 'bond_ranking.csv')
        save_csv(bond_allocation, output_dir / 'bond_allocation.csv')
        
        logger.info("Running Module 3: Portfolio")
        snapshot, rebalance, outsiders = evaluate_portfolio(date, config, data_dir, stock_ranking, bond_ranking, logger, diagnostics)
        save_csv(snapshot, output_dir / 'portfolio_snapshot.csv')
        save_csv(rebalance, output_dir / 'rebalance_recommendations.csv')
        save_csv(outsiders, output_dir / 'outsiders.csv')
        
        from module4_report import build_report
        build_report(date, output_dir, output_dir, logger)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
    finally:
        diag_df = pd.DataFrame(diagnostics)
        save_csv(diag_df, output_dir / 'diagnostics.csv')
        logger.info("Run completed successfully.")

if __name__ == '__main__':
    main()