import pandas as pd
import yaml
import logging
from pathlib import Path
import hashlib

def get_file_hash(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def validate_schema(df: pd.DataFrame, schema: dict) -> bool:
    for col, dtype in schema.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return True

def setup_logging(date: str, output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"run_{date}.log"
    # filemode='w' обеспечивает идемпотентность (перезапись)
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger()

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, sep=';', encoding='utf-8', float_format='%.4f')