import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def load_data() -> pd.DataFrame:
    PROJECT_ROOT = Path(__file__).parent.parent
    raw_path = PROJECT_ROOT / config["paths"]["raw_data"]

    if not raw_path.exists():
        logger.error("Data file does not found at: %s", raw_path)
        raise FileNotFoundError(f"Data file not found at: {raw_path}")

    df = pd.read_csv(raw_path)
    expected_columns = set(
        config["columns"]["numeric"]
        + config["columns"]["categorical"]
        + [config["target"]["column"]]
    )
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns
    if missing:
        logger.error("Missing expected columns: %s", missing)
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info(
        "Data loaded successfully: %d rows, %d columns", df.shape[0], df.shape[1]
    )
    return df
