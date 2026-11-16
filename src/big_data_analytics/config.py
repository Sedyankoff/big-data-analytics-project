from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


RANDOM_SEED = 42


DATA_START_DATE = date(2025, 1, 1)
DATA_END_DATE = date(2025, 12, 31)

CUSTOMER_COUNT = 500
PRODUCT_COUNT = 30
ORDER_COUNT = 2_000
REVIEW_COUNT = 800