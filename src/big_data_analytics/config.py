from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


RANDOM_SEED = 42


DATA_START_DATE = date(2025, 1, 1)
DATA_END_DATE = date(2025, 12, 31)

CUSTOMER_COUNT = 500
PRODUCT_COUNT = 30
ORDER_COUNT = 2_000
REVIEW_COUNT = 800

# Delivered and shipped orders are paid sales; cancelled orders are excluded.
REVENUE_STATUSES = ["Delivered", "Shipped"]

TEST_SIZE = 0.2
FORECAST_HORIZON = 30
