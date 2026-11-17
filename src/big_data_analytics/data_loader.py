import pandas as pd

from big_data_analytics.config import RAW_DATA_DIR


DATASET_NAMES = [
    "customers",
    "products",
    "orders",
    "order_items",
    "reviews",
]

MINIMUM_SIZES = {
    "customers": 100,
    "products": 20,
    "orders": 1000,
    "reviews": 300,
}


def load_datasets(directory=RAW_DATA_DIR):
    datasets = {}

    for name in DATASET_NAMES:
        file_path = directory / f"{name}.csv"
        datasets[name] = pd.read_csv(file_path)

    return datasets


def inspect_datasets(datasets):
    for name, dataframe in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"{name.upper()}")
        print(f"{'=' * 50}")

        print(f"Shape: {dataframe.shape[0]:,} rows x {dataframe.shape[1]} columns")
        print(f"Columns: {list(dataframe.columns)}")

        print("\nData types:")
        print(dataframe.dtypes)

        missing = dataframe.isna().sum()
        print(f"\nMissing values: {int(missing.sum())}")
        if missing.any():
            print(missing[missing > 0])

        print(f"Duplicate rows: {int(dataframe.duplicated().sum())}")

        numeric = dataframe.select_dtypes("number")
        if not numeric.empty:
            print("\nBasic statistics:")
            print(numeric.describe().round(2))

        print("\nFirst 3 records:")
        print(dataframe.head(3))


def validate_dataset_requirements(datasets):
    print("\nDATASET REQUIREMENTS")
    print("-" * 50)

    checks = {}

    for name, minimum in MINIMUM_SIZES.items():
        unique_rows = len(datasets[name].drop_duplicates())
        checks[f"{name}: {unique_rows:,} records (minimum {minimum:,})"] = (
            unique_rows >= minimum
        )

    order_dates = pd.to_datetime(
        datasets["orders"]["order_date"],
        format="%Y-%m-%d",
        errors="coerce",
    ).dropna()

    months = order_dates.dt.to_period("M").nunique()
    checks[
        f"Orders cover {months} months "
        f"({order_dates.min().date()} to {order_dates.max().date()})"
    ] = months >= 12

    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")

    return all(checks.values())


def validate_relationships(datasets):
    customers = datasets["customers"]
    products = datasets["products"]
    orders = datasets["orders"]
    items = datasets["order_items"]
    reviews = datasets["reviews"]

    customer_ids = set(customers["customer_id"])
    product_ids = set(products["product_id"])
    order_ids = set(orders["order_id"])

    checks = {
        "Orders reference existing customers":
            orders["customer_id"].isin(customer_ids).all(),

        "Order items reference existing orders":
            items["order_id"].isin(order_ids).all(),

        "Order items reference existing products":
            items["product_id"].isin(product_ids).all(),

        "Reviews reference existing orders":
            reviews["order_id"].isin(order_ids).all(),

        "Reviews reference existing customers":
            reviews["customer_id"].isin(customer_ids).all(),

        "Reviews reference existing products":
            reviews["product_id"].isin(product_ids).all(),
    }

    print("\nRELATIONSHIP VALIDATION")
    print("-" * 50)

    for description, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {description}")

    if not all(checks.values()):
        raise ValueError("One or more relationship checks failed.")
