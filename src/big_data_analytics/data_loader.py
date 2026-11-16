
import pandas as pd

from big_data_analytics.config import RAW_DATA_DIR


def load_datasets():
    datasets = {}

    for name in [
        "customers",
        "products",
        "orders",
        "order_items",
        "reviews",
    ]:
        file_path = RAW_DATA_DIR / f"{name}.csv"
        datasets[name] = pd.read_csv(file_path)

    return datasets


def inspect_datasets(datasets):
    for name, dataframe in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"{name.upper()}")
        print(f"{'=' * 50}")

        print(f"Records: {len(dataframe):,}")
        print(f"Columns: {list(dataframe.columns)}")

        print("\nData types:")
        print(dataframe.dtypes)

        print("\nFirst 3 records:")
        print(dataframe.head(3))


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
