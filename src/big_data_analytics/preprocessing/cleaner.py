import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from big_data_analytics.config import PROCESSED_DATA_DIR
from big_data_analytics.data_loader import load_datasets, validate_relationships
from big_data_analytics.plotting import save_figure


DATE_FORMATS = ["%Y-%m-%d", "%d.%m.%Y"]

PRIMARY_KEYS = {
    "customers": "customer_id",
    "products": "product_id",
    "orders": "order_id",
    "order_items": "order_item_id",
    "reviews": "review_id",
}

CUSTOMER_CATEGORY_ORDER = ["New", "Regular", "VIP"]


def parse_dates(values):
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    for date_format in DATE_FORMATS:
        missing = parsed.isna()
        parsed[missing] = pd.to_datetime(
            values[missing],
            format=date_format,
            errors="coerce",
        )

    return parsed


def detect_outliers(values, z_threshold=3):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    iqr_outliers = (
        (values < q1 - 1.5 * iqr) |
        (values > q3 + 1.5 * iqr)
    )

    standard_deviation = values.std()

    if standard_deviation > 0:
        z_scores = ((values - values.mean()) / standard_deviation).abs()
        zscore_outliers = z_scores > z_threshold
    else:
        zscore_outliers = pd.Series(False, index=values.index)

    return iqr_outliers, zscore_outliers


def clean_datasets(datasets):
    cleaned = {
        name: dataframe.copy()
        for name, dataframe in datasets.items()
    }

    print("\nCLEANING STEPS")
    print("=" * 55)

    # Duplicates: identical rows and repeated primary keys
    for name, key in PRIMARY_KEYS.items():
        rows_before = len(cleaned[name])

        cleaned[name] = (
            cleaned[name]
            .drop_duplicates()
            .drop_duplicates(subset=key, keep="first")
            .reset_index(drop=True)
        )

        removed = rows_before - len(cleaned[name])
        print(f"{name}: removed {removed} duplicate rows")

    # Customers: inconsistent text, invalid ages and missing values
    customers = cleaned["customers"]

    customers["city"] = customers["city"].str.strip().str.title()

    age = pd.to_numeric(customers["age"], errors="coerce")
    invalid_age = age.notna() & ~age.between(18, 100)
    print(f"customers: {int(invalid_age.sum())} invalid ages, "
          f"{int(age.isna().sum())} missing ages")

    age[invalid_age] = np.nan
    customers["age"] = age.fillna(age.median()).round().astype(int)

    print(f"customers: {int(customers['city'].isna().sum())} missing cities "
          "filled with the most frequent city")
    customers["city"] = customers["city"].fillna(customers["city"].mode()[0])

    customers["registration_date"] = parse_dates(customers["registration_date"])

    # Products: numeric columns stored as text with decimal commas
    products = cleaned["products"]

    for column in ["selling_price", "cost"]:
        products[column] = pd.to_numeric(
            products[column].astype(str).str.replace(",", ".").str.strip(),
            errors="coerce",
        )

    # Orders: date formats, label consistency and missing payment methods
    orders = cleaned["orders"]

    orders["status"] = orders["status"].str.strip().str.title()

    print(f"orders: {int(orders['payment_method'].isna().sum())} missing "
          "payment methods set to 'Unknown'")
    orders["payment_method"] = orders["payment_method"].fillna("Unknown")

    orders["order_date"] = parse_dates(orders["order_date"])

    invalid_dates = orders["order_date"].isna()
    print(f"orders: {int(invalid_dates.sum())} orders with unrecoverable dates removed")

    orders = orders[
        ~invalid_dates & orders["customer_id"].isin(customers["customer_id"])
    ].reset_index(drop=True)
    cleaned["orders"] = orders

    # Order items: keep only valid orders and products, fix prices and quantities
    items = cleaned["order_items"]

    items = items[
        items["order_id"].isin(orders["order_id"])
        & items["product_id"].isin(products["product_id"])
    ].copy()

    for column in ["quantity", "unit_price", "line_total"]:
        items[column] = pd.to_numeric(items[column], errors="coerce")

    missing_prices = items["unit_price"].isna()
    print(f"order_items: {int(missing_prices.sum())} missing unit prices restored")

    catalog_prices = items["product_id"].map(
        products.set_index("product_id")["selling_price"]
    )
    items["unit_price"] = (
        items["unit_price"]
        .fillna(items["line_total"] / items["quantity"])
        .fillna(catalog_prices)
        .round(2)
    )

    iqr_outliers, zscore_outliers = detect_outliers(items["quantity"])
    outliers = iqr_outliers & zscore_outliers

    print(f"order_items: quantity outliers - IQR: {int(iqr_outliers.sum())}, "
          f"Z-score: {int(zscore_outliers.sum())}, both methods: {int(outliers.sum())}")
    print("order_items: values flagged by both methods are replaced with the median; "
          "IQR-only values are legitimate multi-unit purchases and are kept")

    items.loc[outliers, "quantity"] = items["quantity"].median()
    items["quantity"] = items["quantity"].astype(int)

    expected_totals = (items["quantity"] * items["unit_price"]).round(2)
    inconsistent = (items["line_total"] - expected_totals).abs() > 0.01
    print(f"order_items: {int(inconsistent.sum())} line totals recalculated "
          "from quantity x unit price")

    items["line_total"] = expected_totals
    cleaned["order_items"] = items.reset_index(drop=True)

    # Reviews: missing texts, invalid ratings and removed orders
    reviews = cleaned["reviews"]

    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    invalid_reviews = (
        reviews["review_text"].isna()
        | ~reviews["rating"].between(1, 5)
        | ~reviews["order_id"].isin(orders["order_id"])
    )
    print(f"reviews: {int(invalid_reviews.sum())} reviews with missing text, "
          "invalid rating or unknown order removed")

    reviews = reviews[~invalid_reviews].copy()
    reviews["rating"] = reviews["rating"].astype(int)
    reviews["review_date"] = parse_dates(reviews["review_date"])
    cleaned["reviews"] = reviews.reset_index(drop=True)

    return cleaned


def print_quality_report(before, after):
    print("\nDATA QUALITY REPORT")
    print("=" * 55)

    for name, dataframe in before.items():
        cleaned = after[name]

        missing_before = int(dataframe.isna().sum().sum())
        missing_after = int(cleaned.isna().sum().sum())

        duplicates_before = int(dataframe.duplicated().sum())
        duplicates_after = int(cleaned.duplicated().sum())

        print(f"\n{name.upper()}")
        print(f"Rows before: {len(dataframe):,}")
        print(f"Rows after:  {len(cleaned):,}")
        print(f"Missing values: {missing_before} -> {missing_after}")
        print(f"Duplicate rows: {duplicates_before} -> {duplicates_after}")

    print("\nDATA TYPES AFTER CLEANING")
    print("=" * 55)
    for name, dataframe in after.items():
        types = ", ".join(f"{column}: {dtype}" for column, dtype in dataframe.dtypes.items())
        print(f"{name}: {types}")


def compare_statistics(before, after):
    print("\nSTATISTICS BEFORE AND AFTER CLEANING")
    print("=" * 55)

    for column in ["quantity", "line_total"]:
        comparison = pd.DataFrame({
            "before": pd.to_numeric(
                before["order_items"][column], errors="coerce"
            ).describe(),
            "after": after["order_items"][column].describe(),
        })

        print(f"\n{column}")
        print(comparison.round(2))


def plot_outliers(before, after):
    quantity_before = pd.to_numeric(before["order_items"]["quantity"], errors="coerce")
    quantity_after = after["order_items"]["quantity"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sns.boxplot(y=quantity_before, ax=axes[0], color="#2a78d6")
    axes[0].set_yscale("log")
    axes[0].set_title("Quantity before cleaning (log scale)")
    axes[0].set_ylabel("Quantity")

    sns.boxplot(y=quantity_after, ax=axes[1], color="#2a78d6")
    axes[1].set_title("Quantity after cleaning")
    axes[1].set_ylabel("Quantity")

    sales = after["order_items"].merge(
        after["products"][["product_id", "category"]],
        on="product_id",
    )
    sns.boxplot(data=sales, x="category", y="line_total", ax=axes[2], color="#2a78d6")
    axes[2].set_title("Line total by category")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Line total")
    axes[2].tick_params(axis="x", rotation=30)

    return save_figure(fig, "01_outliers_boxplots")


def encode_categorical_features(datasets):
    customers = datasets["customers"].copy()
    orders = datasets["orders"].copy()

    # Label encoding: ordinal mapping for the customer category,
    # alphabetical codes for the city
    customers["customer_category_code"] = customers["customer_category"].map(
        {category: code for code, category in enumerate(CUSTOMER_CATEGORY_ORDER)}
    )

    city_encoder = LabelEncoder()
    customers["city_code"] = city_encoder.fit_transform(customers["city"])

    # One-hot encoding for nominal order attributes
    orders = pd.get_dummies(
        orders,
        columns=["status", "payment_method"],
        prefix=["status", "payment"],
        dtype=int,
    )

    print("\nCATEGORICAL ENCODING")
    print("=" * 55)
    print("Label encoding (customer_category):",
          dict(enumerate(CUSTOMER_CATEGORY_ORDER)))
    print("Label encoding (city):",
          dict(enumerate(city_encoder.classes_)))
    print("One-hot columns (orders):",
          [column for column in orders.columns
           if column.startswith(("status_", "payment_"))])

    return customers, orders


def save_cleaned_datasets(datasets):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataframe in datasets.items():
        dataframe.to_csv(
            PROCESSED_DATA_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def preprocess():
    raw_datasets = load_datasets()

    cleaned_datasets = clean_datasets(raw_datasets)

    print_quality_report(raw_datasets, cleaned_datasets)
    compare_statistics(raw_datasets, cleaned_datasets)
    validate_relationships(cleaned_datasets)

    figure_path = plot_outliers(raw_datasets, cleaned_datasets)
    print(f"\nBox plots saved to: {figure_path}")

    customers_encoded, orders_encoded = encode_categorical_features(cleaned_datasets)

    save_cleaned_datasets(cleaned_datasets)
    save_cleaned_datasets({
        "customers_encoded": customers_encoded,
        "orders_encoded": orders_encoded,
    })

    print(f"\nCleaned datasets saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    preprocess()
