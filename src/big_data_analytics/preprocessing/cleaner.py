import pandas as pd

from big_data_analytics.config import PROCESSED_DATA_DIR
from big_data_analytics.data_loader import load_datasets


def clean_datasets(datasets):
    cleaned = {
        name: dataframe.copy()
        for name, dataframe in datasets.items()
    }

    cleaned["customers"] = cleaned["customers"].drop_duplicates(
        subset="customer_id",
        keep="first",
    )

    customers = cleaned["customers"]
    customers["city"] = customers["city"].fillna(
        customers["city"].mode()[0]
    )

    reviews = cleaned["reviews"]
    reviews = reviews.dropna(subset=["review_text"])
    cleaned["reviews"] = reviews

    date_columns = {
        "customers": ["registration_date"],
        "orders": ["order_date"],
        "reviews": ["review_date"],
    }

    for dataset_name, columns in date_columns.items():
        for column in columns:
            cleaned[dataset_name][column] = pd.to_datetime(
                cleaned[dataset_name][column],
                errors="coerce",
            )

    orders = cleaned["orders"]
    orders["order_date"] = orders["order_date"].fillna(
        orders["order_date"].median()
    )

    items = cleaned["order_items"]
    quantity = pd.to_numeric(items["quantity"])

    q1 = quantity.quantile(0.25)
    q3 = quantity.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    iqr_outliers = (
        (quantity < lower_bound) |
        (quantity > upper_bound)
    )

    standard_deviation = quantity.std()

    if standard_deviation > 0:
        z_scores = (
            (quantity - quantity.mean()) / standard_deviation
        ).abs()
        zscore_outliers = z_scores > 3
    else:
        zscore_outliers = pd.Series(False, index=quantity.index)

    outliers = iqr_outliers | zscore_outliers

    print(f"Quantity outliers detected: {outliers.sum()}")

    items.loc[outliers, "quantity"] = int(quantity.median())

    items["line_total"] = (
        items["quantity"] * items["unit_price"]
    ).round(2)

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

    save_cleaned_datasets(cleaned_datasets)

    print(f"\nCleaned datasets saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    preprocess()