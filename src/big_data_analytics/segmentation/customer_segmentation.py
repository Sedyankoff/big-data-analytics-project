import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from big_data_analytics.config import (
    DATA_END_DATE,
    PROCESSED_DATA_DIR,
)


def load_customer_data():
    customers = pd.read_csv(
        PROCESSED_DATA_DIR / "customers.csv"
    )

    orders = pd.read_csv(
        PROCESSED_DATA_DIR / "orders.csv"
    )

    items = pd.read_csv(
        PROCESSED_DATA_DIR / "order_items.csv"
    )

    products = pd.read_csv(
        PROCESSED_DATA_DIR / "products.csv"
    )

    return customers, orders, items, products


def prepare_customer_features(customers, orders, items):
    # Only completed purchases count
    delivered_orders = orders[
        orders["status"] == "Delivered"
    ]

    order_revenue = (
        items.groupby("order_id")["line_total"]
        .sum()
        .reset_index()
    )

    customer_orders = delivered_orders.merge(
        order_revenue,
        on="order_id",
        how="inner",
    )

    customer_features = customer_orders.groupby(
        "customer_id"
    ).agg(
        purchase_frequency=("order_id", "count"),
        total_spending=("line_total", "sum"),
        average_order_value=("line_total", "mean"),
        last_purchase=("order_date", "max"),
    ).reset_index()

    # Include customers who have never completed an order
    customer_features = customers[
        ["customer_id"]
    ].merge(
        customer_features,
        on="customer_id",
        how="left",
    )

    customer_features["purchase_frequency"] = (
        customer_features["purchase_frequency"].fillna(0)
    )

    customer_features["total_spending"] = (
        customer_features["total_spending"].fillna(0)
    )

    customer_features["average_order_value"] = (
        customer_features["average_order_value"].fillna(0)
    )

    customer_features["last_purchase"] = pd.to_datetime(
        customer_features["last_purchase"]
    )

    customer_features["recency_days"] = (
        pd.Timestamp(DATA_END_DATE)
        - customer_features["last_purchase"]
    ).dt.days

    # Customers without purchases receive a recency value
    # equal to the full observation period
    customer_features["recency_days"] = (
        customer_features["recency_days"]
        .fillna(
            (DATA_END_DATE - pd.Timestamp("2025-01-01").date()).days
        )
    )

    return customer_features


def perform_clustering(customer_features):
    feature_columns = [
        "purchase_frequency",
        "total_spending",
        "average_order_value",
    ]

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        customer_features[feature_columns]
    )

    # Compare several cluster counts
    scores = {}

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(scaled_features)

        scores[k] = silhouette_score(
            scaled_features,
            labels,
        )

    print("\nCLUSTER SELECTION")
    print("=" * 45)

    for k, score in scores.items():
        print(f"K={k}: Silhouette Score = {score:.3f}")

    best_k = max(scores, key=scores.get)

    print(f"\nSelected number of clusters: {best_k}")

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    customer_features["segment"] = final_model.fit_predict(
        scaled_features
    )

    return customer_features


def analyze_segments(customer_features):
    print("\nCUSTOMER SEGMENT ANALYSIS")
    print("=" * 55)

    segment_summary = customer_features.groupby("segment").agg(
        customers=("customer_id", "count"),
        average_orders=("purchase_frequency", "mean"),
        average_spending=("total_spending", "mean"),
        average_order_value=("average_order_value", "mean"),
        average_recency=("recency_days", "mean"),
    ).round(2)

    print(segment_summary)

    return segment_summary


def create_visualizations(customer_features):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=customer_features,
        x="purchase_frequency",
        y="total_spending",
        hue="segment",
        palette="viridis",
        s=80,
    )

    plt.title("Customer Segmentation")
    plt.xlabel("Purchase Frequency")
    plt.ylabel("Total Spending")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=customer_features,
        x="segment",
        y="total_spending",
    )

    plt.title("Spending Distribution by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Total Spending")
    plt.tight_layout()
    plt.show()


def run_segmentation():
    customers, orders, items, products = load_customer_data()

    customer_features = prepare_customer_features(
        customers,
        orders,
        items,
    )

    customer_features = perform_clustering(
        customer_features
    )

    analyze_segments(customer_features)

    create_visualizations(customer_features)

    output_path = PROCESSED_DATA_DIR / "customer_segments.csv"

    customer_features.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nCustomer segments saved to: {output_path}")


if __name__ == "__main__":
    run_segmentation()
