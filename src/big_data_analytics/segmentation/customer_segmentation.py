import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from big_data_analytics.config import (
    DATA_END_DATE,
    DATA_START_DATE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REVENUE_STATUSES,
)
from big_data_analytics.plotting import PALETTE, save_figure


FEATURE_COLUMNS = [
    "purchase_frequency",
    "total_spending",
    "average_order_value",
    "recency_days",
]

CLUSTER_RANGE = range(2, 9)


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
    # Only paid purchases count
    paid_orders = orders[
        orders["status"].isin(REVENUE_STATUSES)
    ]

    order_revenue = (
        items.groupby("order_id")["line_total"]
        .sum()
        .reset_index()
    )

    customer_orders = paid_orders.merge(
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
        ["customer_id", "customer_category"]
    ].merge(
        customer_features,
        on="customer_id",
        how="left",
    )

    for column in ["purchase_frequency", "total_spending", "average_order_value"]:
        customer_features[column] = customer_features[column].fillna(0)

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
        .fillna((DATA_END_DATE - DATA_START_DATE).days + 1)
        .astype(int)
    )

    return customer_features


def select_cluster_count(scaled_features):
    results = []

    for k in CLUSTER_RANGE:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
        )

        labels = model.fit_predict(scaled_features)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(scaled_features, labels),
        })

    results = pd.DataFrame(results).set_index("k")

    print("\nCLUSTER SELECTION")
    print("=" * 45)
    print(results.round(3))

    best_k = int(results["silhouette"].idxmax())

    print(f"\nSelected number of clusters: {best_k} (highest silhouette score)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(results.index, results["inertia"], marker="o", color=PALETTE[0])
    axes[0].axvline(best_k, color="#898781", linewidth=1)
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(results.index, results["silhouette"], marker="o", color=PALETTE[0])
    axes[1].axvline(best_k, color="#898781", linewidth=1)
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")

    save_figure(fig, "15_cluster_selection")

    return best_k


def name_segments(profile):
    """Give each cluster a business name based on its average profile."""
    overall = profile.mean()
    names = {}

    for cluster, row in profile.iterrows():
        if row["purchase_frequency"] < 1:
            name = "Inactive"
        elif row["total_spending"] == profile["total_spending"].max():
            name = "High-Value"
        elif row["recency_days"] > overall["recency_days"]:
            name = "At Risk"
        elif row["average_order_value"] > overall["average_order_value"]:
            name = "Big-Basket"
        elif row["purchase_frequency"] > overall["purchase_frequency"]:
            name = "Loyal"
        else:
            name = "Occasional"

        names[cluster] = name

    # Clusters with identical names are numbered by spending
    counts = pd.Series(names).value_counts()
    for name in counts[counts > 1].index:
        clusters = [cluster for cluster, value in names.items() if value == name]
        clusters.sort(key=lambda cluster: -profile.loc[cluster, "total_spending"])
        for position, cluster in enumerate(clusters, start=1):
            names[cluster] = f"{name} {position}"

    return names


def perform_clustering(customer_features):
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        customer_features[FEATURE_COLUMNS]
    )

    best_k = select_cluster_count(scaled_features)

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_SEED,
        n_init=10,
    )

    customer_features["cluster"] = final_model.fit_predict(
        scaled_features
    )

    profile = customer_features.groupby("cluster")[FEATURE_COLUMNS].mean()
    customer_features["segment"] = customer_features["cluster"].map(
        name_segments(profile)
    )

    return customer_features


def analyze_segments(customer_features):
    print("\nCUSTOMER SEGMENT ANALYSIS")
    print("=" * 55)

    total_revenue = customer_features["total_spending"].sum()

    segment_summary = customer_features.groupby("segment").agg(
        customers=("customer_id", "count"),
        average_orders=("purchase_frequency", "mean"),
        average_spending=("total_spending", "mean"),
        average_order_value=("average_order_value", "mean"),
        average_recency=("recency_days", "mean"),
        revenue=("total_spending", "sum"),
    )

    segment_summary["customer_share_%"] = (
        segment_summary["customers"] / segment_summary["customers"].sum() * 100
    )
    segment_summary["revenue_share_%"] = segment_summary["revenue"] / total_revenue * 100
    segment_summary = segment_summary.sort_values("average_spending", ascending=False)

    print(segment_summary.round(2))

    print("\nCUSTOMER CATEGORY BY SEGMENT (% of segment)")
    print("=" * 55)
    print(
        (pd.crosstab(
            customer_features["segment"],
            customer_features["customer_category"],
            normalize="index",
        ) * 100).round(1)
    )

    print("\nINTERPRETATION")
    print("=" * 55)

    for segment, row in segment_summary.iterrows():
        print(
            f"- {segment}: {row['customer_share_%']:.1f}% of customers, "
            f"{row['revenue_share_%']:.1f}% of revenue; "
            f"{row['average_orders']:.1f} orders, average order value "
            f"{row['average_order_value']:,.2f}, last purchase "
            f"{row['average_recency']:.0f} days before the end of the period."
        )

    return segment_summary


def create_visualizations(customer_features, segment_summary):
    segments = list(segment_summary.index)
    colors = dict(zip(segments, PALETTE))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    sns.scatterplot(
        data=customer_features,
        x="purchase_frequency",
        y="total_spending",
        hue="segment",
        hue_order=segments,
        palette=colors,
        alpha=0.7,
        s=45,
        ax=axes[0],
    )
    axes[0].set_title("Customer segments")
    axes[0].set_xlabel("Purchase frequency")
    axes[0].set_ylabel("Total spending")

    sns.boxplot(
        data=customer_features,
        x="segment",
        y="total_spending",
        order=segments,
        hue="segment",
        palette=colors,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Spending distribution by segment")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total spending")
    axes[1].tick_params(axis="x", rotation=20)

    positions = np.arange(len(segments))
    axes[2].bar(positions - 0.2, segment_summary["customer_share_%"], width=0.4,
                color=PALETTE[0], label="Customers, %")
    axes[2].bar(positions + 0.2, segment_summary["revenue_share_%"], width=0.4,
                color=PALETTE[1], label="Revenue, %")
    axes[2].set_xticks(positions, segments, rotation=20)
    axes[2].set_title("Share of customers vs share of revenue")
    axes[2].set_ylabel("%")
    axes[2].legend()

    save_figure(fig, "16_customer_segments")


def run_segmentation():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    customers, orders, items, products = load_customer_data()

    customer_features = prepare_customer_features(
        customers,
        orders,
        items,
    )

    customer_features = perform_clustering(
        customer_features
    )

    segment_summary = analyze_segments(customer_features)

    create_visualizations(customer_features, segment_summary)

    output_path = PROCESSED_DATA_DIR / "customer_segments.csv"

    customer_features.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nCustomer segments saved to: {output_path}")


if __name__ == "__main__":
    run_segmentation()
