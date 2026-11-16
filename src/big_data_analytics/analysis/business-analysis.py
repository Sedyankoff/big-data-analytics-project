
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from big_data_analytics.config import PROCESSED_DATA_DIR


def load_clean_data():
    datasets = {}

    for name in [
        "customers",
        "products",
        "orders",
        "order_items",
        "reviews",
    ]:
        datasets[name] = pd.read_csv(
            PROCESSED_DATA_DIR / f"{name}.csv"
        )

    return datasets


def prepare_business_data(datasets):
    orders = datasets["orders"].copy()
    items = datasets["order_items"].copy()
    products = datasets["products"].copy()
    customers = datasets["customers"].copy()

    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # Add product costs to individual order items
    items = items.merge(
        products[["product_id", "cost", "category"]],
        on="product_id",
        how="left",
    )

    items["revenue"] = items["quantity"] * items["unit_price"]
    items["total_cost"] = items["quantity"] * items["cost"]
    items["profit"] = items["revenue"] - items["total_cost"]

    # Aggregate individual items into order-level metrics
    order_summary = items.groupby("order_id").agg(
        order_revenue=("revenue", "sum"),
        order_cost=("total_cost", "sum"),
        order_profit=("profit", "sum"),
        item_count=("product_id", "count"),
        total_quantity=("quantity", "sum"),
    ).reset_index()

    orders = orders.merge(
        order_summary,
        on="order_id",
        how="left",
    )

    orders = orders.merge(
        customers[["customer_id", "age", "city", "customer_category"]],
        on="customer_id",
        how="left",
    )

    return items, orders


def descriptive_statistics(items, orders):
    print("\nPRODUCT PRICE AND QUANTITY STATISTICS")
    print("=" * 55)

    print(items[["unit_price", "quantity", "line_total"]].describe())

    print("\nORDER VALUE STATISTICS")
    print("=" * 55)

    print(orders["order_revenue"].describe())

    print("\nADDITIONAL STATISTICS")
    print("=" * 55)

    for column in ["unit_price", "quantity", "order_revenue"]:
        values = orders[column] if column == "order_revenue" else items[column]

        print(f"\n{column}")
        print(f"Mean: {values.mean():.2f}")
        print(f"Median: {values.median():.2f}")
        print(f"Mode: {values.mode().iloc[0]:.2f}")
        print(f"Variance: {values.var():.2f}")
        print(f"Standard deviation: {values.std():.2f}")
        print(f"Minimum: {values.min():.2f}")
        print(f"Maximum: {values.max():.2f}")


def business_analysis(items, orders):
    # Cancelled orders are excluded from realized business results
    completed_orders = orders[
        orders["status"] == "Delivered"
    ]

    delivered_items = items[
        items["order_id"].isin(completed_orders["order_id"])
    ]

    total_revenue = delivered_items["revenue"].sum()
    total_cost = delivered_items["total_cost"].sum()
    total_profit = total_revenue - total_cost

    average_order_value = completed_orders["order_revenue"].mean()

    profit_margin = (
        total_profit / total_revenue * 100
        if total_revenue else 0
    )

    print("\nBUSINESS PERFORMANCE")
    print("=" * 55)
    print(f"Delivered orders: {len(completed_orders):,}")
    print(f"Revenue: {total_revenue:,.2f}")
    print(f"Cost: {total_cost:,.2f}")
    print(f"Profit: {total_profit:,.2f}")
    print(f"Profit margin: {profit_margin:.2f}%")
    print(f"Average order value: {average_order_value:,.2f}")

    print("\nREVENUE BY PRODUCT CATEGORY")
    print("=" * 55)

    category_performance = (
        delivered_items.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            cost=("total_cost", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )

    print(category_performance.round(2))

    print("\nTOP 10 CUSTOMERS BY REVENUE")
    print("=" * 55)

    top_customers = (
        completed_orders.groupby("customer_id")
        .agg(
            total_spent=("order_revenue", "sum"),
            order_count=("order_id", "count"),
        )
        .sort_values("total_spent", ascending=False)
        .head(10)
    )

    print(top_customers.round(2))

    return category_performance, top_customers


def correlation_analysis(items, orders):
    print("\nCORRELATION ANALYSIS")
    print("=" * 55)

    numeric_data = orders[
        ["order_revenue", "order_cost", "order_profit",
         "item_count", "total_quantity", "age"]
    ]

    print("\nPearson correlation:")
    print(numeric_data.corr(method="pearson").round(2))

    print("\nSpearman correlation:")
    print(numeric_data.corr(method="spearman").round(2))

    correlation, p_value = stats.pearsonr(
        orders["order_revenue"],
        orders["total_quantity"],
    )

    print(
        "\nOrder revenue vs quantity: "
        f"Pearson r={correlation:.3f}, p-value={p_value:.5f}"
    )


def distribution_analysis():
    rng = np.random.default_rng(42)

    normal = rng.normal(loc=50, scale=10, size=1000)
    uniform = rng.uniform(low=0, high=100, size=1000)
    binomial = rng.binomial(n=10, p=0.5, size=1000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(normal, kde=True, ax=axes[0])
    axes[0].set_title("Normal Distribution")

    sns.histplot(uniform, kde=True, ax=axes[1])
    axes[1].set_title("Uniform Distribution")

    sns.histplot(binomial, discrete=True, ax=axes[2])
    axes[2].set_title("Binomial Distribution")

    plt.tight_layout()
    plt.show()


def create_visualizations(items, orders, category_performance):
    # Order value distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(orders["order_revenue"], bins=30, kde=True)
    plt.title("Order Value Distribution")
    plt.xlabel("Order value")
    plt.ylabel("Number of orders")
    plt.tight_layout()
    plt.show()

    # Revenue by category
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=category_performance.reset_index(),
        x="category",
        y="revenue",
    )
    plt.title("Revenue by Product Category")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    numeric_data = orders[
        ["order_revenue", "order_cost", "order_profit",
         "item_count", "total_quantity", "age"]
    ]

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Order Metrics Correlation")
    plt.tight_layout()
    plt.show()

    # Revenue over time
    monthly_revenue = (
        orders[orders["status"] == "Delivered"]
        .set_index("order_date")["order_revenue"]
        .resample("ME")
        .sum()
    )

    plt.figure(figsize=(10, 5))
    monthly_revenue.plot(marker="o")
    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()


def main():
    datasets = load_clean_data()

    items, orders = prepare_business_data(datasets)

    descriptive_statistics(items, orders)

    category_performance, _ = business_analysis(items, orders)

    correlation_analysis(items, orders)

    distribution_analysis()

    create_visualizations(items, orders, category_performance)


if __name__ == "__main__":
    main()
