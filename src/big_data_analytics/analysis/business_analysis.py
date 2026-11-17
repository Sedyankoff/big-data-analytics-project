import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from big_data_analytics.config import PROCESSED_DATA_DIR, RANDOM_SEED, REVENUE_STATUSES
from big_data_analytics.plotting import PALETTE, save_figure


CORRELATION_COLUMNS = [
    "order_revenue", "order_cost", "order_profit",
    "item_count", "total_quantity", "age",
]


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
    orders["is_revenue"] = orders["status"].isin(REVENUE_STATUSES)

    # Add product costs to individual order items
    items = items.merge(
        products[["product_id", "product_name", "cost", "category"]],
        on="product_id",
        how="left",
    )

    items["revenue"] = items["line_total"]
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

    items = items.merge(
        orders[["order_id", "order_date", "status", "is_revenue",
                "customer_id", "customer_category"]],
        on="order_id",
        how="left",
    )

    return items, orders


def summary_statistics(values):
    q1, q2, q3 = values.quantile([0.25, 0.50, 0.75])

    return {
        "mean": values.mean(),
        "median": values.median(),
        "mode": values.mode().iloc[0],
        "variance": values.var(),
        "std": values.std(),
        "min": values.min(),
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "IQR": q3 - q1,
        "max": values.max(),
        "skewness": stats.skew(values),
    }


def descriptive_statistics(items, orders):
    revenue_orders = orders[orders["is_revenue"]]

    table = pd.DataFrame({
        "unit_price": summary_statistics(items["unit_price"]),
        "quantity": summary_statistics(items["quantity"]),
        "line_total": summary_statistics(items["line_total"]),
        "order_value": summary_statistics(revenue_orders["order_revenue"]),
    })

    print("\nDESCRIPTIVE STATISTICS")
    print("=" * 55)
    print(table.round(2))

    print(
        "\nAll variables are right-skewed (skewness > 0): the mean is above the "
        "median, so the median is the more representative 'typical' value."
    )

    return table


def distribution_analysis(items, orders):
    revenue_orders = orders[orders["is_revenue"]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    sns.histplot(items["unit_price"], bins=40, ax=axes[0, 0], color=PALETTE[0])
    axes[0, 0].set_title("Unit price distribution")
    axes[0, 0].set_xlabel("Unit price")

    sns.histplot(items["quantity"], discrete=True, ax=axes[0, 1], color=PALETTE[0])
    axes[0, 1].set_title("Quantity per order line")
    axes[0, 1].set_xlabel("Quantity")

    sns.histplot(revenue_orders["order_revenue"], bins=40, kde=True,
                 ax=axes[0, 2], color=PALETTE[0])
    axes[0, 2].set_title("Order value distribution")
    axes[0, 2].set_xlabel("Order value")

    sns.boxplot(data=items, x="category", y="unit_price", ax=axes[1, 0], color=PALETTE[0])
    axes[1, 0].set_title("Unit price by category")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=items, x="category", y="quantity", ax=axes[1, 1], color=PALETTE[0])
    axes[1, 1].set_title("Quantity by category")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=30)

    sns.histplot(np.log10(revenue_orders["order_revenue"]), bins=40, kde=True,
                 ax=axes[1, 2], color=PALETTE[0])
    axes[1, 2].set_title("Order value (log10 scale)")
    axes[1, 2].set_xlabel("log10(order value)")

    return save_figure(fig, "02_distributions")


def theoretical_distributions(datasets):
    rng = np.random.default_rng(RANDOM_SEED)

    ages = datasets["customers"]["age"]

    order_dates = pd.to_datetime(datasets["orders"]["order_date"])
    day_of_month = order_dates.dt.day
    day_of_month = day_of_month[day_of_month <= 28]

    positive = (datasets["reviews"]["rating"] >= 4).to_numpy()
    batch_size = 10
    batches = len(positive) // batch_size
    positives_per_batch = positive[:batches * batch_size].reshape(batches, batch_size).sum(axis=1)
    positive_share = positive.mean()

    print("\nTHEORETICAL DISTRIBUTIONS")
    print("=" * 55)

    # Normal distribution: customer age
    mean_age, std_age = ages.mean(), ages.std()
    normal_sample = rng.normal(mean_age, std_age, size=len(ages))
    _, normal_p = stats.normaltest(ages)
    print(f"Normal: customer age ~ N({mean_age:.1f}, {std_age:.1f}); "
          f"D'Agostino test p-value = {normal_p:.4f}")

    # Uniform distribution: day of month of the orders
    uniform_sample = rng.uniform(1, 29, size=len(day_of_month))
    observed = day_of_month.value_counts().reindex(range(1, 29), fill_value=0)
    _, uniform_p = stats.chisquare(observed)
    print(f"Uniform: order day of month ~ U(1, 28); "
          f"chi-square test p-value = {uniform_p:.4f}")

    # Binomial distribution: positive reviews out of every 10 reviews
    binomial_sample = rng.binomial(batch_size, positive_share, size=batches)
    print(f"Binomial: positive reviews per {batch_size} reviews ~ "
          f"B({batch_size}, {positive_share:.2f}); "
          f"expected {batch_size * positive_share:.2f}, "
          f"observed mean {positives_per_batch.mean():.2f}")

    print("A p-value above 0.05 means the data is consistent with the theoretical "
          "distribution; below 0.05 the deviation is statistically significant.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    x = np.linspace(ages.min(), ages.max(), 200)
    axes[0].hist(ages, bins=20, density=True, alpha=0.55, color=PALETTE[0], label="Customer ages")
    axes[0].hist(normal_sample, bins=20, density=True, histtype="step",
                 color=PALETTE[1], linewidth=1.5, label="NumPy sample")
    axes[0].plot(x, stats.norm.pdf(x, mean_age, std_age), color="#0b0b0b",
                 linewidth=2, label="SciPy normal PDF")
    axes[0].set_title("Normal distribution - customer age")
    axes[0].set_xlabel("Age")
    axes[0].legend()

    axes[1].hist(day_of_month, bins=28, range=(1, 29), density=True, alpha=0.55,
                 color=PALETTE[0], label="Order day of month")
    axes[1].hist(uniform_sample, bins=28, range=(1, 29), density=True, histtype="step",
                 color=PALETTE[1], linewidth=1.5, label="NumPy sample")
    axes[1].plot([1, 29], [stats.uniform.pdf(10, 1, 28)] * 2, color="#0b0b0b",
                 linewidth=2, label="SciPy uniform PDF")
    axes[1].set_title("Uniform distribution - day of month")
    axes[1].set_xlabel("Day of month")
    axes[1].legend()

    k = np.arange(0, batch_size + 1)
    observed_share = np.bincount(positives_per_batch, minlength=batch_size + 1) / batches
    simulated_share = np.bincount(binomial_sample, minlength=batch_size + 1) / batches
    axes[2].bar(k - 0.2, observed_share, width=0.4, color=PALETTE[0], label="Observed reviews")
    axes[2].bar(k + 0.2, simulated_share, width=0.4, color=PALETTE[1], label="NumPy sample")
    axes[2].plot(k, stats.binom.pmf(k, batch_size, positive_share), "o-",
                 color="#0b0b0b", linewidth=2, label="SciPy binomial PMF")
    axes[2].set_title(f"Binomial distribution - positive reviews per {batch_size}")
    axes[2].set_xlabel("Number of positive reviews")
    axes[2].legend()

    return save_figure(fig, "03_theoretical_distributions")


def correlation_strength(value):
    value = abs(value)

    if value >= 0.7:
        return "strong"
    if value >= 0.4:
        return "moderate"
    if value >= 0.2:
        return "weak"
    return "negligible"


def correlation_analysis(items, orders):
    revenue_orders = orders[orders["is_revenue"]]
    numeric_data = revenue_orders[CORRELATION_COLUMNS]

    pearson = numeric_data.corr(method="pearson")
    spearman = numeric_data.corr(method="spearman")

    print("\nCORRELATION ANALYSIS")
    print("=" * 55)

    print("\nPearson correlation:")
    print(pearson.round(2))

    print("\nSpearman correlation:")
    print(spearman.round(2))

    pairs = [
        ("total_quantity", "order_revenue", revenue_orders),
        ("item_count", "order_revenue", revenue_orders),
        ("age", "order_revenue", revenue_orders),
        ("unit_price", "quantity", items),
    ]

    print("\nINTERPRETATION")
    print("-" * 55)

    for x_column, y_column, data in pairs:
        pearson_r, pearson_p = stats.pearsonr(data[x_column], data[y_column])
        spearman_r, spearman_p = stats.spearmanr(data[x_column], data[y_column])

        direction = "positive" if spearman_r > 0 else "negative"
        significant = "significant" if spearman_p < 0.05 else "not significant"

        print(
            f"{x_column} vs {y_column}: Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
            f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f}) -> "
            f"{correlation_strength(spearman_r)} {direction} relationship, {significant}"
        )

    print(
        "\nNotes:\n"
        "- order_revenue, order_cost and order_profit are almost perfectly correlated "
        "because cost and profit are derived from the same order lines.\n"
        "- The difference between Pearson and Spearman shows the effect of skewed "
        "values: Spearman uses ranks and is less sensitive to very large orders.\n"
        "- Correlation describes association only. For example, the negative link "
        "between unit price and quantity does not prove that a higher price causes "
        "customers to buy fewer units; product type (cheap consumables vs. "
        "expensive electronics) explains both."
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for axis, matrix, title in [
        (axes[0], pearson, "Pearson correlation"),
        (axes[1], spearman, "Spearman correlation"),
    ]:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    center=0, square=True, ax=axis, cbar_kws={"shrink": 0.75})
        axis.set_title(title)

    heatmap_path = save_figure(fig, "04_correlation_heatmaps")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.scatterplot(data=revenue_orders, x="total_quantity", y="order_revenue",
                    alpha=0.4, ax=axes[0], color=PALETTE[0])
    axes[0].set_title("Units per order vs order value")

    sns.scatterplot(data=revenue_orders, x="age", y="order_revenue",
                    alpha=0.4, ax=axes[1], color=PALETTE[0])
    axes[1].set_title("Customer age vs order value")

    sns.scatterplot(data=items, x="unit_price", y="quantity", hue="category",
                    alpha=0.5, ax=axes[2], palette=PALETTE[:6])
    axes[2].set_xscale("log")
    axes[2].set_title("Unit price vs quantity")

    scatter_path = save_figure(fig, "05_scatter_plots")

    return pearson, spearman, heatmap_path, scatter_path


def business_analysis(items, orders):
    completed_orders = orders[orders["is_revenue"]]
    sold_items = items[items["is_revenue"]]

    total_revenue = sold_items["revenue"].sum()
    total_cost = sold_items["total_cost"].sum()
    total_profit = total_revenue - total_cost

    average_order_value = completed_orders["order_revenue"].mean()

    profit_margin = (
        total_profit / total_revenue * 100
        if total_revenue else 0
    )

    cancelled = orders[orders["status"] == "Cancelled"]

    print("\nBUSINESS PERFORMANCE")
    print("=" * 55)
    print(f"Paid orders (delivered + shipped): {len(completed_orders):,}")
    print(f"Active customers: {completed_orders['customer_id'].nunique():,}")
    print(f"Revenue: {total_revenue:,.2f}")
    print(f"Cost: {total_cost:,.2f}")
    print(f"Profit: {total_profit:,.2f}")
    print(f"Profit margin: {profit_margin:.2f}%")
    print(f"Average order value: {average_order_value:,.2f}")
    print(f"Cancelled orders: {len(cancelled):,} "
          f"({len(cancelled) / len(orders) * 100:.1f}%), "
          f"lost revenue {cancelled['order_revenue'].sum():,.2f}")

    category_performance = (
        sold_items.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            cost=("total_cost", "sum"),
            profit=("profit", "sum"),
            units=("quantity", "sum"),
        )
        .sort_values("profit", ascending=False)
    )
    category_performance["margin_%"] = (
        category_performance["profit"] / category_performance["revenue"] * 100
    )
    category_performance["revenue_share_%"] = (
        category_performance["revenue"] / total_revenue * 100
    )

    print("\nPERFORMANCE BY PRODUCT CATEGORY")
    print("=" * 55)
    print(category_performance.round(2))

    product_performance = (
        sold_items.groupby(["product_name", "category"])
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum"))
        .sort_values("profit", ascending=False)
    )

    print("\nTOP 5 PRODUCTS BY PROFIT")
    print("=" * 55)
    print(product_performance.head(5).round(2))

    monthly_performance = (
        completed_orders.set_index("order_date")
        .resample("ME")
        .agg(
            revenue=("order_revenue", "sum"),
            profit=("order_profit", "sum"),
            orders=("order_id", "count"),
        )
    )
    monthly_performance["revenue_growth_%"] = (
        monthly_performance["revenue"].pct_change() * 100
    )
    monthly_performance.index = monthly_performance.index.strftime("%Y-%m")

    print("\nMONTHLY PERFORMANCE")
    print("=" * 55)
    print(monthly_performance.round(2))

    customer_behaviour = (
        completed_orders.groupby("customer_category")
        .agg(
            customers=("customer_id", "nunique"),
            orders=("order_id", "count"),
            revenue=("order_revenue", "sum"),
            average_order_value=("order_revenue", "mean"),
        )
    )
    customer_behaviour["orders_per_customer"] = (
        customer_behaviour["orders"] / customer_behaviour["customers"]
    )
    customer_behaviour["revenue_share_%"] = (
        customer_behaviour["revenue"] / total_revenue * 100
    )

    print("\nCUSTOMER BEHAVIOUR BY CUSTOMER CATEGORY")
    print("=" * 55)
    print(customer_behaviour.round(2))

    city_performance = (
        completed_orders.groupby("city")["order_revenue"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "revenue", "count": "orders", "mean": "average_order_value"})
        .sort_values("revenue", ascending=False)
    )

    print("\nREVENUE BY CITY")
    print("=" * 55)
    print(city_performance.round(2))

    payment_methods = (
        completed_orders["payment_method"].value_counts(normalize=True).mul(100)
    )

    print("\nPAYMENT METHODS (% OF PAID ORDERS)")
    print("=" * 55)
    print(payment_methods.round(1))

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "profit_margin": profit_margin,
        "category_performance": category_performance,
        "product_performance": product_performance,
        "monthly_performance": monthly_performance,
        "customer_behaviour": customer_behaviour,
    }


def create_visualizations(results):
    category_performance = results["category_performance"].sort_values("revenue")
    monthly = results["monthly_performance"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    positions = np.arange(len(category_performance))
    axes[0].barh(positions - 0.2, category_performance["revenue"], height=0.4,
                 color=PALETTE[0], label="Revenue")
    axes[0].barh(positions + 0.2, category_performance["profit"], height=0.4,
                 color=PALETTE[1], label="Profit")
    axes[0].set_yticks(positions, category_performance.index)
    axes[0].set_title("Revenue and profit by category")
    axes[0].set_xlabel("Amount")
    axes[0].legend()

    margins = category_performance["margin_%"].sort_values()
    axes[1].barh(margins.index, margins.values, color=PALETTE[0])
    axes[1].set_title("Profit margin by category (%)")
    axes[1].set_xlabel("Margin, %")
    for position, value in enumerate(margins.values):
        axes[1].text(value + 0.5, position, f"{value:.1f}%", va="center")

    category_path = save_figure(fig, "06_category_performance")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(monthly.index, monthly["revenue"], marker="o", color=PALETTE[0],
                 label="Revenue")
    axes[0].plot(monthly.index, monthly["profit"], marker="o", color=PALETTE[1],
                 label="Profit")
    axes[0].set_title("Monthly revenue and profit")
    axes[0].set_ylabel("Amount")
    axes[0].legend()

    growth = monthly["revenue_growth_%"].fillna(0)
    axes[1].bar(monthly.index, growth,
                color=[PALETTE[0] if value >= 0 else PALETTE[7] for value in growth])
    axes[1].axhline(0, color="#52514e", linewidth=1)
    axes[1].set_title("Month-over-month revenue growth (%)")
    axes[1].set_ylabel("Growth, %")
    axes[1].tick_params(axis="x", rotation=45)

    monthly_path = save_figure(fig, "07_monthly_performance")

    return category_path, monthly_path


def print_key_findings(results):
    categories = results["category_performance"]
    monthly = results["monthly_performance"]
    behaviour = results["customer_behaviour"]

    top_profit = categories["profit"].idxmax()
    top_margin = categories["margin_%"].idxmax()
    low_margin = categories["margin_%"].idxmin()

    growth = monthly["revenue_growth_%"].dropna()
    worst_month = growth.idxmin()
    best_month = monthly["revenue"].idxmax()
    weakest_month = monthly["revenue"].idxmin()

    top_customer_group = behaviour["revenue_share_%"].idxmax()

    print("\nKEY FINDINGS")
    print("=" * 55)
    print(f"- {top_profit} generates the highest profit "
          f"({categories.loc[top_profit, 'profit']:,.2f}, "
          f"{categories.loc[top_profit, 'revenue_share_%']:.1f}% of revenue).")
    print(f"- {top_margin} has the highest margin "
          f"({categories.loc[top_margin, 'margin_%']:.1f}%), while {low_margin} has the "
          f"lowest ({categories.loc[low_margin, 'margin_%']:.1f}%).")
    print(f"- The strongest month is {best_month} "
          f"({monthly.loc[best_month, 'revenue']:,.2f}); the weakest is {weakest_month} "
          f"({monthly.loc[weakest_month, 'revenue']:,.2f}).")
    print(f"- The largest month-over-month decline is in {worst_month} "
          f"({growth.loc[worst_month]:+.1f}%).")
    print(f"- {top_customer_group} customers bring "
          f"{behaviour.loc[top_customer_group, 'revenue_share_%']:.1f}% of revenue with "
          f"{behaviour.loc[top_customer_group, 'orders_per_customer']:.2f} orders per customer.")


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    datasets = load_clean_data()

    items, orders = prepare_business_data(datasets)

    descriptive_statistics(items, orders)
    distribution_analysis(items, orders)
    theoretical_distributions(datasets)

    correlation_analysis(items, orders)

    results = business_analysis(items, orders)
    create_visualizations(results)

    print_key_findings(results)


if __name__ == "__main__":
    main()
