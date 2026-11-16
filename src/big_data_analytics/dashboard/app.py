
import pandas as pd
import plotly.express as px
import streamlit as st

from big_data_analytics.config import PROCESSED_DATA_DIR


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Online Shop Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Online Shop — Business Analytics")
st.caption("Sales performance, customer behavior and consumer opinions")


# -------------------- DATA LOADING --------------------

@st.cache_data
def load_data():
    data = {}

    for name in ["customers", "products", "orders", "order_items"]:
        data[name] = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")

    data["orders"]["order_date"] = pd.to_datetime(
        data["orders"]["order_date"],
        errors="coerce",
    )

    data["order_items"] = data["order_items"].merge(
        data["products"][["product_id", "category", "cost"]],
        on="product_id",
        how="left",
    )

    data["order_items"]["revenue"] = data["order_items"]["line_total"]

    data["order_items"]["profit"] = (
        data["order_items"]["line_total"]
        - data["order_items"]["quantity"] * data["order_items"]["cost"]
    )

    return data


try:
    data = load_data()
except FileNotFoundError as error:
    st.error(f"Missing processed dataset: {error.filename}")
    st.stop()


customers = data["customers"].copy()
orders = data["orders"].copy()
items = data["order_items"].copy()

# Only delivered orders contribute to sales KPIs.
orders = orders[
    orders["status"].astype(str).str.lower() == "delivered"
].copy()

# Load customer segmentation results.
segments_path = PROCESSED_DATA_DIR / "customer_segments.csv"

if segments_path.exists():
    segments = pd.read_csv(segments_path)

    customers = customers.merge(
        segments[["customer_id", "segment"]].drop_duplicates("customer_id"),
        on="customer_id",
        how="left",
    )
else:
    customers["segment"] = "Unassigned"

customers["segment"] = customers["segment"].fillna("Unassigned")

# Attach customer information to orders.
orders = orders.merge(
    customers[["customer_id", "segment"]],
    on="customer_id",
    how="left",
)

# Attach order information to order items.
sales = items.merge(
    orders[["order_id", "order_date", "customer_id", "segment"]],
    on="order_id",
    how="inner",
)

sales["segment"] = sales["segment"].fillna("Unassigned")
sales = sales.dropna(subset=["order_date"])

if sales.empty:
    st.warning("No delivered sales were found in the processed data.")
    st.stop()


# -------------------- SIDEBAR FILTERS --------------------

st.sidebar.header("Dashboard Filters")

min_date = sales["order_date"].min().date()
max_date = sales["order_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

categories = sorted(sales["category"].dropna().unique().tolist())

selected_categories = st.sidebar.multiselect(
    "Product category",
    options=categories,
    default=categories,
)

available_segments = sorted(
    sales["segment"].dropna().unique().tolist()
)

selected_segments = st.sidebar.multiselect(
    "Customer segment",
    options=available_segments,
    default=available_segments,
)

if len(selected_dates) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

if start_date > end_date:
    st.warning("The start date must be before the end date.")
    st.stop()

filtered_sales = sales[
    (sales["order_date"].dt.date >= start_date)
    & (sales["order_date"].dt.date <= end_date)
    & (sales["category"].isin(selected_categories))
    & (sales["segment"].isin(selected_segments))
].copy()

if filtered_sales.empty:
    st.warning("No sales match the selected filters.")
    st.stop()


# -------------------- KPI CALCULATIONS --------------------

total_revenue = filtered_sales["revenue"].sum()
total_profit = filtered_sales["profit"].sum()

profit_margin = (
    total_profit / total_revenue * 100
    if total_revenue != 0
    else 0
)

order_revenue = filtered_sales.groupby("order_id")["revenue"].sum()

total_orders = len(order_revenue)
total_customers = filtered_sales["customer_id"].nunique()

average_order_value = (
    order_revenue.mean()
    if total_orders > 0
    else 0
)

# Compare with the immediately preceding period of equal length.
period_days = (end_date - start_date).days + 1

previous_start = pd.Timestamp(start_date) - pd.Timedelta(days=period_days)
previous_end = pd.Timestamp(start_date)

previous_sales = sales[
    (sales["order_date"] >= previous_start)
    & (sales["order_date"] < previous_end)
    & (sales["category"].isin(selected_categories))
    & (sales["segment"].isin(selected_segments))
]

previous_revenue = previous_sales["revenue"].sum()

revenue_growth = (
    (total_revenue - previous_revenue) / previous_revenue * 100
    if previous_revenue > 0
    else None
)


# -------------------- KPI DISPLAY --------------------

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{total_revenue:,.2f}")
col2.metric("Total Profit", f"{total_profit:,.2f}")
col3.metric("Profit Margin", f"{profit_margin:.2f}%")

col4, col5, col6 = st.columns(3)

col4.metric("Orders", f"{total_orders:,}")
col5.metric("Customers", f"{total_customers:,}")
col6.metric("Average Order Value", f"{average_order_value:,.2f}")

if revenue_growth is not None:
    st.metric(
        "Revenue Growth vs. Previous Period",
        f"{revenue_growth:+.2f}%",
    )
else:
    st.metric(
        "Revenue Growth vs. Previous Period",
        "Unavailable",
        help="There is no previous-period revenue available for comparison.",
    )


# -------------------- SALES PERFORMANCE --------------------

st.divider()
st.subheader("Sales Performance")

daily_revenue = (
    filtered_sales.groupby(filtered_sales["order_date"].dt.date)["revenue"]
    .sum()
    .reset_index()
)

daily_revenue.columns = ["date", "revenue"]

fig_revenue = px.line(
    daily_revenue,
    x="date",
    y="revenue",
    title="Daily Revenue",
    markers=True,
    labels={
        "date": "Date",
        "revenue": "Revenue",
    },
)

st.plotly_chart(fig_revenue, use_container_width=True)


category_revenue = (
    filtered_sales.groupby("category")["revenue"]
    .sum()
    .reset_index()
    .sort_values("revenue", ascending=False)
)

fig_category = px.bar(
    category_revenue,
    x="category",
    y="revenue",
    title="Revenue by Product Category",
    text_auto=".2s",
    labels={
        "category": "Product Category",
        "revenue": "Revenue",
    },
)

st.plotly_chart(fig_category, use_container_width=True)


# -------------------- CUSTOMER SEGMENTATION --------------------

st.divider()
st.subheader("Customer Segmentation")

segment_distribution = (
    filtered_sales.groupby("segment")["customer_id"]
    .nunique()
    .reset_index(name="customers")
    .sort_values("customers", ascending=False)
)

fig_segments = px.pie(
    segment_distribution,
    names="segment",
    values="customers",
    title="Customer Distribution by Segment",
    hole=0.4,
)

st.plotly_chart(fig_segments, use_container_width=True)

st.caption(
    "Segment distribution includes customers with delivered purchases "
    "within the selected date range and filters."
)


# -------------------- CUSTOMER SENTIMENT --------------------

st.divider()
st.subheader("Customer Review Sentiment")

sentiment_path = PROCESSED_DATA_DIR / "reviews_analyzed.csv"

if sentiment_path.exists():
    reviews = pd.read_csv(sentiment_path)

    if "sentiment" in reviews.columns:
        sentiment_counts = (
            reviews["sentiment"]
            .value_counts()
            .rename_axis("sentiment")
            .reset_index(name="reviews")
        )

        fig_sentiment = px.pie(
            sentiment_counts,
            names="sentiment",
            values="reviews",
            title="Review Sentiment Distribution",
            hole=0.4,
            color="sentiment",
            color_discrete_map={
                "Positive": "#80c4f8",
                "Neutral": "#0866c6",
                "Negative": "#f5a3a8",
            },
        )

        st.plotly_chart(fig_sentiment, use_container_width=True)

        st.caption(
            "Sentiment represents all analyzed reviews and is not affected "
            "by the sales dashboard filters."
        )
    else:
        st.info("The sentiment column was not found in reviews_analyzed.csv.")
else:
    st.info("Run the sentiment analysis script to display review sentiment.")


# -------------------- SALES FORECAST --------------------

st.divider()
st.subheader("Sales Forecast")

forecast_path = PROCESSED_DATA_DIR / "lstm_forecast.csv"

if forecast_path.exists():
    forecast = pd.read_csv(forecast_path)

    date_column = next(
        (column for column in forecast.columns if "date" in column.lower()),
        None,
    )

    forecast_column = next(
        (
            column for column in forecast.columns
            if "forecast" in column.lower()
        ),
        None,
    )

    if date_column and forecast_column:
        forecast[date_column] = pd.to_datetime(
            forecast[date_column],
            errors="coerce",
        )

        forecast = forecast.dropna(
            subset=[date_column, forecast_column]
        )

        fig_forecast = px.line(
            forecast,
            x=date_column,
            y=forecast_column,
            title="LSTM Revenue Forecast — Next 30 Days",
            markers=True,
            labels={
                date_column: "Date",
                forecast_column: "Forecasted Revenue",
            },
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        st.caption(
            "This chart shows the future forecast. It is not historical "
            "model validation. Historical actual-versus-predicted results "
            "are evaluated separately in the forecasting module."
        )
    else:
        st.warning(
            "Could not identify the date and forecast columns "
            "in lstm_forecast.csv."
        )
else:
    st.info("Run the LSTM forecasting script to display the forecast.")


# -------------------- FILTERED SALES DATA --------------------

st.divider()

with st.expander("View filtered sales data"):
    st.dataframe(
        filtered_sales[
            [
                "order_id",
                "order_date",
                "customer_id",
                "category",
                "quantity",
                "revenue",
                "profit",
                "segment",
            ]
        ],
        use_container_width=True,
    )