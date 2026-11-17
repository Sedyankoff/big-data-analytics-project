import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from big_data_analytics.config import (
    FORECAST_HORIZON,
    PROCESSED_DATA_DIR,
    REVENUE_STATUSES,
)
from big_data_analytics.plotting import PALETTE, SENTIMENT_COLORS


SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
MODEL_ORDER = ["Naive", "Moving Average", "Linear Regression", "LSTM"]
MODEL_COLORS = dict(zip(MODEL_ORDER, [PALETTE[3], PALETTE[2], PALETTE[1], PALETTE[6]]))
ACTUAL_COLOR = "#52514e"
POSITIVE_COLOR = PALETTE[0]
NEGATIVE_COLOR = PALETTE[7]


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Online Shop Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- DATA LOADING --------------------

def read_optional(name, **kwargs):
    path = PROCESSED_DATA_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


@st.cache_data
def load_data():
    data = {}

    for name in ["customers", "products", "orders", "order_items"]:
        data[name] = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")

    data["orders"]["order_date"] = pd.to_datetime(data["orders"]["order_date"])

    data["segments"] = read_optional("customer_segments.csv")
    data["reviews"] = read_optional("reviews_analyzed.csv", parse_dates=["review_date"])
    data["metrics"] = read_optional("forecast_metrics.csv", index_col="model")
    data["test_predictions"] = read_optional(
        "forecast_test_predictions.csv", parse_dates=["date"]
    )
    data["lstm_forecast"] = read_optional("lstm_forecast.csv", parse_dates=["date"])
    data["baseline_forecast"] = read_optional("baseline_forecast.csv", parse_dates=["date"])

    return data


def build_sales(data):
    customers = data["customers"][["customer_id", "customer_category"]].copy()

    if data["segments"] is not None:
        customers = customers.merge(
            data["segments"][["customer_id", "segment"]],
            on="customer_id",
            how="left",
        )
    else:
        customers["segment"] = "Unassigned"

    customers["segment"] = customers["segment"].fillna("Unassigned")

    orders = data["orders"][data["orders"]["status"].isin(REVENUE_STATUSES)]

    sales = (
        data["order_items"]
        .merge(
            data["products"][["product_id", "product_name", "category", "cost"]],
            on="product_id",
            how="left",
        )
        .merge(
            orders[["order_id", "order_date", "customer_id"]],
            on="order_id",
            how="inner",
        )
        .merge(customers, on="customer_id", how="left")
    )

    sales["revenue"] = sales["line_total"]
    sales["total_cost"] = sales["quantity"] * sales["cost"]
    sales["profit"] = sales["revenue"] - sales["total_cost"]

    return sales, customers


def style_figure(fig, height=360, legend=True, x_title=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        font=dict(size=13),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title_text="",
        ),
    )

    has_lines = any(
        trace.type == "scatter" and "lines" in (trace.mode or "")
        for trace in fig.data
    )
    fig.update_layout(hovermode="x unified" if has_lines else "closest")
    fig.update_xaxes(showgrid=False)
    if not x_title:
        fig.update_xaxes(title_text="")
    fig.update_yaxes(zeroline=False)

    return fig


def chart(fig, height=360, legend=True, x_title=False):
    st.plotly_chart(style_figure(fig, height, legend, x_title), width="stretch")


def card_title(title, caption=None):
    st.markdown(f"**{title}**")
    if caption:
        st.caption(caption)


def format_money(value):
    return f"{value:,.0f}"


try:
    data = load_data()
except FileNotFoundError as error:
    st.error(
        f"Missing processed dataset: {error.filename}. "
        "Run the data pipeline first (see README)."
    )
    st.stop()

sales, customers = build_sales(data)

if sales.empty:
    st.warning("No paid sales were found in the processed data.")
    st.stop()

categories = sorted(sales["category"].dropna().unique())
CATEGORY_COLORS = dict(zip(categories, PALETTE))

if data["segments"] is not None:
    segment_order = (
        data["segments"].groupby("segment")["total_spending"].mean()
        .sort_values(ascending=False).index.tolist()
    )
else:
    segment_order = ["Unassigned"]
SEGMENT_COLORS = dict(zip(segment_order, PALETTE))


# -------------------- SIDEBAR FILTERS --------------------

st.sidebar.header("Filters")

min_date = sales["order_date"].min().date()
max_date = sales["order_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

selected_categories = st.sidebar.multiselect(
    "Product category",
    options=categories,
    placeholder="All categories",
)

selected_segments = st.sidebar.multiselect(
    "Customer segment",
    options=segment_order,
    placeholder="All segments",
)

st.sidebar.caption(
    "Leave a filter empty to include all values. Filters apply to KPIs, sales, "
    "products, customers and sentiment. The forecast covers the whole shop."
)

if len(selected_dates) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates
category_filter = selected_categories or categories
segment_filter = selected_segments or segment_order


def apply_filters(dataframe, date_column, start, end):
    return dataframe[
        (dataframe[date_column].dt.date >= start)
        & (dataframe[date_column].dt.date <= end)
        & dataframe["category"].isin(category_filter)
        & dataframe["segment"].isin(segment_filter)
    ]


filtered_sales = apply_filters(sales, "order_date", start_date, end_date).copy()

reviews = None
if data["reviews"] is not None:
    reviews = (
        data["reviews"]
        .merge(data["products"][["product_id", "product_name", "category"]], on="product_id")
        .merge(customers[["customer_id", "segment"]], on="customer_id", how="left")
    )
    reviews = apply_filters(reviews, "review_date", start_date, end_date)


# -------------------- HEADER --------------------

st.title("Online Shop — Business Analytics")
st.caption(
    f"Sales, customer behaviour and customer opinions · "
    f"{start_date:%d %b %Y} – {end_date:%d %b %Y} · "
    f"{len(category_filter)} of {len(categories)} categories · "
    f"{len(segment_filter)} of {len(segment_order)} segments"
)

if filtered_sales.empty:
    st.warning("No sales match the selected filters.")
    st.stop()


# -------------------- KPI CALCULATIONS --------------------

total_revenue = filtered_sales["revenue"].sum()
total_profit = filtered_sales["profit"].sum()
profit_margin = total_profit / total_revenue * 100 if total_revenue else 0

total_orders = filtered_sales["order_id"].nunique()
total_customers = filtered_sales["customer_id"].nunique()
average_order_value = total_revenue / total_orders if total_orders else 0


def revenue_growth():
    """Compare with the preceding period of equal length when it exists in the data,
    otherwise compare the last two complete months inside the selected period."""
    period_days = (end_date - start_date).days + 1
    previous_end = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=period_days - 1)

    if previous_start.date() >= min_date:
        previous = apply_filters(sales, "order_date", previous_start.date(), previous_end.date())
        previous_revenue = previous["revenue"].sum()

        if previous_revenue > 0:
            return (
                (total_revenue / previous_revenue - 1) * 100,
                f"vs previous {period_days} days",
            )

    months = pd.period_range(start_date, end_date, freq="M")
    complete_months = [
        month for month in months
        if month.start_time.date() >= start_date and month.end_time.date() <= end_date
    ]

    if len(complete_months) >= 2:
        monthly = filtered_sales.groupby(filtered_sales["order_date"].dt.to_period("M"))[
            "revenue"
        ].sum()
        last, previous = complete_months[-1], complete_months[-2]

        if monthly.get(previous, 0) > 0:
            return (
                (monthly.get(last, 0) / monthly[previous] - 1) * 100,
                f"{last.strftime('%b')} vs {previous.strftime('%b %Y')}",
            )

    return None, "no comparable period"


growth, growth_label = revenue_growth()

positive_share = None
average_rating = None
if reviews is not None and not reviews.empty:
    positive_share = (reviews["sentiment"] == "Positive").mean() * 100
    average_rating = reviews["rating"].mean()


# -------------------- KPI DISPLAY --------------------

row1 = st.columns(4)
row1[0].metric("Total revenue", format_money(total_revenue), border=True)
row1[1].metric("Total profit", format_money(total_profit), border=True)
row1[2].metric("Profit margin", f"{profit_margin:.1f}%", border=True)
row1[3].metric("Average order value", f"{average_order_value:,.2f}", border=True)

row2 = st.columns(4)
row2[0].metric("Orders", f"{total_orders:,}", border=True)
row2[1].metric("Customers", f"{total_customers:,}", border=True,
               help="Customers with at least one paid order in the selection.")
row2[2].metric(
    "Revenue growth",
    f"{growth:+.1f}%" if growth is not None else "n/a",
    delta=growth_label,
    delta_color="off",
    delta_arrow="off",
    border=True,
    help="Percentage change of revenue against a comparable period.",
)
row2[3].metric(
    "Positive reviews",
    f"{positive_share:.1f}%" if positive_share is not None else "n/a",
    delta=f"average rating {average_rating:.2f} / 5" if average_rating else None,
    delta_color="off",
    delta_arrow="off",
    border=True,
)

st.write("")

tab_sales, tab_products, tab_customers, tab_sentiment, tab_forecast, tab_insights = st.tabs(
    ["Sales", "Products", "Customers", "Sentiment", "Forecast", "Insights"]
)


# -------------------- SALES --------------------

with tab_sales:
    granularity = st.segmented_control(
        "Time granularity",
        options=["Daily", "Weekly", "Monthly"],
        default="Weekly",
        required=True,
    )
    frequency = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[granularity]

    time_series = (
        filtered_sales.set_index("order_date")
        .resample(frequency, label="left", closed="left")
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
        .reset_index()
    )

    left, right = st.columns([3, 2])

    with left, st.container(border=True):
        card_title(f"{granularity} revenue and profit",
                   "The first and last periods can be partial (e.g. 29-31 Dec).")
        fig = go.Figure()
        fig.add_scatter(x=time_series["order_date"], y=time_series["revenue"], name="Revenue",
                        mode="lines", line=dict(color=PALETTE[0], width=2))
        fig.add_scatter(x=time_series["order_date"], y=time_series["profit"], name="Profit",
                        mode="lines", line=dict(color=PALETTE[1], width=2))
        fig.update_yaxes(tickformat=",.0f")
        chart(fig)

    with right, st.container(border=True):
        card_title(f"{granularity} number of orders", "Paid orders per period.")
        fig = px.bar(time_series, x="order_date", y="orders",
                     color_discrete_sequence=[PALETTE[0]],
                     labels={"orders": "Orders", "order_date": "Period"})
        chart(fig, legend=False)

    monthly = (
        filtered_sales.groupby(filtered_sales["order_date"].dt.to_period("M"))
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
    )
    monthly["growth"] = monthly["revenue"].pct_change() * 100
    monthly.index = monthly.index.to_timestamp()

    left, right = st.columns([3, 2])

    with left, st.container(border=True):
        card_title("Month-over-month revenue growth",
                   "Negative bars show the months in which sales declined.")
        growth_data = monthly["growth"].dropna().reset_index()
        growth_data.columns = ["month", "growth"]
        fig = go.Figure(go.Bar(
            x=growth_data["month"],
            y=growth_data["growth"],
            marker_color=[POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
                          for value in growth_data["growth"]],
            hovertemplate="%{x|%b %Y}: %{y:+.1f}%<extra></extra>",
        ))
        fig.update_yaxes(ticksuffix="%")
        fig.update_xaxes(tickformat="%b")
        chart(fig, legend=False)

    with right, st.container(border=True):
        card_title("Monthly summary")
        table = monthly.reset_index().rename(columns={"order_date": "Month"})
        st.dataframe(
            table,
            hide_index=True,
            height=325,
            column_config={
                "Month": st.column_config.DateColumn(format="MMM YYYY"),
                "revenue": st.column_config.NumberColumn("Revenue", format="%.0f"),
                "profit": st.column_config.NumberColumn("Profit", format="%.0f"),
                "orders": st.column_config.NumberColumn("Orders"),
                "growth": st.column_config.NumberColumn("Growth", format="%+.1f%%"),
            },
        )


# -------------------- PRODUCTS --------------------

category_performance = (
    filtered_sales.groupby("category")
    .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum"))
    .sort_values("revenue")
)
category_performance["margin"] = (
    category_performance["profit"] / category_performance["revenue"] * 100
)

with tab_products:
    left, right = st.columns(2)

    with left, st.container(border=True):
        card_title("Revenue and profit by category")
        fig = go.Figure()
        fig.add_bar(y=category_performance.index, x=category_performance["revenue"],
                    name="Revenue", orientation="h", marker_color=PALETTE[0])
        fig.add_bar(y=category_performance.index, x=category_performance["profit"],
                    name="Profit", orientation="h", marker_color=PALETTE[1])
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
        fig.update_xaxes(tickformat=",.0f")
        chart(fig, height=380)

    with right, st.container(border=True):
        card_title("Profit margin by category")
        margins = category_performance.sort_values("margin")
        fig = go.Figure(go.Bar(
            y=margins.index,
            x=margins["margin"],
            orientation="h",
            marker_color=PALETTE[0],
            text=[f"{value:.1f}%" for value in margins["margin"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(ticksuffix="%", range=[0, margins["margin"].max() * 1.18])
        chart(fig, height=380, legend=False)

    with st.container(border=True):
        card_title("Top 10 products by revenue")
        top_products = (
            filtered_sales.groupby(["product_name", "category"])
            .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum"))
            .reset_index()
            .nlargest(10, "revenue")
            .sort_values("revenue")
        )
        fig = px.bar(
            top_products,
            x="revenue",
            y="product_name",
            color="category",
            orientation="h",
            color_discrete_map=CATEGORY_COLORS,
            category_orders={"category": categories},
            hover_data={"profit": ":,.0f", "units": True, "revenue": ":,.0f"},
            labels={"product_name": "Product", "revenue": "Revenue", "category": "Category"},
        )
        fig.update_yaxes(categoryorder="total ascending", title_text="")
        chart(fig, height=420)


# -------------------- CUSTOMERS --------------------

with tab_customers:
    if data["segments"] is None:
        st.info("Run the customer segmentation script to display customer segments.")
    else:
        segment_customers = data["segments"][data["segments"]["segment"].isin(segment_filter)]

        segment_revenue = (
            filtered_sales.groupby("segment")
            .agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"))
            .reindex(segment_filter)
            .dropna()
            .reset_index()
        )

        left, right = st.columns(2)

        with left, st.container(border=True):
            card_title("Customer distribution by segment",
                       "All registered customers, segmented with K-means.")
            distribution = (
                segment_customers["segment"].value_counts()
                .reindex(segment_filter).dropna().reset_index()
            )
            distribution.columns = ["segment", "customers"]
            distribution["share"] = distribution["customers"] / distribution["customers"].sum()
            fig = px.bar(
                distribution, x="customers", y="segment", color="segment", orientation="h",
                color_discrete_map=SEGMENT_COLORS,
                text=[f"{value:.0%}" for value in distribution["share"]],
                labels={"customers": "Customers", "segment": ""},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_yaxes(categoryorder="array", categoryarray=segment_filter[::-1])
            fig.update_xaxes(range=[0, distribution["customers"].max() * 1.15])
            chart(fig, height=300, legend=False)

        with right, st.container(border=True):
            card_title("Revenue by segment", "Paid orders within the selected filters.")
            segment_revenue["share"] = segment_revenue["revenue"] / segment_revenue["revenue"].sum()
            fig = px.bar(
                segment_revenue, x="revenue", y="segment", color="segment", orientation="h",
                color_discrete_map=SEGMENT_COLORS,
                text=[f"{value:.0%}" for value in segment_revenue["share"]],
                labels={"revenue": "Revenue", "segment": ""},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_yaxes(categoryorder="array", categoryarray=segment_filter[::-1])
            fig.update_xaxes(tickformat=",.0f", range=[0, segment_revenue["revenue"].max() * 1.15])
            chart(fig, height=300, legend=False)

        left, right = st.columns(2)

        with left, st.container(border=True):
            card_title("Segment profiles", "Average values per customer.")
            profile = (
                segment_customers.groupby("segment")
                .agg(
                    customers=("customer_id", "count"),
                    orders=("purchase_frequency", "mean"),
                    spending=("total_spending", "mean"),
                    order_value=("average_order_value", "mean"),
                    recency=("recency_days", "mean"),
                )
                .reindex(segment_filter)
                .dropna()
                .reset_index()
            )
            st.dataframe(
                profile,
                hide_index=True,
                column_config={
                    "segment": "Segment",
                    "customers": "Customers",
                    "orders": st.column_config.NumberColumn("Orders", format="%.1f"),
                    "spending": st.column_config.NumberColumn("Spending", format="%.0f"),
                    "order_value": st.column_config.NumberColumn("Order value", format="%.0f"),
                    "recency": st.column_config.NumberColumn(
                        "Recency", format="%.0f",
                        help="Days between the last paid order and the end of the period.",
                    ),
                },
            )
            st.caption(
                "High-Value: frequent, recent, highest spending. Big-Basket: few but large "
                "orders. Occasional: average activity. Inactive: no or very old purchases."
            )

        with right, st.container(border=True):
            card_title("Purchase frequency vs total spending")
            fig = px.scatter(
                segment_customers,
                x="purchase_frequency",
                y="total_spending",
                color="segment",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"segment": segment_order},
                opacity=0.7,
                hover_data={"customer_id": True, "average_order_value": ":,.0f"},
                labels={"purchase_frequency": "Orders", "total_spending": "Total spending",
                        "segment": "Segment"},
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
            chart(fig, height=330, x_title=True)


# -------------------- SENTIMENT --------------------

with tab_sentiment:
    if reviews is None:
        st.info("Run the sentiment analysis script to display review sentiment.")
    elif reviews.empty:
        st.info("No reviews match the selected filters.")
    else:
        left, right = st.columns([2, 3])

        with left, st.container(border=True):
            card_title("Sentiment of customer reviews", f"{len(reviews):,} reviews")
            sentiment_counts = (
                reviews["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
                .reset_index()
            )
            sentiment_counts.columns = ["sentiment", "reviews"]
            sentiment_counts["share"] = sentiment_counts["reviews"] / len(reviews)
            fig = px.bar(
                sentiment_counts, x="sentiment", y="reviews", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                text=[f"{value:.0%}" for value in sentiment_counts["share"]],
                labels={"reviews": "Reviews", "sentiment": ""},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            chart(fig, height=340, legend=False)

        with right, st.container(border=True):
            card_title("Sentiment share over time")
            monthly_sentiment = (
                pd.crosstab(reviews["review_date"].dt.to_period("M").dt.to_timestamp(),
                            reviews["sentiment"], normalize="index")
                .reindex(columns=SENTIMENT_ORDER, fill_value=0)
                .mul(100)
                .reset_index()
                .melt(id_vars="review_date", var_name="sentiment", value_name="share")
            )
            fig = px.bar(
                monthly_sentiment, x="review_date", y="share", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                category_orders={"sentiment": SENTIMENT_ORDER},
                labels={"share": "Share of reviews", "review_date": "Month"},
            )
            fig.update_layout(barmode="stack", bargap=0.15)
            fig.update_traces(marker_line_width=1, marker_line_color="white",
                              hovertemplate="%{x|%b %Y}: %{y:.1f}%")
            fig.update_yaxes(ticksuffix="%", range=[0, 100])
            fig.update_xaxes(tickformat="%b", dtick="M1")
            chart(fig, height=340)

        left, right = st.columns([2, 3])

        with left, st.container(border=True):
            card_title("Average rating by category")
            ratings = (
                reviews.groupby("category")
                .agg(rating=("rating", "mean"), reviews=("review_id", "count"))
                .sort_values("rating")
                .reset_index()
            )
            fig = go.Figure(go.Bar(
                y=ratings["category"],
                x=ratings["rating"],
                orientation="h",
                marker_color=PALETTE[0],
                text=[f"{value:.2f}" for value in ratings["rating"]],
                textposition="outside",
                cliponaxis=False,
                customdata=ratings["reviews"],
                hovertemplate="%{y}: %{x:.2f} (%{customdata} reviews)<extra></extra>",
            ))
            fig.update_xaxes(range=[1, 5.4])
            chart(fig, height=320, legend=False)

        with right, st.container(border=True):
            card_title("Latest negative reviews")
            negative = (
                reviews[reviews["sentiment"] == "Negative"]
                .sort_values("review_date", ascending=False)
                [["review_date", "product_name", "category", "rating", "review_text"]]
                .head(20)
            )
            st.dataframe(
                negative,
                hide_index=True,
                height=320,
                column_config={
                    "review_date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
                    "product_name": "Product",
                    "category": "Category",
                    "rating": st.column_config.NumberColumn("Rating", format="%d ★"),
                    "review_text": st.column_config.TextColumn("Review", width="large"),
                },
            )

        st.caption(
            "Sentiment is assigned by a rule-based Bulgarian lexicon with negation handling. "
            "It cannot interpret sarcasm and assigns one label to mixed reviews."
        )


# -------------------- FORECAST --------------------

with tab_forecast:
    if data["test_predictions"] is None or data["metrics"] is None:
        st.info("Run the forecasting and LSTM scripts to display the forecast.")
    else:
        metrics = data["metrics"].reindex([m for m in MODEL_ORDER if m in data["metrics"].index])
        predictions = data["test_predictions"]
        best_model = metrics["RMSE"].idxmin()

        daily_actual = (
            sales.groupby("order_date")["revenue"].sum()
            .reindex(pd.date_range(min_date, max_date, freq="D"), fill_value=0)
        )

        forecasts = {}
        if data["lstm_forecast"] is not None:
            forecasts["LSTM"] = data["lstm_forecast"]
        if data["baseline_forecast"] is not None:
            baseline_name = data["baseline_forecast"]["model"].iloc[0]
            forecasts[baseline_name] = data["baseline_forecast"]

        last_period = daily_actual.iloc[-FORECAST_HORIZON:].sum()

        columns = st.columns(len(forecasts) + 2)
        columns[0].metric(f"Actual · last {FORECAST_HORIZON} days",
                          format_money(last_period), border=True,
                          help="Actual revenue in the last 30 days of the data.")
        for column, (model_name, forecast) in zip(columns[1:], forecasts.items()):
            projected = forecast["forecasted_revenue"].sum()
            change = (projected / last_period - 1) * 100
            column.metric(
                f"{model_name} · next {FORECAST_HORIZON} days",
                format_money(projected),
                delta=f"{change:+.1f}% vs last {FORECAST_HORIZON} days",
                border=True,
            )
        columns[-1].metric("Lowest test RMSE", best_model, border=True)

        with st.container(border=True):
            card_title(
                "Actual and forecasted daily revenue",
                "Grey: actual revenue. Solid lines: one-step-ahead predictions on the test "
                f"period. Dashed lines: forecast for the next {FORECAST_HORIZON} days.",
            )
            default_models = [m for m in ["LSTM", best_model] if m in metrics.index]
            shown_models = st.multiselect(
                "Models",
                options=list(metrics.index),
                default=list(dict.fromkeys(default_models)),
                label_visibility="collapsed",
            )

            window_start = predictions["date"].min() - pd.Timedelta(days=60)
            history = daily_actual[daily_actual.index >= window_start]

            fig = go.Figure()
            fig.add_scatter(x=history.index, y=history.values, name="Actual",
                            mode="lines", line=dict(color=ACTUAL_COLOR, width=1.5))

            for model_name in shown_models:
                fig.add_scatter(x=predictions["date"], y=predictions[model_name],
                                name=model_name, mode="lines",
                                line=dict(color=MODEL_COLORS[model_name], width=2))

                if model_name in forecasts:
                    future = forecasts[model_name]
                    fig.add_scatter(x=future["date"], y=future["forecasted_revenue"],
                                    name=f"{model_name} forecast", mode="lines",
                                    line=dict(color=MODEL_COLORS[model_name], width=2,
                                              dash="dash"))

            fig.add_vline(x=predictions["date"].min(), line_width=1, line_color="#c3c2b7")
            fig.add_vline(x=daily_actual.index[-1] + pd.Timedelta(hours=12),
                          line_width=1, line_color="#c3c2b7")
            fig.update_yaxes(tickformat=",.0f")
            chart(fig, height=420)

        left, right = st.columns([3, 2])

        with left, st.container(border=True):
            card_title("Model evaluation on the test period",
                       "Chronological split: first 80% of days for training, last 20% "
                       "for testing. Lower values are better.")
            st.dataframe(
                metrics.reset_index(),
                hide_index=True,
                column_config={
                    "model": "Model",
                    "MAE": st.column_config.NumberColumn(format="%.2f"),
                    "MSE": st.column_config.NumberColumn(format="%.0f"),
                    "RMSE": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        with right, st.container(border=True):
            card_title("RMSE by model")
            rmse = metrics["RMSE"].sort_values(ascending=False).reset_index()
            fig = go.Figure(go.Bar(
                y=rmse["model"],
                x=rmse["RMSE"],
                orientation="h",
                marker_color=[MODEL_COLORS[name] for name in rmse["model"]],
                text=[f"{value:,.0f}" for value in rmse["RMSE"]],
                textposition="outside",
                cliponaxis=False,
            ))
            fig.update_xaxes(range=[0, rmse["RMSE"].max() * 1.2])
            chart(fig, height=230, legend=False)


# -------------------- INSIGHTS --------------------

with tab_insights:
    findings = []
    recommendations = []

    if len(category_performance) > 1:
        top_profit = category_performance["profit"].idxmax()
        top_margin = category_performance["margin"].idxmax()
        low_margin = category_performance["margin"].idxmin()
        findings.append(
            f"**{top_profit}** brings the highest profit "
            f"({format_money(category_performance.loc[top_profit, 'profit'])}), and "
            f"**{top_margin}** has the highest margin "
            f"({category_performance.loc[top_margin, 'margin']:.1f}%). "
            f"**{low_margin}** has the lowest margin "
            f"({category_performance.loc[low_margin, 'margin']:.1f}%)."
        )
        recommendations.append(
            f"Promote high-margin {top_margin.lower()} products and review pricing or "
            f"supplier costs in {low_margin.lower()}, where each sale earns the least."
        )

    monthly_growth = monthly["growth"].dropna()
    if len(monthly) > 1 and not monthly_growth.empty:
        best_month = monthly["revenue"].idxmax()
        worst_month = monthly["revenue"].idxmin()
        decline_month = monthly_growth.idxmin()
        findings.append(
            f"Revenue peaks in **{best_month:%B}** "
            f"({format_money(monthly.loc[best_month, 'revenue'])}) and is lowest in "
            f"**{worst_month:%B}** ({format_money(monthly.loc[worst_month, 'revenue'])}). "
            f"The largest monthly drop is in **{decline_month:%B}** "
            f"({monthly_growth.loc[decline_month]:+.1f}%)."
        )
        recommendations.append(
            f"Plan stock and staff for the {best_month:%B} peak and run campaigns before "
            f"{worst_month:%B} and {decline_month:%B} to soften the seasonal decline."
        )

    if data["segments"] is not None and not segment_revenue.empty:
        segment_sizes = data["segments"]["segment"].value_counts(normalize=True) * 100
        revenue_shares = segment_revenue.set_index("segment")["share"] * 100
        top_segment = (revenue_shares / segment_sizes.reindex(revenue_shares.index)).idxmax()
        findings.append(
            f"The **{top_segment}** segment represents {segment_sizes[top_segment]:.1f}% of "
            f"all customers but generates {revenue_shares[top_segment]:.1f}% of the revenue "
            "in the selection."
        )
        if "Inactive" in segment_sizes:
            recommendations.append(
                f"{segment_sizes['Inactive']:.1f}% of customers are inactive: a "
                "re-activation campaign (welcome discount, reminder e-mails) is the "
                "cheapest source of extra revenue. Reward High-Value customers with a "
                "loyalty programme to keep them."
            )

    if reviews is not None and not reviews.empty:
        monthly_positive = (
            reviews.groupby(reviews["review_date"].dt.to_period("M"))["sentiment"]
            .apply(lambda values: (values == "Positive").mean() * 100)
        )
        monthly_positive.index = monthly_positive.index.to_timestamp()
        worst_category = reviews.groupby("category")["rating"].mean().idxmin()
        findings.append(
            f"{positive_share:.1f}% of the reviews are positive. Satisfaction is lowest in "
            f"**{monthly_positive.idxmin():%B}** ({monthly_positive.min():.1f}% positive) and "
            f"the lowest-rated category is **{worst_category}**."
        )
        recommendations.append(
            f"Investigate delivery capacity during {monthly_positive.idxmin():%B} and the "
            f"quality of the lowest-rated {worst_category.lower()} products, as negative "
            "reviews mention delays and damaged items."
        )

    if data["metrics"] is not None and data["lstm_forecast"] is not None:
        lstm_total = data["lstm_forecast"]["forecasted_revenue"].sum()
        last_30 = sales.groupby("order_date")["revenue"].sum().iloc[-FORECAST_HORIZON:].sum()
        best = data["metrics"]["RMSE"].idxmin()
        findings.append(
            f"The LSTM projects {format_money(lstm_total)} revenue for the next "
            f"{FORECAST_HORIZON} days ({(lstm_total / last_30 - 1) * 100:+.1f}% vs the last "
            f"{FORECAST_HORIZON} days). The most accurate model on the test period is "
            f"**{best}**."
        )

    left, right = st.columns(2)

    with left, st.container(border=True):
        card_title("Findings", "Calculated from the data in the current selection.")
        st.markdown("\n".join(f"- {finding}" for finding in findings))

    with right, st.container(border=True):
        card_title("Recommendations")
        st.markdown("\n".join(f"- {recommendation}" for recommendation in recommendations))


# -------------------- FILTERED SALES DATA --------------------

with st.expander("View filtered sales data"):
    st.dataframe(
        filtered_sales[
            [
                "order_id",
                "order_date",
                "customer_id",
                "segment",
                "product_name",
                "category",
                "quantity",
                "revenue",
                "profit",
            ]
        ].sort_values("order_date"),
        hide_index=True,
        column_config={
            "order_date": st.column_config.DateColumn("Order date", format="DD.MM.YYYY"),
            "revenue": st.column_config.NumberColumn("Revenue", format="%.2f"),
            "profit": st.column_config.NumberColumn("Profit", format="%.2f"),
        },
    )
