import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from big_data_analytics.config import (
    DATA_END_DATE,
    DATA_START_DATE,
    FORECAST_HORIZON,
    PROCESSED_DATA_DIR,
    REVENUE_STATUSES,
    TEST_SIZE,
)
from big_data_analytics.plotting import PALETTE, save_figure


MOVING_AVERAGE_WINDOW = 7

MODEL_COLORS = {
    "Naive": PALETTE[3],
    "Moving Average": PALETTE[2],
    "Linear Regression": PALETTE[1],
    "LSTM": PALETTE[6],
}


def load_daily_sales():
    orders = pd.read_csv(PROCESSED_DATA_DIR / "orders.csv")
    items = pd.read_csv(PROCESSED_DATA_DIR / "order_items.csv")

    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # Cancelled orders do not generate revenue
    paid_orders = orders[orders["status"].isin(REVENUE_STATUSES)]

    sales = items.merge(
        paid_orders[["order_id", "order_date"]],
        on="order_id",
        how="inner",
    )

    daily = sales.groupby("order_date").agg(
        revenue=("line_total", "sum"),
        orders=("order_id", "nunique"),
    )

    # Days without sales become explicit zero values
    date_range = pd.date_range(
        start=DATA_START_DATE,
        end=DATA_END_DATE,
        freq="D",
    )

    daily = daily.reindex(date_range, fill_value=0)
    daily.index.name = "date"

    return daily


def load_sales_data():
    return load_daily_sales()["revenue"]


def chronological_split(series, test_size=TEST_SIZE):
    return int(len(series) * (1 - test_size))


def time_series_analysis(daily):
    revenue = daily["revenue"]

    zero_days = int((daily["orders"] == 0).sum())

    weekly = daily.resample("W-SUN").sum()
    monthly = daily.resample("ME").sum()

    weekday_profile = daily.groupby(daily.index.dayofweek).agg(
        revenue=("revenue", "mean"),
        orders=("orders", "mean"),
    )
    weekday_profile.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    print("\nTIME SERIES OVERVIEW")
    print("=" * 55)
    print(f"Period: {daily.index.min().date()} to {daily.index.max().date()} "
          f"({len(daily)} days)")
    print(f"Days without sales (filled with 0): {zero_days}")
    print(f"Average daily revenue: {revenue.mean():,.2f}")
    print(f"Average daily orders: {daily['orders'].mean():.2f}")
    print(f"Average weekly revenue: {weekly['revenue'].mean():,.2f}")

    print("\nAverage revenue and orders by weekday:")
    print(weekday_profile.round(2))

    print("\nMonthly revenue and orders:")
    monthly_table = monthly.copy()
    monthly_table.index = monthly_table.index.strftime("%Y-%m")
    print(monthly_table.round(2))

    first_quarter = monthly["revenue"].iloc[:3].mean()
    last_quarter = monthly["revenue"].iloc[-3:].mean()
    print(f"\nTrend: the average monthly revenue in Q4 is "
          f"{(last_quarter / first_quarter - 1) * 100:+.1f}% compared with Q1.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    axes[0, 0].plot(revenue.index, revenue, color="#c3c2b7", linewidth=1, label="Daily revenue")
    axes[0, 0].plot(revenue.rolling(7).mean(), color=PALETTE[0], linewidth=2,
                    label="7-day moving average")
    axes[0, 0].plot(revenue.rolling(30).mean(), color=PALETTE[1], linewidth=2,
                    label="30-day moving average")
    axes[0, 0].set_title("Daily revenue and trend")
    axes[0, 0].set_ylabel("Revenue")
    axes[0, 0].legend()

    axes[0, 1].bar(weekly.index, weekly["orders"], width=5, color=PALETTE[0])
    axes[0, 1].set_title("Weekly number of orders")
    axes[0, 1].set_ylabel("Orders")

    axes[1, 0].bar(weekday_profile.index, weekday_profile["revenue"], color=PALETTE[0])
    axes[1, 0].set_title("Average revenue by weekday (weekly seasonality)")
    axes[1, 0].set_ylabel("Revenue")

    axes[1, 1].bar(monthly.index.strftime("%b"), monthly["revenue"], color=PALETTE[0])
    axes[1, 1].set_title("Monthly revenue (yearly seasonality)")
    axes[1, 1].set_ylabel("Revenue")

    return save_figure(fig, "08_time_series_analysis")


def regression_features(index, start_position):
    features = pd.DataFrame(
        {"time_index": np.arange(start_position, start_position + len(index))},
        index=index,
    )

    # One-hot encoded weekday captures the weekly pattern
    weekdays = pd.get_dummies(
        pd.Categorical(index.dayofweek, categories=range(7)),
        prefix="weekday",
        drop_first=True,
        dtype=int,
    )
    weekdays.index = index

    return features.join(weekdays)


def fit_regression(series):
    model = LinearRegression()
    model.fit(regression_features(series.index, 0), series.values)

    return model


def baseline_predictions(series, split_index, window=MOVING_AVERAGE_WINDOW):
    """One-step-ahead forecasts: each day uses only data observed before it."""
    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    naive = series.shift(1).iloc[split_index:].to_numpy()

    moving_average = (
        series.shift(1)
        .rolling(window)
        .mean()
        .iloc[split_index:]
        .to_numpy()
    )

    model = fit_regression(train)
    regression = np.maximum(
        model.predict(regression_features(test.index, split_index)),
        0,
    )

    return {
        "Naive": naive,
        "Moving Average": moving_average,
        "Linear Regression": regression,
    }


def evaluate_forecasts(actual, predictions):
    mse = mean_squared_error(actual, predictions)

    return {
        "MAE": mean_absolute_error(actual, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }


def future_forecast(series, model_name, horizon=FORECAST_HORIZON,
                    window=MOVING_AVERAGE_WINDOW):
    future_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    if model_name == "Naive":
        values = np.full(horizon, series.iloc[-1])

    elif model_name == "Moving Average":
        history = list(series.values)
        values = []

        for _ in range(horizon):
            prediction = float(np.mean(history[-window:]))
            values.append(prediction)
            history.append(prediction)

    else:
        model = fit_regression(series)
        values = np.maximum(
            model.predict(regression_features(future_dates, len(series))),
            0,
        )

    return pd.Series(values, index=future_dates, name="forecasted_revenue")


def run_forecasting():
    daily = load_daily_sales()
    revenue = daily["revenue"]

    time_series_analysis(daily)

    split_index = chronological_split(revenue)

    train = revenue.iloc[:split_index]
    test = revenue.iloc[split_index:]

    print("\nCHRONOLOGICAL SPLIT")
    print("=" * 55)
    print(f"Training: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Testing:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

    predictions = baseline_predictions(revenue, split_index)

    metrics = pd.DataFrame({
        model_name: evaluate_forecasts(test.values, forecast)
        for model_name, forecast in predictions.items()
    }).T

    print("\nFORECAST EVALUATION (one-step-ahead on the test period)")
    print("=" * 55)
    print(metrics.round(2))

    best_model = metrics["RMSE"].idxmin()
    print(f"\nBest baseline model based on RMSE: {best_model}")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(test.index, test.values, color="#0b0b0b", linewidth=2, label="Actual")

    for model_name, forecast in predictions.items():
        ax.plot(test.index, forecast, color=MODEL_COLORS[model_name],
                linewidth=1.5, label=model_name)

    ax.set_title("Actual vs forecasted daily revenue (test period)")
    ax.set_ylabel("Revenue")
    ax.legend()
    save_figure(fig, "09_baseline_forecasts")

    forecast = future_forecast(revenue, best_model)

    print(f"\nNEXT {FORECAST_HORIZON} DAYS FORECAST ({best_model})")
    print("=" * 55)
    print(f"Projected revenue: {forecast.sum():,.2f}")
    print(f"Actual revenue in the last {FORECAST_HORIZON} days: "
          f"{revenue.iloc[-FORECAST_HORIZON:].sum():,.2f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(revenue.index[-90:], revenue.iloc[-90:], color="#0b0b0b", linewidth=1.5,
            label="Actual revenue")
    ax.plot(forecast.index, forecast.values, color=MODEL_COLORS[best_model],
            linewidth=2, linestyle="--", label=f"{best_model} forecast")
    ax.set_title(f"Historical revenue and {FORECAST_HORIZON}-day forecast")
    ax.set_ylabel("Revenue")
    ax.legend()
    save_figure(fig, "10_baseline_future_forecast")

    test_predictions = pd.DataFrame({"date": test.index, "actual": test.values})
    for model_name, values in predictions.items():
        test_predictions[model_name] = values

    test_predictions.to_csv(
        PROCESSED_DATA_DIR / "forecast_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.rename_axis("model").to_csv(
        PROCESSED_DATA_DIR / "forecast_metrics.csv",
        encoding="utf-8-sig",
    )

    forecast.rename_axis("date").reset_index().assign(model=best_model).to_csv(
        PROCESSED_DATA_DIR / "baseline_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nForecast results saved to: {PROCESSED_DATA_DIR}")

    return metrics


if __name__ == "__main__":
    run_forecasting()
