import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from big_data_analytics.config import (
    DATA_END_DATE,
    DATA_START_DATE,
    PROCESSED_DATA_DIR,
)


def load_sales_data():
    orders = pd.read_csv(PROCESSED_DATA_DIR / "orders.csv")
    items = pd.read_csv(PROCESSED_DATA_DIR / "order_items.csv")

    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # Only delivered orders count as realized revenue
    delivered_orders = orders[orders["status"] == "Delivered"]

    sales = items.merge(
        delivered_orders[["order_id", "order_date"]],
        on="order_id",
        how="inner",
    )

    sales["revenue"] = sales["quantity"] * sales["unit_price"]

    daily_revenue = (
        sales.groupby("order_date")["revenue"]
        .sum()
        .sort_index()
    )

    date_range = pd.date_range(
        start=DATA_START_DATE,
        end=DATA_END_DATE,
        freq="D",
    )

    daily_revenue = daily_revenue.reindex(
        date_range,
        fill_value=0,
    )

    daily_revenue.index.name = "date"

    return daily_revenue


def moving_average_forecast(training_data, horizon, window=7):
    history = list(training_data)
    predictions = []

    for _ in range(horizon):
        prediction = float(np.mean(history[-window:]))
        predictions.append(prediction)
        history.append(prediction)

    return np.array(predictions)


def evaluate_forecasts(actual, predictions):
    return {
        "MAE": mean_absolute_error(actual, predictions),
        "MSE": mean_squared_error(actual, predictions),
        "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
    }


def run_forecasting():
    revenue = load_sales_data()

    # Chronological 80/20 split
    split_index = int(len(revenue) * 0.8)

    train = revenue.iloc[:split_index]
    test = revenue.iloc[split_index:]

    print(f"Total days: {len(revenue)}")
    print(f"Training days: {len(train)}")
    print(f"Testing days: {len(test)}")

    # 1. Naive forecast
    naive_predictions = np.full(
        len(test),
        train.iloc[-1],
    )

    # 2. Seven-day moving average
    moving_predictions = moving_average_forecast(
        train.values,
        len(test),
        window=7,
    )

    # 3. Linear regression
    train_x = np.arange(len(train)).reshape(-1, 1)
    test_x = np.arange(len(train), len(revenue)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(train_x, train.values)

    regression_predictions = model.predict(test_x)
    regression_predictions = np.maximum(regression_predictions, 0)

    # Compare models
    predictions = {
        "Naive": naive_predictions,
        "Moving Average": moving_predictions,
        "Linear Regression": regression_predictions,
    }

    results = {}

    print("\nFORECAST EVALUATION")
    print("=" * 55)

    for model_name, forecast in predictions.items():
        metrics = evaluate_forecasts(test.values, forecast)
        results[model_name] = metrics

        print(f"\n{model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:,.2f}")

    best_model = min(
        results,
        key=lambda name: results[name]["RMSE"],
    )

    print(f"\nBest model based on RMSE: {best_model}")

    # Visualize actual vs predicted revenue
    plt.figure(figsize=(13, 6))
    plt.plot(test.index, test.values, label="Actual", linewidth=2)

    for model_name, forecast in predictions.items():
        plt.plot(test.index, forecast, label=model_name, alpha=0.8)

    plt.title("Actual vs Forecasted Daily Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Monthly revenue trend
    monthly_revenue = revenue.resample("ME").sum()

    plt.figure(figsize=(11, 5))
    monthly_revenue.plot(marker="o")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Forecast the next 30 days using the best model
    horizon = 30

    if best_model == "Naive":
        future_predictions = np.full(horizon, revenue.iloc[-1])

    elif best_model == "Moving Average":
        future_predictions = moving_average_forecast(
            revenue.values,
            horizon,
            window=7,
        )

    else:
        full_x = np.arange(len(revenue)).reshape(-1, 1)
        future_x = np.arange(
            len(revenue),
            len(revenue) + horizon,
        ).reshape(-1, 1)

        final_model = LinearRegression()
        final_model.fit(full_x, revenue.values)

        future_predictions = np.maximum(
            final_model.predict(future_x),
            0,
        )

    future_dates = pd.date_range(
        start=revenue.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    forecast = pd.Series(
        future_predictions,
        index=future_dates,
        name="forecasted_revenue",
    )

    print("\nNEXT 30 DAYS FORECAST")
    print("=" * 55)
    print(forecast.round(2))
    print(f"\nProjected revenue: {forecast.sum():,.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(
        revenue.index[-60:],
        revenue.iloc[-60:],
        label="Historical revenue",
    )
    plt.plot(
        forecast.index,
        forecast.values,
        label="30-day forecast",
        linestyle="--",
        marker="o",
    )

    plt.title("Historical Revenue and 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_forecasting()