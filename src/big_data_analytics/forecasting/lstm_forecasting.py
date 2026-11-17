import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from big_data_analytics.config import (  # noqa: E402
    FORECAST_HORIZON,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from big_data_analytics.forecasting.forecasting import (  # noqa: E402
    MODEL_COLORS,
    baseline_predictions,
    chronological_split,
    evaluate_forecasts,
    load_sales_data,
)
from big_data_analytics.plotting import save_figure  # noqa: E402


LOOKBACK = 30
EPOCHS = 60
BATCH_SIZE = 16


def create_sequences(data, lookback):
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(LOOKBACK, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    return model


def run_lstm_forecasting():
    keras.utils.set_random_seed(RANDOM_SEED)

    revenue = load_sales_data()

    values = revenue.values.reshape(-1, 1)

    split_index = chronological_split(revenue)

    train_values = values[:split_index]

    # Fit the scaler only on training data to prevent data leakage
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_values)

    X_train, y_train = create_sequences(
        scaled_train,
        LOOKBACK,
    )

    # Transform the full series using the training scaler
    scaled_all = scaler.transform(values)

    # Each test day is predicted from the preceding 30 actual days
    X_test = np.array([
        scaled_all[i - LOOKBACK:i]
        for i in range(split_index, len(scaled_all))
    ])

    model = build_model()

    print("\nTRAINING LSTM MODEL")
    print("=" * 50)

    # The last 15% of the training windows are used for validation
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    scaled_predictions = model.predict(X_test, verbose=0)

    predictions = np.maximum(
        scaler.inverse_transform(scaled_predictions).flatten(),
        0,
    )

    test = revenue.iloc[split_index:]
    actual = test.values

    # Compare the LSTM with the baseline models on the same test days
    all_predictions = baseline_predictions(revenue, split_index)
    all_predictions["LSTM"] = predictions

    metrics = pd.DataFrame({
        model_name: evaluate_forecasts(actual, forecast)
        for model_name, forecast in all_predictions.items()
    }).T

    print("\nMODEL COMPARISON (one-step-ahead on the test period)")
    print("=" * 50)
    print(metrics.round(2))

    best_baseline = metrics.drop(index="LSTM")["RMSE"].idxmin()
    lstm_rmse = metrics.loc["LSTM", "RMSE"]
    baseline_rmse = metrics.loc[best_baseline, "RMSE"]
    difference = (lstm_rmse / baseline_rmse - 1) * 100

    print(f"\nBest baseline: {best_baseline} (RMSE {baseline_rmse:,.2f})")
    print(f"LSTM RMSE: {lstm_rmse:,.2f} ({difference:+.1f}% vs. best baseline)")

    if difference < 0:
        print("The LSTM improves on the best baseline model.")
    else:
        print(
            "The LSTM does not outperform the best baseline. With one year of noisy "
            "daily data, simple models are hard to beat."
        )

    # Forecast the next 30 days recursively
    history_values = list(scaled_all.flatten())
    future_predictions = []

    for _ in range(FORECAST_HORIZON):
        sequence = np.array(
            history_values[-LOOKBACK:]
        ).reshape(1, LOOKBACK, 1)

        next_prediction = float(model.predict(sequence, verbose=0)[0, 0])

        future_predictions.append(next_prediction)
        history_values.append(next_prediction)

    future_predictions = np.maximum(
        scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten(),
        0,
    )

    future_dates = pd.date_range(
        start=revenue.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D",
    )

    forecast = pd.DataFrame({
        "date": future_dates,
        "forecasted_revenue": future_predictions,
    })

    forecast.to_csv(
        PROCESSED_DATA_DIR / "lstm_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_predictions = pd.DataFrame({"date": test.index, "actual": actual})
    for model_name, forecast_values in all_predictions.items():
        test_predictions[model_name] = forecast_values

    test_predictions.to_csv(
        PROCESSED_DATA_DIR / "forecast_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.rename_axis("model").to_csv(
        PROCESSED_DATA_DIR / "forecast_metrics.csv",
        encoding="utf-8-sig",
    )

    print(f"\nNEXT {FORECAST_HORIZON} DAYS")
    print("=" * 50)
    print(f"Projected revenue: {future_predictions.sum():,.2f}")
    print(f"Forecast saved to: {PROCESSED_DATA_DIR / 'lstm_forecast.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2.4]})

    axes[0].plot(history.history["loss"], color=MODEL_COLORS["LSTM"], label="Training loss")
    axes[0].plot(history.history["val_loss"], color=MODEL_COLORS["Linear Regression"],
                 label="Validation loss")
    axes[0].set_title("LSTM training history")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (scaled)")
    axes[0].legend()

    axes[1].plot(test.index, actual, color="#0b0b0b", linewidth=2, label="Actual revenue")
    axes[1].plot(test.index, predictions, color=MODEL_COLORS["LSTM"], linewidth=1.5,
                 label="LSTM")
    axes[1].plot(test.index, all_predictions[best_baseline],
                 color=MODEL_COLORS[best_baseline], linewidth=1.5,
                 label=f"{best_baseline} (best baseline)")
    axes[1].set_title("LSTM vs best baseline on the test period")
    axes[1].set_ylabel("Revenue")
    axes[1].legend()

    save_figure(fig, "11_lstm_evaluation")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(revenue.index[-90:], revenue.iloc[-90:], color="#0b0b0b", linewidth=1.5,
            label="Actual revenue")
    ax.plot(future_dates, future_predictions, color=MODEL_COLORS["LSTM"], linewidth=2,
            linestyle="--", label="LSTM forecast")
    ax.set_title(f"LSTM {FORECAST_HORIZON}-day revenue forecast")
    ax.set_ylabel("Revenue")
    ax.legend()

    save_figure(fig, "12_lstm_future_forecast")

    return metrics


if __name__ == "__main__":
    run_lstm_forecasting()
