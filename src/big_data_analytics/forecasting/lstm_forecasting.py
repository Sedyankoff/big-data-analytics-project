
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from big_data_analytics.config import PROCESSED_DATA_DIR
from big_data_analytics.forecasting.forecasting import load_sales_data


LOOKBACK = 30
EPOCHS = 40
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
    revenue = load_sales_data()

    values = revenue.values.reshape(-1, 1)

    split_index = int(len(values) * 0.8)

    train_values = values[:split_index]

    # Fit the scaler only on training data to prevent data leakage
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_values)

    # Prepare training sequences
    X_train, y_train = create_sequences(
        scaled_train,
        LOOKBACK,
    )

    # Transform the full series using the training scaler
    scaled_all = scaler.transform(values)

    # Prepare test sequences using the preceding 30 actual days
    X_test = []
    y_test = []

    for i in range(split_index, len(scaled_all)):
        X_test.append(scaled_all[i - LOOKBACK:i])
        y_test.append(scaled_all[i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # LSTM expects: samples, time steps, features
    model = build_model()

    print("\nTRAINING LSTM MODEL")
    print("=" * 50)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1,
    )

    # Evaluate on the chronological test period
    scaled_predictions = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(
        scaled_predictions
    ).flatten()

    actual = scaler.inverse_transform(
        y_test
    ).flatten()

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)

    print("\nLSTM EVALUATION")
    print("=" * 50)
    print(f"MAE:  {mae:,.2f}")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    # Forecast the next 30 days recursively
    history = list(scaled_all.flatten())
    future_predictions = []

    for _ in range(30):
        sequence = np.array(
            history[-LOOKBACK:]
        ).reshape(1, LOOKBACK, 1)

        next_prediction = model.predict(
            sequence,
            verbose=0,
        )[0, 0]

        future_predictions.append(next_prediction)
        history.append(next_prediction)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()

    future_predictions = np.maximum(
        future_predictions,
        0,
    )

    future_dates = pd.date_range(
        start=revenue.index[-1] + pd.Timedelta(days=1),
        periods=30,
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

    print("\nNEXT 30 DAYS")
    print("=" * 50)
    print(f"Projected revenue: {future_predictions.sum():,.2f}")
    print(f"Forecast saved to: {PROCESSED_DATA_DIR / 'lstm_forecast.csv'}")

    # Plot actual vs predicted values
    test_dates = revenue.index[split_index:]

    plt.figure(figsize=(13, 6))
    plt.plot(
        test_dates,
        actual,
        label="Actual revenue",
    )
    plt.plot(
        test_dates,
        predictions,
        label="LSTM predictions",
    )

    plt.title("LSTM: Actual vs Predicted Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot future forecast
    plt.figure(figsize=(12, 5))
    plt.plot(
        revenue.index[-60:],
        revenue.iloc[-60:],
        label="Historical revenue",
    )
    plt.plot(
        future_dates,
        future_predictions,
        label="LSTM forecast",
        linestyle="--",
        marker="o",
    )

    plt.title("LSTM 30-Day Revenue Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_lstm_forecasting()
