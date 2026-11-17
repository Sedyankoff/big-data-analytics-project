import time

from big_data_analytics.analysis.business_analysis import main as run_analysis
from big_data_analytics.config import FIGURES_DIR, PROCESSED_DATA_DIR
from big_data_analytics.data_generation.generator import main as generate_data
from big_data_analytics.data_loader import (
    inspect_datasets,
    load_datasets,
    validate_dataset_requirements,
    validate_relationships,
)
from big_data_analytics.forecasting.forecasting import run_forecasting
from big_data_analytics.forecasting.lstm_forecasting import run_lstm_forecasting
from big_data_analytics.preprocessing.cleaner import preprocess
from big_data_analytics.segmentation.customer_segmentation import run_segmentation
from big_data_analytics.sentiment.sentiment_analysis import run_sentiment_analysis


def inspect_data():
    datasets = load_datasets()

    inspect_datasets(datasets)
    validate_dataset_requirements(datasets)
    validate_relationships(datasets)


STAGES = [
    ("1. Data generation", generate_data),
    ("1. Data inspection", inspect_data),
    ("2. Preprocessing", preprocess),
    ("3. Statistical and business analysis", run_analysis),
    ("4. Time series forecasting", run_forecasting),
    ("4. LSTM forecasting", run_lstm_forecasting),
    ("5. Sentiment analysis", run_sentiment_analysis),
    ("6. Customer segmentation", run_segmentation),
]


def main():
    started = time.perf_counter()

    for title, stage in STAGES:
        print(f"\n{'#' * 70}\n# {title}\n{'#' * 70}")
        stage()

    print(f"\nPipeline completed in {time.perf_counter() - started:.0f} seconds.")
    print(f"Processed data: {PROCESSED_DATA_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("\nStart the dashboard with:")
    print("  streamlit run src/big_data_analytics/dashboard/app.py")


if __name__ == "__main__":
    main()
