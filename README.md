# Big Data Analytics — Online Shop

A Python-based business analytics system for analyzing online shop
sales, customer behavior, product performance, and customer sentiment.

## Project Overview

The application generates realistic synthetic e-commerce data, cleans
and validates it, performs statistical analysis, forecasts sales
(baseline models and an LSTM network), analyzes customer reviews,
segments customers with K-means, and presents the results in an
interactive Streamlit dashboard.

## Technology Stack

- Python 3.14
- Faker, NumPy and Pandas — data generation and processing
- SciPy — statistical tests and theoretical distributions
- Scikit-learn — encoding, scaling, linear regression, K-means, TF-IDF
- Keras (PyTorch backend) — LSTM forecasting
- Matplotlib and Seaborn — analysis figures
- Plotly and Streamlit — interactive dashboard

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

## Running the Project

Run the complete pipeline (about 30 seconds):

```bash
python scripts/run_pipeline.py
```

Then start the dashboard:

```bash
streamlit run src/big_data_analytics/dashboard/app.py
```

The dashboard opens at http://localhost:8501.

Each stage can also be run separately, in this order:

| Step | Command | Output |
|------|---------|--------|
| 1 | `python scripts/generate_data.py` | `data/raw/*.csv` |
| 2 | `python scripts/inspect_data.py` | structure, sizes and relationship checks |
| 3 | `python scripts/preprocess_data.py` | `data/processed/*.csv` |
| 4 | `python scripts/run_analysis.py` | statistics, correlations, business KPIs |
| 5 | `python scripts/run_forecasting.py` | baseline forecasts and metrics |
| 6 | `python scripts/run_lstm.py` | LSTM forecast and model comparison |
| 7 | `python scripts/run_sentiment.py` | `reviews_analyzed.csv` |
| 8 | `python scripts/run_segmentation.py` | `customer_segments.csv` |

All figures are saved to `reports/figures/`. Results are printed to the
console.

### Optional: AI business insights

The dashboard includes an optional page that sends the aggregated
results to a language model through OpenRouter: overall KPIs, category,
monthly and segment summaries, sentiment statistics, the LSTM forecast
and the evaluation metrics of all forecasting models. Customer names,
review texts and other personal data are not sent.

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to use it,
or run `python scripts/run_llm_insights.py`. Every number in the
generated report is compared with the calculated results, and numbers
that are not found are listed for manual review. The report, the exact
input data and the verification table are saved to `data/processed/`
(`llm_business_insights.md`, `llm_input_data.json`,
`llm_report_verification.csv`).

## Project Structure

```
scripts/                        entry points for every stage
src/big_data_analytics/
    config.py                   paths, random seed, dataset sizes
    data_generation/            synthetic data with controlled quality issues
    data_loader.py              CSV loading, inspection and validation
    preprocessing/              cleaning, outliers, encoding
    analysis/                   descriptive statistics, correlations, business KPIs
    forecasting/                time series, baselines and LSTM
    sentiment/                  text processing and sentiment analysis
    segmentation/               customer features and K-means
    dashboard/                  Streamlit application
data/raw, data/processed        generated CSV files
reports/figures                 generated charts
```

## Methodology

**Data.** 500 customers, 30 products in 6 categories, 2,000 orders,
about 3,700 order lines and 800 reviews for 2025 (12 months). A fixed
seed (`RANDOM_SEED = 42`) makes every run identical. Orders follow a
yearly seasonality (peak in November–December), weekday effects and an
upward trend; VIP and regular customers order more often; prices and
margins depend on the product category; ratings depend on product
quality and are lower during the busy holiday period. Order values are
always calculated from quantity × unit price.

**Data quality issues.** The raw files intentionally contain missing
values, duplicate rows, invalid and mixed date formats, decimal commas,
inconsistent text labels, impossible ages, extreme quantities and
out-of-range ratings. The preprocessing stage fixes or removes them,
compares statistics before and after cleaning, and checks referential
integrity.

**Outliers.** Quantities are checked with IQR and Z-score; values
flagged by both methods are treated as errors and replaced with the
median. Box plots show the data before and after cleaning.

**Encoding and scaling.** Label encoding for customer category and city,
one-hot encoding for order status, payment method and weekday (used by
the regression model). StandardScaler is used for K-means and
MinMaxScaler (fitted on training data only) for the LSTM.

**Revenue.** Delivered and shipped orders count as revenue; cancelled
orders are excluded.

**Forecasting.** Daily revenue is split chronologically (80% training,
20% test). Naive (last observed value), 7-day moving average, linear
regression (trend + weekday) and an LSTM (30-day window) are compared
with MAE, MSE and RMSE. All models are evaluated one step ahead, so each
prediction uses only data available before that day.

**Sentiment.** Texts are cleaned (URLs, symbols, case), tokenized and
analyzed with word counts, n-grams (CountVectorizer) and TF-IDF.
Sentiment is assigned by a Bulgarian lexicon with negation handling and
validated against the star ratings.

**Segmentation.** Customers are described by purchase frequency, total
spending, average order value and recency. The number of clusters is
chosen with the elbow method and the silhouette score, and each cluster
receives a business name based on its profile.

## Dashboard

- KPIs: revenue, profit, margin, orders, customers, average order value,
  revenue growth for comparable periods and share of positive reviews
- Filters: period, product category and customer segment (applied to all
  KPIs, sales, product, customer and sentiment views)
- Tabs: sales trends, product categories, customer segments, review
  sentiment, actual vs forecasted sales, and data-driven findings with
  recommendations

## Limitations

- The data is synthetic, so the patterns reflect the assumptions of the
  generator.
- One year of daily data is short for deep learning; the LSTM does not
  outperform the simple moving average on the test period.
- The lexicon-based sentiment classifier does not understand sarcasm and
  assigns a single label to mixed reviews.
