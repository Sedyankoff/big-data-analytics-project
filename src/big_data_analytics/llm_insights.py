
import json
import math
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv

from big_data_analytics.config import (
    FORECAST_HORIZON,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REVENUE_STATUSES,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"

NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def load_business_data():
    required_files = [
        "orders",
        "order_items",
        "products",
        "customers",
    ]

    data = {
        name: pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
        for name in required_files
    }

    data["orders"]["order_date"] = pd.to_datetime(
        data["orders"]["order_date"],
        errors="coerce",
    )

    return data


def build_business_context():
    data = load_business_data()

    orders = data["orders"]
    items = data["order_items"]
    products = data["products"]
    customers = data["customers"]

    # Analyze completed orders only.
    paid_orders = orders[
        orders["status"].isin(REVENUE_STATUSES)
    ].copy()

    sales = (
        items.merge(products, on="product_id", how="left")
        .merge(
            paid_orders[["order_id", "customer_id", "order_date"]],
            on="order_id",
            how="inner",
        )
    )

    sales["revenue"] = sales["line_total"]
    sales["cost_total"] = sales["quantity"] * sales["cost"]
    sales["profit"] = sales["revenue"] - sales["cost_total"]

    total_revenue = float(sales["revenue"].sum())
    total_cost = float(sales["cost_total"].sum())
    total_profit = float(sales["profit"].sum())

    order_values = sales.groupby("order_id")["revenue"].sum()

    metrics = {
        "period": f"{paid_orders['order_date'].min():%Y-%m-%d} to "
                  f"{paid_orders['order_date'].max():%Y-%m-%d}",
        "total_revenue": round(total_revenue, 2),
        "total_cost": round(total_cost, 2),
        "total_profit": round(total_profit, 2),
        "profit_margin_percent": round(
            total_profit / total_revenue * 100, 2
        ) if total_revenue else 0,
        "paid_orders": int(paid_orders["order_id"].nunique()),
        "cancelled_orders": int((orders["status"] == "Cancelled").sum()),
        "registered_customers": int(customers["customer_id"].nunique()),
        "customers_with_paid_orders": int(
            paid_orders["customer_id"].nunique()
        ),
        "average_order_value": round(float(order_values.mean()), 2)
        if not order_values.empty else 0,
    }

    # Product category performance.
    categories = (
        sales.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units_sold=("quantity", "sum"),
        )
        .sort_values("profit", ascending=False)
    )
    categories["profit_margin_percent"] = categories["profit"] / categories["revenue"] * 100
    categories["revenue_share_percent"] = categories["revenue"] / total_revenue * 100
    categories["profit_rank"] = range(1, len(categories) + 1)
    categories["revenue_rank"] = categories["revenue"].rank(ascending=False).astype(int)
    categories["margin_rank"] = (
        categories["profit_margin_percent"].rank(ascending=False).astype(int)
    )

    category_summary = (
        categories.round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    monthly = (
        sales.groupby(sales["order_date"].dt.to_period("M"))
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
        )
    )
    monthly["revenue_change_vs_previous_month_percent"] = (
        monthly["revenue"].pct_change() * 100
    )
    monthly.index = monthly.index.strftime("%Y-%m")

    monthly_records = monthly.round(2).rename_axis("month").reset_index().to_dict("records")

    for record in monthly_records:
        if pd.isna(record["revenue_change_vs_previous_month_percent"]):
            record["revenue_change_vs_previous_month_percent"] = None

    monthly_summary = {
        "months": monthly_records,
        "highest_revenue_month": monthly["revenue"].idxmax(),
        "lowest_revenue_month": monthly["revenue"].idxmin(),
        "largest_monthly_decline": monthly[
            "revenue_change_vs_previous_month_percent"
        ].idxmin(),
    }

    # Customer segment performance.
    segment_path = PROCESSED_DATA_DIR / "customer_segments.csv"

    segment_summary = []

    if segment_path.exists():
        segments = pd.read_csv(segment_path)

        segment_table = (
            segments.groupby("segment")
            .agg(
                customers=("customer_id", "count"),
                average_orders=("purchase_frequency", "mean"),
                average_spending=("total_spending", "mean"),
                average_order_value=("average_order_value", "mean"),
                average_days_since_last_order=("recency_days", "mean"),
                total_revenue=("total_spending", "sum"),
            )
            .sort_values("total_revenue", ascending=False)
        )
        segment_table["customer_share_percent"] = (
            segment_table["customers"] / segment_table["customers"].sum() * 100
        )
        segment_table["revenue_share_percent"] = (
            segment_table["total_revenue"] / segment_table["total_revenue"].sum() * 100
        )
        segment_table["revenue_rank"] = range(1, len(segment_table) + 1)

        segment_summary = (
            segment_table.round(2)
            .reset_index()
            .to_dict(orient="records")
        )

    # Review sentiment statistics.
    sentiment_summary = {}

    reviews_path = PROCESSED_DATA_DIR / "reviews_analyzed.csv"

    if reviews_path.exists():
        reviews = pd.read_csv(reviews_path, parse_dates=["review_date"])

        if "sentiment" in reviews.columns:
            reviews = reviews.merge(
                products[["product_id", "category"]],
                on="product_id",
                how="left",
            )

            monthly_positive = (
                reviews.groupby(reviews["review_date"].dt.strftime("%Y-%m"))["sentiment"]
                .apply(lambda values: (values == "Positive").mean() * 100)
            )

            sentiment_summary = {
                "reviews": int(len(reviews)),
                "sentiment_percentages": (
                    reviews["sentiment"]
                    .value_counts(normalize=True)
                    .mul(100)
                    .round(2)
                    .to_dict()
                ),
                "average_rating": round(float(reviews["rating"].mean()), 2),
                "average_rating_by_category": (
                    reviews.groupby("category")["rating"].mean()
                    .sort_values()
                    .round(2)
                    .to_dict()
                ),
                "month_with_lowest_positive_share": monthly_positive.idxmin(),
                "lowest_positive_share_percent": round(float(monthly_positive.min()), 2),
                "month_with_highest_positive_share": monthly_positive.idxmax(),
                "highest_positive_share_percent": round(float(monthly_positive.max()), 2),
            }

    # Future forecast summary.
    forecast_summary = {}

    forecast_path = PROCESSED_DATA_DIR / "lstm_forecast.csv"

    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path, parse_dates=["date"])
        values = forecast["forecasted_revenue"]

        daily_revenue = sales.groupby("order_date")["revenue"].sum()
        last_period_revenue = float(
            daily_revenue[
                daily_revenue.index > daily_revenue.index.max()
                - pd.Timedelta(days=FORECAST_HORIZON)
            ].sum()
        )

        forecast_summary = {
            "model": "LSTM",
            "forecast_days": int(len(values)),
            "first_forecast_date": f"{forecast['date'].min():%Y-%m-%d}",
            "total_forecasted_revenue": round(float(values.sum()), 2),
            "average_daily_forecasted_revenue": round(float(values.mean()), 2),
            "actual_revenue_last_period": round(last_period_revenue, 2),
            "change_vs_last_period_percent": round(
                (float(values.sum()) / last_period_revenue - 1) * 100, 2
            ) if last_period_revenue else None,
        }

    evaluation_summary = {}

    metrics_path = PROCESSED_DATA_DIR / "forecast_metrics.csv"

    if metrics_path.exists():
        evaluation = pd.read_csv(metrics_path).sort_values("RMSE")

        evaluation_summary = {
            "method": "Chronological split, last 20% of days used as test period",
            "models_sorted_by_rmse": evaluation.round(2).to_dict(orient="records"),
            "most_accurate_model": evaluation["model"].iloc[0],
        }

    return {
        "business_metrics": metrics,
        "product_category_performance": category_summary,
        "monthly_performance": monthly_summary,
        "customer_segment_performance": segment_summary,
        "customer_sentiment": sentiment_summary,
        "lstm_forecast": forecast_summary,
        "forecast_model_evaluation": evaluation_summary,
    }


def collect_numbers(value):
    if isinstance(value, bool) or value is None:
        return []

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, str):
        return [
            float(match.replace(",", ""))
            for match in NUMBER_PATTERN.findall(value)
        ]

    if isinstance(value, dict):
        value = list(value.values())

    return [number for item in value for number in collect_numbers(item)]


def verify_report(report, context, tolerance=0.005):
    reference = collect_numbers(context)
    checks = []

    for match in NUMBER_PATTERN.finditer(report):
        text = match.group()
        number = float(text.replace(",", ""))

        is_small_integer = "." not in text and abs(number) <= 12
        is_year = "." not in text and "," not in text and 1900 <= number <= 2100

        if is_small_integer or is_year:
            continue

        verified = any(
            math.isclose(abs(number), abs(value), rel_tol=tolerance, abs_tol=0.01)
            for value in reference
        )

        start = report.rfind("\n", 0, match.start()) + 1
        end = report.find("\n", match.end())
        sentence = report[start:end if end != -1 else None].strip(" *-#")

        checks.append({
            "number": text,
            "verified": verified,
            "text": sentence[:200],
        })

    return pd.DataFrame(checks, columns=["number", "verified", "text"])


def request_report(context, api_key, model):
    system_prompt = """
You are a professional business data analyst.

Analyze the supplied online shop metrics and produce a concise business report.

Your report must contain:
1. Executive summary.
2. Sales and profitability analysis.
3. Product category performance.
4. Monthly sales trend and periods of decline.
5. Customer segment interpretation.
6. Customer sentiment interpretation.
7. Sales forecast interpretation.
8. Three to five specific, actionable business recommendations.

Rules:
- Use only the supplied data.
- Do not invent figures, trends, causes or business facts.
- Copy numbers exactly as they appear in the data. Do not calculate new totals,
  sums, shares, differences or averages.
- Use the rank fields for rankings. Rank 1 is the highest value.
- Do not use currency symbols or currency names; write amounts as plain numbers.
- Support conclusions with actual numerical values.
- Clearly distinguish observations from hypotheses.
- Do not claim correlation proves causation.
- Explain when information is unavailable.
- Treat the LSTM forecast as an estimate, not a certainty, and compare it with
  the evaluation metrics of the other models.
- Write in professional, clear English using Markdown headings and bullet points.
"""

    user_prompt = (
        "Analyze the following calculated business data and prepare the report.\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Online Shop Business Analytics",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 3000,
            },
            timeout=180,
        )
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach OpenRouter: {error}") from error

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter API error ({response.status_code}): "
            f"{response.text[:1000]}"
        )

    result = response.json()

    if "error" in result:
        raise RuntimeError(f"OpenRouter API error: {result['error']}")

    choices = result.get("choices") or []

    if not choices:
        raise RuntimeError("The model returned no response.")

    content = (choices[0].get("message") or {}).get("content") or ""

    if not content.strip():
        raise RuntimeError("The model returned an empty response.")

    if choices[0].get("finish_reason") == "length":
        raise RuntimeError(
            "The report was cut off because it reached the token limit."
        )

    return content.strip()


def generate_business_insights():
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to the project root .env file."
        )

    context = build_business_context()
    report = request_report(context, api_key, model)
    verification = verify_report(report, context)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    (PROCESSED_DATA_DIR / "llm_business_insights.md").write_text(
        report, encoding="utf-8"
    )
    (PROCESSED_DATA_DIR / "llm_input_data.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification.to_csv(
        PROCESSED_DATA_DIR / "llm_report_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "model": model,
        "context": context,
        "report": report,
        "verification": verification,
    }


def print_insights(result):
    verification = result["verification"]
    verified = int(verification["verified"].sum())

    print(result["report"])

    print("\n" + "=" * 55)
    print(f"VERIFICATION ({result['model']})")
    print("=" * 55)
    print(f"Numbers in the report: {len(verification)}")
    print(f"Found in the calculated data: {verified}")

    unverified = verification[~verification["verified"]]

    if unverified.empty:
        print("All numbers in the report match the calculated results.")
    else:
        print("Numbers not found in the calculated data (check manually):")
        for row in unverified.itertuples(index=False):
            print(f"- {row.number}: {row.text}")

    print(f"\nReport, input data and verification saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    print_insights(generate_business_insights())
