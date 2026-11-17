from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from big_data_analytics.config import (
    CUSTOMER_COUNT,
    DATA_END_DATE,
    DATA_START_DATE,
    ORDER_COUNT,
    PRODUCT_COUNT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REVIEW_COUNT,
)


CITY_WEIGHTS = {
    "София": 0.32, "Пловдив": 0.13, "Варна": 0.12, "Бургас": 0.09,
    "Русе": 0.07, "Стара Загора": 0.07, "Плевен": 0.06,
    "Велико Търново": 0.05, "Благоевград": 0.05, "Смолян": 0.04,
}

PRODUCT_CATALOG = {
    "Electronics": [
        ("Headphones", 89), ("Smartphone", 649), ("Monitor", 279),
        ("Keyboard", 79), ("Mouse", 35),
    ],
    "Clothing": [
        ("T-shirt", 25), ("Jeans", 69), ("Jacket", 149),
        ("Sneakers", 95), ("Hoodie", 55),
    ],
    "Home": [
        ("Lamp", 39), ("Chair", 189), ("Desk", 349),
        ("Pillow", 29), ("Blanket", 45),
    ],
    "Sports": [
        ("Dumbbells", 79), ("Yoga Mat", 32), ("Backpack", 89),
        ("Bottle", 19), ("Resistance Bands", 22),
    ],
    "Books": [
        ("Novel", 18), ("Cookbook", 32), ("Biography", 26),
        ("Notebook", 9), ("Dictionary", 39),
    ],
    "Beauty": [
        ("Shampoo", 12), ("Face Cream", 38), ("Perfume", 85),
        ("Soap", 15), ("Body Lotion", 19),
    ],
}

COST_RATIO = {
    "Electronics": (0.68, 0.80),
    "Clothing": (0.38, 0.52),
    "Home": (0.50, 0.62),
    "Sports": (0.45, 0.58),
    "Books": (0.55, 0.68),
    "Beauty": (0.32, 0.45),
}

# Relative demand per calendar month (January ... December)
MONTHLY_DEMAND = [0.80, 0.78, 0.92, 0.98, 1.00, 0.92, 0.85, 0.83, 0.98, 1.05, 1.40, 1.55]
WEEKDAY_DEMAND = [1.05, 1.00, 0.98, 0.97, 1.00, 0.90, 1.10]

CATEGORY_SEASONALITY = {
    "Electronics": [0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.5, 1.6],
    "Clothing": [1.2, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.9, 1.1, 1.2, 1.3, 1.3],
    "Home": [1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2],
    "Sports": [1.3, 1.0, 1.1, 1.2, 1.3, 1.3, 1.2, 1.1, 0.9, 0.8, 0.8, 0.8],
    "Books": [1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.2, 1.3, 1.0, 1.0, 1.3],
    "Beauty": [0.9, 1.3, 1.2, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.5],
}

BLACK_FRIDAY_WEEK = (date(2025, 11, 24), date(2025, 12, 1))

REVIEW_TEXTS = {
    "positive": [
        "Много съм доволен от продукта. Отлично качество!",
        "Всичко е чудесно, бих поръчал отново.",
        "Продуктът напълно оправда очакванията ми.",
        "Страхотен продукт на добра цена.",
        "Препоръчвам го на всеки!",
        "Изключително съм доволна от покупката.",
        "Работи перфектно, точно както е описан.",
        "Отлична покупка, качеството е страхотно.",
    ],
    "neutral": [
        "Продуктът е нормален, нищо особено.",
        "Качеството е приемливо за цената.",
        "Очаквах малко повече, но като цяло става.",
        "Стандартен продукт, нищо изненадващо.",
        "Средно качество, ще видим колко ще издържи.",
        "Продуктът става, но има какво да се подобри.",
    ],
    "negative": [
        "Не съм доволен от качеството.",
        "Продуктът не отговаря на очакванията ми.",
        "Имаше проблем с поръчката.",
        "Не бих поръчал този продукт отново.",
        "Много лошо качество, разочарован съм.",
        "Продуктът пристигна повреден.",
        "Счупи се след седмица употреба.",
        "Ужасно обслужване и бавна доставка.",
    ],
}

SERVICE_COMMENTS = {
    "positive": [
        "Доставката беше бърза.",
        "Опаковката беше идеална.",
        "Обслужването е на ниво.",
    ],
    "neutral": [
        "Доставката отне няколко дни.",
        "Опаковката беше обикновена.",
    ],
    "negative": [
        "Доставката закъсня с цяла седмица.",
        "Никой не отговаря на имейлите.",
    ],
}

NOISE_SUFFIXES = [" !!!", " :)", " :(", " ...", " ??", " 👍", " 👎"]


def _rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def _daily_demand():
    days = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="D")
    progress = np.linspace(0, 1, len(days))

    weights = (
        (1 + 0.25 * progress)
        * np.array([MONTHLY_DEMAND[day.month - 1] for day in days])
        * np.array([WEEKDAY_DEMAND[day.weekday()] for day in days])
    )

    black_friday = (
        (days.date >= BLACK_FRIDAY_WEEK[0]) & (days.date <= BLACK_FRIDAY_WEEK[1])
    )
    weights[black_friday] *= 1.8

    after_christmas = days.date >= date(2025, 12, 24)
    weights[after_christmas] *= 0.6

    return days, weights / weights.sum()


def generate_customers(fake, rng):
    customers = []

    cities = list(CITY_WEIGHTS)
    city_probabilities = list(CITY_WEIGHTS.values())

    for customer_id in range(1, CUSTOMER_COUNT + 1):
        registration_date = fake.date_between(
            start_date=date(2023, 1, 1),
            end_date=DATA_END_DATE,
        )

        tenure_days = (DATA_END_DATE - registration_date).days

        if tenure_days < 180:
            category_probabilities = [0.80, 0.18, 0.02]
        elif tenure_days < 540:
            category_probabilities = [0.25, 0.60, 0.15]
        else:
            category_probabilities = [0.05, 0.65, 0.30]

        customers.append({
            "customer_id": customer_id,
            "name": fake.name(),
            "age": int(np.clip(rng.normal(38, 12), 18, 75)),
            "city": rng.choice(cities, p=city_probabilities),
            "registration_date": registration_date,
            "customer_category": rng.choice(
                ["New", "Regular", "VIP"], p=category_probabilities
            ),
        })

    return pd.DataFrame(customers)


def generate_products(rng):
    products = []
    product_id = 1

    for category, catalog in PRODUCT_CATALOG.items():
        for name, base_price in catalog:
            price = round(base_price * float(rng.uniform(0.92, 1.08)), 2)
            cost = round(price * float(rng.uniform(*COST_RATIO[category])), 2)

            products.append({
                "product_id": product_id,
                "product_name": name,
                "category": category,
                "selling_price": price,
                "cost": cost,
            })

            product_id += 1

    products = pd.DataFrame(products)

    if len(products) != PRODUCT_COUNT:
        raise ValueError("The product catalog does not match PRODUCT_COUNT.")

    return products


def generate_orders(customers, rng):
    days, day_probabilities = _daily_demand()

    order_days = np.sort(rng.choice(days, size=ORDER_COUNT, p=day_probabilities))

    category_activity = customers["customer_category"].map(
        {"New": 1.0, "Regular": 1.6, "VIP": 2.8}
    ).to_numpy()
    activity = category_activity * rng.lognormal(0, 0.5, size=len(customers))

    registration_dates = pd.to_datetime(customers["registration_date"]).to_numpy()
    customer_ids = customers["customer_id"].to_numpy()

    in_transit_from = pd.Timestamp(DATA_END_DATE) - pd.Timedelta(days=5)

    payment_methods = ["Card", "Cash", "Bank Transfer", "PayPal"]

    orders = []

    for order_id, order_day in enumerate(order_days, start=1):
        order_day = pd.Timestamp(order_day)

        eligible = registration_dates <= order_day.to_datetime64()
        weights = activity * eligible

        customer_id = int(rng.choice(customer_ids, p=weights / weights.sum()))

        if rng.random() < 0.06:
            status = "Cancelled"
        elif order_day >= in_transit_from:
            status = "Shipped"
        else:
            status = "Delivered"

        orders.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "order_date": order_day.date(),
            "status": status,
            "payment_method": rng.choice(
                payment_methods, p=[0.48, 0.27, 0.08, 0.17]
            ),
        })

    return pd.DataFrame(orders)


def generate_order_items(orders, products, rng):
    popularity = (
        products["selling_price"].to_numpy() ** -0.35
        * rng.lognormal(0, 0.3, size=len(products))
    )

    product_ids = products["product_id"].to_numpy()
    prices = products["selling_price"].to_numpy()
    categories = products["category"].to_numpy()

    order_items = []

    for order in orders.itertuples(index=False):
        order_date = order.order_date
        month_index = order_date.month - 1

        seasonal_weights = popularity * np.array([
            CATEGORY_SEASONALITY[category][month_index] for category in categories
        ])

        item_count = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.15, 0.10]))

        selected = rng.choice(
            len(product_ids),
            size=item_count,
            replace=False,
            p=seasonal_weights / seasonal_weights.sum(),
        )

        is_black_friday = BLACK_FRIDAY_WEEK[0] <= order_date <= BLACK_FRIDAY_WEEK[1]

        for index in selected:
            price = prices[index]

            if price < 40:
                extra_units = rng.poisson(0.8)
            elif price < 150:
                extra_units = rng.poisson(0.35)
            else:
                extra_units = rng.poisson(0.1)

            quantity = int(min(1 + extra_units, 6))

            if is_black_friday:
                discount = rng.uniform(0.15, 0.30)
            elif rng.random() < 0.25:
                discount = rng.uniform(0.05, 0.15)
            else:
                discount = 0.0

            unit_price = round(float(price * (1 - discount)), 2)

            order_items.append({
                "order_item_id": len(order_items) + 1,
                "order_id": int(order.order_id),
                "product_id": int(product_ids[index]),
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": round(quantity * unit_price, 2),
            })

    return pd.DataFrame(order_items)


def _compose_review(sentiment, product_id, rng):
    text = str(rng.choice(REVIEW_TEXTS[sentiment]))

    if rng.random() < 0.45:
        if rng.random() < 0.15:
            other = [name for name in SERVICE_COMMENTS if name != sentiment]
            comment_sentiment = str(rng.choice(other))
        else:
            comment_sentiment = sentiment

        text += " " + str(rng.choice(SERVICE_COMMENTS[comment_sentiment]))

    if rng.random() < 0.07:
        text += f" https://shop.example.bg/products/{product_id}"

    if rng.random() < 0.10:
        text += str(rng.choice(NOISE_SUFFIXES))

    if rng.random() < 0.03:
        text = text.upper()

    return text


def generate_reviews(order_items, orders, rng):
    product_quality = {
        product_id: rng.uniform(3.2, 4.6)
        for product_id in sorted(order_items["product_id"].unique())
    }

    eligible_orders = orders[orders["status"] == "Delivered"]

    eligible_items = order_items[
        order_items["order_id"].isin(eligible_orders["order_id"])
    ]

    if len(eligible_items) < REVIEW_COUNT:
        raise ValueError("Not enough delivered order items to generate reviews.")

    selected_items = eligible_items.sample(
        n=REVIEW_COUNT,
        random_state=RANDOM_SEED,
    )

    order_lookup = orders.set_index("order_id")

    reviews = []

    for review_id, item in enumerate(selected_items.itertuples(index=False), start=1):
        order = order_lookup.loc[item.order_id]
        order_date = order["order_date"]

        expected_rating = product_quality[item.product_id]

        # Peak season deliveries are slower, which lowers satisfaction
        if order_date >= BLACK_FRIDAY_WEEK[0]:
            expected_rating -= 0.5

        rating = int(np.clip(round(rng.normal(expected_rating, 1.0)), 1, 5))

        review_date = min(
            order_date + timedelta(days=int(rng.integers(3, 21))),
            DATA_END_DATE,
        )

        reviews.append({
            "review_id": review_id,
            "order_id": int(item.order_id),
            "customer_id": int(order["customer_id"]),
            "product_id": int(item.product_id),
            "review_date": review_date,
            "review_text": _compose_review(
                _rating_to_sentiment(rating), item.product_id, rng
            ),
            "rating": rating,
        })

    return pd.DataFrame(reviews)


def generate_dataset(seed=RANDOM_SEED):
    fake = Faker("bg_BG")
    fake.seed_instance(seed)

    rng = np.random.default_rng(seed)

    customers = generate_customers(fake, rng)
    products = generate_products(rng)
    orders = generate_orders(customers, rng)
    order_items = generate_order_items(orders, products, rng)
    reviews = generate_reviews(order_items, orders, rng)

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "reviews": reviews,
    }


def save_datasets(datasets):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataframe in datasets.items():
        dataframe.to_csv(
            RAW_DATA_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def introduce_data_quality_issues(datasets, seed=RANDOM_SEED):
    """Create controlled imperfections in copies of the generated datasets."""
    rng = np.random.default_rng(seed + 1)

    raw = {
        name: dataframe.copy()
        for name, dataframe in datasets.items()
    }

    def pick(dataframe, count):
        return rng.choice(dataframe.index.to_numpy(), size=count, replace=False)

    # Customers: missing values, invalid ages, inconsistent text and duplicates
    customers = raw["customers"]
    customers["age"] = customers["age"].astype("float")

    rows = pick(customers, 33)
    customers.loc[rows[:15], "city"] = np.nan
    customers.loc[rows[15:25], "age"] = np.nan
    customers.loc[rows[25:28], "age"] = [150, 2, 999]
    customers.loc[rows[28:33], "city"] = (
        customers.loc[rows[28:33], "city"].str.upper().radd("  ")
    )

    raw["customers"] = pd.concat(
        [customers, customers.loc[pick(customers, 5)]],
        ignore_index=True,
    )

    # Products: decimal comma in a numeric column
    products = raw["products"]
    products["selling_price"] = products["selling_price"].astype(str)
    rows = pick(products, 2)
    products.loc[rows, "selling_price"] = (
        products.loc[rows, "selling_price"].str.replace(".", ",", regex=False)
    )

    # Orders: mixed date formats, invalid dates, inconsistent labels, duplicates
    orders = raw["orders"]
    orders["order_date"] = orders["order_date"].astype(str)

    rows = pick(orders, 50)
    orders.loc[rows[:15], "order_date"] = (
        pd.to_datetime(orders.loc[rows[:15], "order_date"]).dt.strftime("%d.%m.%Y")
    )
    orders.loc[rows[15:18], "order_date"] = ["unknown", "2025-02-30", ""]
    orders.loc[rows[18:28], "status"] = (
        orders.loc[rows[18:28], "status"].str.lower().radd(" ")
    )
    orders.loc[rows[28:38], "payment_method"] = np.nan

    raw["orders"] = pd.concat(
        [orders, orders.loc[pick(orders, 8)]],
        ignore_index=True,
    )

    # Order items: extreme quantities, missing prices and duplicates
    items = raw["order_items"]
    rows = pick(items, 14)
    items.loc[rows[:8], "quantity"] = [120, 250, 999, 150, 300, 200, 500, 400]
    items.loc[rows[8:14], "unit_price"] = np.nan

    raw["order_items"] = pd.concat(
        [items, items.loc[pick(items, 6)]],
        ignore_index=True,
    )

    # Reviews: missing texts, out-of-range ratings and duplicates
    reviews = raw["reviews"]
    rows = pick(reviews, 9)
    reviews.loc[rows[:6], "review_text"] = np.nan
    reviews.loc[rows[6:9], "rating"] = [0, 6, 10]

    raw["reviews"] = pd.concat(
        [reviews, reviews.loc[pick(reviews, 4)]],
        ignore_index=True,
    )

    return raw


def main():
    generated_data = generate_dataset()
    raw_data = introduce_data_quality_issues(generated_data)

    save_datasets(raw_data)

    print("Dataset generation completed.\n")

    for name, dataframe in raw_data.items():
        print(f"{name}: {len(dataframe):,} records")

    print(f"\nCSV files saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
