
from datetime import timedelta

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


fake = Faker("bg_BG")
fake.seed_instance(RANDOM_SEED)

rng = np.random.default_rng(RANDOM_SEED)


def generate_customers():
    customers = []

    cities = [
        "София", "Пловдив", "Варна", "Бургас",
        "Русе", "Стара Загора", "Плевен", "Смолян",
        "Велико Търново", "Благоевград",
    ]

    categories = ["New", "Regular", "VIP"]

    for customer_id in range(1, CUSTOMER_COUNT + 1):
        registration_date = fake.date_between(
            start_date="-2y",
            end_date=DATA_END_DATE,
        )

        customers.append({
            "customer_id": customer_id,
            "name": fake.name(),
            "age": int(rng.integers(18, 76)),
            "city": rng.choice(cities),
            "registration_date": registration_date,
            "customer_category": rng.choice(
                categories, p=[0.50, 0.35, 0.15]
            ),
        })

    return pd.DataFrame(customers)


def generate_products():
    product_catalog = {
        "Electronics": ["Headphones", "Smartphone", "Monitor", "Keyboard", "Mouse"],
        "Clothing": ["T-shirt", "Jeans", "Jacket", "Sneakers", "Hoodie"],
        "Home": ["Lamp", "Chair", "Desk", "Pillow", "Blanket"],
        "Sports": ["Dumbbells", "Yoga Mat", "Backpack", "Bottle", "Resistance Bands"],
        "Books": ["Novel", "Cookbook", "Biography", "Notebook", "Dictionary"],
        "Beauty": ["Shampoo", "Face Cream", "Perfume", "Soap", "Body Lotion"],
    }

    products = []
    product_id = 1

    for category, names in product_catalog.items():
        for name in names:
            price = round(float(rng.uniform(10, 1000)), 2)
            cost = round(price * float(rng.uniform(0.45, 0.80)), 2)

            products.append({
                "product_id": product_id,
                "product_name": name,
                "category": category,
                "selling_price": price,
                "cost": cost,
            })

            product_id += 1

    return pd.DataFrame(products)


def generate_orders(customers):
    orders = []

    statuses = ["Delivered", "Shipped", "Cancelled"]
    payment_methods = ["Card", "Cash", "Bank Transfer", "PayPal"]

    for order_id in range(1, ORDER_COUNT + 1):
        customer = customers.iloc[
            int(rng.integers(0, len(customers)))
        ]

        registration_date = pd.Timestamp(customer["registration_date"])
        earliest_date = max(
            pd.Timestamp(DATA_START_DATE),
            registration_date,
        )

        available_days = (DATA_END_DATE - earliest_date.date()).days

        order_date = earliest_date + timedelta(
            days=int(rng.integers(0, available_days + 1))
        )

        orders.append({
            "order_id": order_id,
            "customer_id": int(customer["customer_id"]),
            "order_date": order_date,
            "status": rng.choice(
                statuses, p=[0.90, 0.05, 0.05]
            ),
            "payment_method": rng.choice(payment_methods),
        })

    return pd.DataFrame(orders)


def generate_order_items(orders, products):
    order_items = []

    for _, order in orders.iterrows():
        item_count = int(rng.integers(1, 5))

        selected_products = rng.choice(
            products["product_id"].to_numpy(),
            size=item_count,
            replace=False,
        )

        for product_id in selected_products:
            product = products.loc[
                products["product_id"] == product_id
            ].iloc[0]

            quantity = int(rng.integers(1, 5))
            unit_price = round(
                float(product["selling_price"]) *
                float(rng.uniform(0.90, 1.00)),
                2,
            )

            order_items.append({
                "order_item_id": len(order_items) + 1,
                "order_id": int(order["order_id"]),
                "product_id": int(product_id),
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": round(quantity * unit_price, 2),
            })

    return pd.DataFrame(order_items)


def generate_reviews(order_items, orders):
    review_templates = {
        "positive": [
            "Много съм доволен от продукта. Отлично качество!",
            "Всичко е чудесно, бих поръчал отново.",
            "Продуктът напълно оправда очакванията ми.",
            "Бърза доставка и отлично качество.",
            "Препоръчвам го на всеки!",
        ],
        "neutral": [
            "Продуктът е нормален, нищо особено.",
            "Качеството е приемливо за цената.",
            "Очаквах малко повече, но като цяло става.",
            "Добър продукт, но има какво да се подобри.",
        ],
        "negative": [
            "Не съм доволен от качеството.",
            "Продуктът не отговаря на очакванията ми.",
            "Имаше проблем с поръчката.",
            "Не бих поръчал този продукт отново.",
        ],
    }

    ratings = rng.choice(
        [1, 2, 3, 4, 5],
        size=REVIEW_COUNT,
        p=[0.05, 0.10, 0.15, 0.35, 0.35],
    )

    eligible_orders = orders[
        orders["status"] == "Delivered"
    ]

    eligible_items = order_items[
        order_items["order_id"].isin(eligible_orders["order_id"])
    ].copy()

    if len(eligible_items) < REVIEW_COUNT:
        raise ValueError("Not enough delivered order items to generate reviews.")

    selected_items = eligible_items.sample(
        n=REVIEW_COUNT,
        random_state=RANDOM_SEED,
    )

    order_lookup = orders.set_index("order_id")

    reviews = []

    for review_id, (_, item) in enumerate(selected_items.iterrows(), start=1):
        order = order_lookup.loc[item["order_id"]]

        rating = int(ratings[review_id - 1])

        if rating >= 4:
            sentiment = "positive"
        elif rating == 3:
            sentiment = "neutral"
        else:
            sentiment = "negative"

        order_date = pd.Timestamp(order["order_date"])
        max_review_date = min(
            order_date + timedelta(days=30),
            pd.Timestamp(DATA_END_DATE),
        )

        review_date = order_date + timedelta(
            days=int(rng.integers(0, (max_review_date - order_date).days + 1))
        )

        reviews.append({
            "review_id": review_id,
            "order_id": int(item["order_id"]),
            "customer_id": int(order["customer_id"]),
            "product_id": int(item["product_id"]),
            "review_date": review_date,
            "review_text": rng.choice(review_templates[sentiment]),
            "rating": rating,
        })

    return pd.DataFrame(reviews)


def generate_dataset():
    customers = generate_customers()
    products = generate_products()
    orders = generate_orders(customers)
    order_items = generate_order_items(orders, products)
    reviews = generate_reviews(order_items, orders)

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


def introduce_data_quality_issues(datasets):
    """Create controlled imperfections in copies of the generated datasets."""
    raw_datasets = {
        name: dataframe.copy()
        for name, dataframe in datasets.items()
    }

    raw_datasets["customers"].loc[0, "city"] = np.nan
    raw_datasets["reviews"].loc[0, "review_text"] = np.nan

    raw_datasets["customers"] = pd.concat(
        [raw_datasets["customers"], raw_datasets["customers"].iloc[[0]]],
        ignore_index=True,
    )

    raw_datasets["orders"]["order_date"] = (
        raw_datasets["orders"]["order_date"].astype(str)
    )
    raw_datasets["orders"].loc[0, "order_date"] = "invalid-date"

    raw_datasets["order_items"].loc[0, "quantity"] = 999

    return raw_datasets


if __name__ == "__main__":
    generated_data = generate_dataset()
    raw_data = introduce_data_quality_issues(generated_data)

    save_datasets(raw_data)

    print("Dataset generation completed.\n")

    for name, dataframe in raw_data.items():
        print(f"{name}: {len(dataframe):,} records")

    print(f"\nCSV files saved to: {RAW_DATA_DIR}")