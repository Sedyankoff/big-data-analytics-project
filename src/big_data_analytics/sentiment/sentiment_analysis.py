import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from big_data_analytics.config import PROCESSED_DATA_DIR


BULGARIAN_STOP_WORDS = [
    "и", "в", "на", "за", "от", "с", "се", "е", "да",
    "но", "че", "по", "не", "го", "я", "ги", "ми",
    "си", "като", "при", "това", "той", "тя", "те",
    "бих", "беше", "съм", "са", "има", "много",
]

POSITIVE_WORDS = [
    "доволен", "доволна", "чудесно", "отлично",
    "добър", "добра", "качество", "препоръчвам",
    "бърза", "прекрасен", "прекрасна", "оправда",
]

NEGATIVE_WORDS = [
    "недоволен", "недоволна", "проблем", "лошо",
    "лош", "лоша", "очакванията", "подобри",
    "не бих", "не отговаря",
]


def load_reviews():
    reviews = pd.read_csv(
        PROCESSED_DATA_DIR / "reviews.csv"
    )

    reviews["review_date"] = pd.to_datetime(
        reviews["review_date"]
    )

    return reviews


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def classify_sentiment(text):
    negative_phrases = [
        "не съм доволен",
        "не съм доволна",
        "не бих поръчал",
        "не бих поръчала",
        "не отговаря",
    ]

    if any(phrase in text for phrase in negative_phrases):
        return "Negative"

    words = set(text.split())

    positive_score = sum(
        word in words for word in POSITIVE_WORDS
    )

    negative_score = sum(
        word in words for word in NEGATIVE_WORDS
    )

    if positive_score > negative_score:
        return "Positive"

    if negative_score > positive_score:
        return "Negative"

    return "Neutral"


def analyze_word_frequencies(reviews):
    all_words = []

    for text in reviews["cleaned_text"]:
        words = [
            word for word in text.split()
            if word not in BULGARIAN_STOP_WORDS
        ]
        all_words.extend(words)

    word_counts = Counter(all_words)

    print("\nMOST COMMON WORDS")
    print("=" * 45)

    for word, count in word_counts.most_common(15):
        print(f"{word}: {count}")

    return word_counts


def analyze_ngrams(reviews):
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        stop_words=BULGARIAN_STOP_WORDS,
    )

    matrix = vectorizer.fit_transform(
        reviews["cleaned_text"]
    )

    frequencies = matrix.sum(axis=0).A1
    phrases = vectorizer.get_feature_names_out()

    phrase_counts = sorted(
        zip(phrases, frequencies),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nMOST COMMON BIGRAMS")
    print("=" * 45)

    for phrase, count in phrase_counts[:10]:
        print(f"{phrase}: {count}")


def analyze_tfidf(reviews):
    vectorizer = TfidfVectorizer(
        stop_words=BULGARIAN_STOP_WORDS,
        max_features=20,
    )

    matrix = vectorizer.fit_transform(
        reviews["cleaned_text"]
    )

    scores = matrix.mean(axis=0).A1
    words = vectorizer.get_feature_names_out()

    tfidf_results = sorted(
        zip(words, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nTOP TF-IDF TERMS")
    print("=" * 45)

    for word, score in tfidf_results[:10]:
        print(f"{word}: {score:.4f}")


def analyze_sentiment(reviews):
    reviews["sentiment"] = reviews["cleaned_text"].apply(
        classify_sentiment
    )

    sentiment_counts = reviews["sentiment"].value_counts()

    sentiment_percentages = (
        reviews["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
    )

    print("\nSENTIMENT DISTRIBUTION")
    print("=" * 45)

    for sentiment, percentage in sentiment_percentages.items():
        count = sentiment_counts[sentiment]
        print(f"{sentiment}: {count} reviews ({percentage:.1f}%)")

    return reviews


def create_visualizations(reviews, word_counts):
    # Most common words
    common_words = word_counts.most_common(10)

    words, counts = zip(*common_words)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words))
    plt.title("Most Common Words in Customer Reviews")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Sentiment distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=reviews,
        x="sentiment",
        order=["Positive", "Neutral", "Negative"],
    )
    plt.title("Customer Sentiment Distribution")
    plt.tight_layout()
    plt.show()

    # Sentiment over time
    monthly_sentiment = (
        reviews.groupby([
            pd.Grouper(key="review_date", freq="ME"),
            "sentiment",
        ])
        .size()
        .unstack(fill_value=0)
    )

    monthly_sentiment.plot(
        kind="line",
        marker="o",
        figsize=(11, 5),
    )

    plt.title("Customer Sentiment Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.show()


def run_sentiment_analysis():
    reviews = load_reviews()

    reviews["cleaned_text"] = reviews["review_text"].apply(
        clean_text
    )

    word_counts = analyze_word_frequencies(reviews)

    analyze_ngrams(reviews)
    analyze_tfidf(reviews)

    reviews = analyze_sentiment(reviews)

    reviews.to_csv(
        PROCESSED_DATA_DIR / "reviews_analyzed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_visualizations(reviews, word_counts)

    print("\nAnalyzed reviews saved successfully.")


if __name__ == "__main__":
    run_sentiment_analysis()
