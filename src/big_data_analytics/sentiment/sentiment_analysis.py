import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from big_data_analytics.config import PROCESSED_DATA_DIR
from big_data_analytics.plotting import PALETTE, SENTIMENT_COLORS, save_figure


SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

BULGARIAN_STOP_WORDS = [
    "и", "в", "на", "за", "от", "с", "се", "е", "да",
    "но", "че", "по", "не", "го", "я", "ги", "ми",
    "си", "като", "при", "това", "той", "тя", "те",
    "бих", "беше", "съм", "са", "има", "много", "след",
    "ще", "колко", "какво", "както", "точно", "цяла",
]

# Word stems: a word matches when it starts with one of the stems
POSITIVE_WORDS = [
    "доволен", "доволна", "чудесн", "отличн", "страхот",
    "препоръчв", "перфектн", "оправда", "бърз", "идеалн",
    "добр", "добър", "отново", "отговаря", "ниво",
]

NEGATIVE_WORDS = [
    "недовол", "проблем", "лош", "повред", "счупи",
    "ужасн", "разочаров", "бавн", "закъсн", "никой",
]

NEGATIONS = {"не", "нито", "без"}


def load_reviews():
    reviews = pd.read_csv(
        PROCESSED_DATA_DIR / "reviews.csv"
    )

    reviews["review_date"] = pd.to_datetime(
        reviews["review_date"]
    )

    return reviews


def remove_urls(text):
    return re.sub(r"http\S+|www\S+", " ", str(text))


def clean_text(text):
    text = remove_urls(text).lower()
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    return [
        word for word in clean_text(text).split()
        if word not in BULGARIAN_STOP_WORDS and len(word) > 1
    ]


def word_polarity(word):
    if any(word.startswith(stem) for stem in NEGATIVE_WORDS):
        return -1

    if any(word.startswith(stem) for stem in POSITIVE_WORDS):
        return 1

    return 0


def sentiment_score(text):
    """Lexicon score; a negation reverses the polarity until the end of the clause."""
    text = remove_urls(text).lower()
    score = 0

    for clause in re.split(r"[.!?,;:]+|\bно\b", text):
        negated = False

        for word in re.findall(r"[а-яa-z]+", clause):
            if word in NEGATIONS:
                negated = True
                continue

            polarity = word_polarity(word)
            score += -polarity if negated else polarity

    return score


def classify_sentiment(text):
    score = sentiment_score(text)

    if score > 0:
        return "Positive"

    if score < 0:
        return "Negative"

    return "Neutral"


def rating_sentiment(rating):
    if rating >= 4:
        return "Positive"

    if rating == 3:
        return "Neutral"

    return "Negative"


def analyze_word_frequencies(reviews):
    word_counts = Counter(
        word
        for text in reviews["review_text"]
        for word in tokenize(text)
    )

    print("\nMOST COMMON WORDS")
    print("=" * 45)

    for word, count in word_counts.most_common(15):
        print(f"{word}: {count}")

    return word_counts


def analyze_ngrams(reviews):
    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
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

    print("\nMOST COMMON N-GRAMS (CountVectorizer)")
    print("=" * 45)

    for phrase, count in phrase_counts[:10]:
        print(f"{phrase}: {count}")

    return phrase_counts


def analyze_tfidf(reviews):
    vectorizer = TfidfVectorizer(
        stop_words=BULGARIAN_STOP_WORDS,
    )

    matrix = vectorizer.fit_transform(
        reviews["cleaned_text"]
    )

    words = vectorizer.get_feature_names_out()
    scores = pd.DataFrame(matrix.toarray(), columns=words, index=reviews.index)

    print("\nTOP TF-IDF TERMS")
    print("=" * 45)

    overall = scores.mean().sort_values(ascending=False)
    for word, score in overall.head(10).items():
        print(f"{word}: {score:.4f}")

    print("\nTOP TF-IDF TERMS BY SENTIMENT")
    print("=" * 45)

    for sentiment in SENTIMENT_ORDER:
        mask = reviews["sentiment"] == sentiment
        if mask.any():
            top_terms = scores[mask].mean().sort_values(ascending=False).head(6)
            print(f"{sentiment}: {', '.join(top_terms.index)}")

    return overall


def analyze_sentiment(reviews):
    reviews["sentiment_score"] = reviews["review_text"].apply(sentiment_score)
    reviews["sentiment"] = reviews["review_text"].apply(classify_sentiment)
    reviews["rating_sentiment"] = reviews["rating"].apply(rating_sentiment)

    sentiment_counts = reviews["sentiment"].value_counts().reindex(
        SENTIMENT_ORDER, fill_value=0
    )
    sentiment_percentages = sentiment_counts / sentiment_counts.sum() * 100

    print("\nSENTIMENT DISTRIBUTION")
    print("=" * 45)

    for sentiment, percentage in sentiment_percentages.items():
        count = sentiment_counts[sentiment]
        print(f"{sentiment}: {count} reviews ({percentage:.1f}%)")

    monthly_share = (
        pd.crosstab(
            reviews["review_date"].dt.to_period("M"),
            reviews["sentiment"],
            normalize="index",
        )
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
        .mul(100)
    )

    print("\nMONTHLY SENTIMENT SHARE (%)")
    print("=" * 45)
    print(monthly_share.round(1))

    return reviews, monthly_share


def evaluate_against_ratings(reviews):
    accuracy = accuracy_score(reviews["rating_sentiment"], reviews["sentiment"])

    matrix = pd.DataFrame(
        confusion_matrix(
            reviews["rating_sentiment"],
            reviews["sentiment"],
            labels=SENTIMENT_ORDER,
        ),
        index=[f"rating: {label}" for label in SENTIMENT_ORDER],
        columns=[f"text: {label}" for label in SENTIMENT_ORDER],
    )

    print("\nVALIDATION AGAINST STAR RATINGS")
    print("=" * 45)
    print(f"Agreement between text sentiment and rating: {accuracy * 100:.1f}%")
    print(matrix)

    return accuracy, matrix


def print_interpretation(reviews, accuracy):
    shares = reviews["sentiment"].value_counts(normalize=True).mul(100)

    monthly_positive = (
        reviews.groupby(reviews["review_date"].dt.to_period("M"))["sentiment"]
        .apply(lambda values: (values == "Positive").mean() * 100)
    )

    disagreements = reviews[reviews["sentiment"] != reviews["rating_sentiment"]]

    print("\nINTERPRETATION")
    print("=" * 45)
    print(f"- {shares.get('Positive', 0):.1f}% of the reviews are positive and "
          f"{shares.get('Negative', 0):.1f}% are negative.")
    print(f"- The share of positive reviews is highest in "
          f"{monthly_positive.idxmax()} ({monthly_positive.max():.1f}%) and lowest in "
          f"{monthly_positive.idxmin()} ({monthly_positive.min():.1f}%).")
    print(f"- The text classifier agrees with the star rating in {accuracy * 100:.1f}% "
          f"of the reviews; {len(disagreements)} reviews differ.")

    print("\nLIMITATIONS")
    print("=" * 45)
    print("- The classifier is a rule-based lexicon for Bulgarian; words outside the "
          "lexicon are ignored.")
    print("- Mixed reviews (e.g. good product, late delivery) receive a single label, "
          "which explains most disagreements with the rating.")
    print("- Negation is handled only within a clause; sarcasm, irony and emojis are "
          "not interpreted.")
    print("- The reviews are synthetic, so the vocabulary is smaller than in real data "
          "and the accuracy would be lower on real customer texts.")


def create_visualizations(reviews, word_counts, phrase_counts, monthly_share, matrix):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    words, counts = zip(*word_counts.most_common(12))
    axes[0].barh(list(words)[::-1], list(counts)[::-1], color=PALETTE[0])
    axes[0].set_title("Most common words in customer reviews")
    axes[0].set_xlabel("Frequency")

    top_phrases = phrase_counts[:12]
    axes[1].barh([phrase for phrase, _ in top_phrases][::-1],
                 [count for _, count in top_phrases][::-1], color=PALETTE[0])
    axes[1].set_title("Most common phrases (n-grams)")
    axes[1].set_xlabel("Frequency")

    save_figure(fig, "13_review_words_and_phrases")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 2.2, 1.3]})

    counts = reviews["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    axes[0].bar(counts.index, counts.values,
                color=[SENTIMENT_COLORS[label] for label in counts.index])
    axes[0].set_title("Sentiment distribution")
    axes[0].set_ylabel("Number of reviews")

    bottom = None
    labels = monthly_share.index.strftime("%b")
    for sentiment in SENTIMENT_ORDER:
        axes[1].bar(labels, monthly_share[sentiment], bottom=bottom,
                    color=SENTIMENT_COLORS[sentiment], label=sentiment,
                    edgecolor="white", linewidth=1)
        bottom = monthly_share[sentiment] if bottom is None else bottom + monthly_share[sentiment]
    axes[1].set_title("Monthly sentiment share (%)")
    axes[1].set_ylabel("Share of reviews, %")
    axes[1].legend(loc="lower left")

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2])
    axes[2].set_title("Text sentiment vs star rating")

    save_figure(fig, "14_sentiment_analysis")


def run_sentiment_analysis():
    reviews = load_reviews()

    reviews["cleaned_text"] = reviews["review_text"].apply(
        clean_text
    )

    word_counts = analyze_word_frequencies(reviews)
    phrase_counts = analyze_ngrams(reviews)

    reviews, monthly_share = analyze_sentiment(reviews)

    analyze_tfidf(reviews)

    accuracy, matrix = evaluate_against_ratings(reviews)
    print_interpretation(reviews, accuracy)

    reviews.to_csv(
        PROCESSED_DATA_DIR / "reviews_analyzed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_visualizations(reviews, word_counts, phrase_counts, monthly_share, matrix)

    print("\nAnalyzed reviews saved successfully.")


if __name__ == "__main__":
    run_sentiment_analysis()
