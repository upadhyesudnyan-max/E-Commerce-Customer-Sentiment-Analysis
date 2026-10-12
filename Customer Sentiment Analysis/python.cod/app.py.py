import re
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import nltk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


# ============================================================
# CUSTOMER SENTIMENT ANALYSIS
# STEP 3 - EDA AND TEXT PREPROCESSING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cleaned_data"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CUSTOMER SENTIMENT ANALYSIS")
print("STEP 3 - EDA AND TEXT PREPROCESSING")
print("=" * 60)


# ------------------------------------------------------------
# Download NLTK resources
# ------------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")


# ------------------------------------------------------------
# Load dataset with a robust fallback for raw/cleaned files
# ------------------------------------------------------------

def resolve_dataset_path():
    candidate_paths = [
        PROJECT_ROOT / "outputs" / "cleaned_data" / "cleaned_customer_sentiment.csv",
        PROJECT_ROOT / "Data" / "Ecommerce_Customer_Sentiment_csv.csv",
        PROJECT_ROOT / "Data" / "Ecommerce_Customer_Sentiment_Cleaned.csv",
        PROJECT_ROOT / "Data" / "Ecommerce_Customer_Sentiment.csv",
        PROJECT_ROOT / "Data" / "Ecommerce_Customer_Sentiment.csv.csv",
        PROJECT_ROOT / "customer_sentiment_cleaned.csv",
    ]

    file_path = next((path for path in candidate_paths if path.exists()), None)
    if file_path is not None:
        return file_path

    raise FileNotFoundError(
        "No dataset found. Expected a cleaned file in outputs/cleaned_data or a raw CSV in Data/."
    )


file_path = resolve_dataset_path()
df = pd.read_csv(file_path)

rename_map = {
    "review_text": "Review_Text",
    "sentiment": "Sentiment",
    "rating": "Rating",
    "complaint_type": "Complaint_Type",
    "brand": "Brand",
    "category": "Category",
    "review_date": "Review_Date",
    "sentiment_score": "Sentiment_Score",
}

df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

print("\nDataset loaded successfully from:", file_path)

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)


# ============================================================
# PART 1 - BASIC EDA
# ============================================================

print("\n")
print("=" * 60)
print("PART 1 - BASIC EDA")
print("=" * 60)


# ------------------------------------------------------------
# Dataset information
# ------------------------------------------------------------

print("\nDataset Information:")
print(df.info())


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\nMissing Values:")

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# Duplicate records
# ------------------------------------------------------------

print("\nDuplicate Rows:")

print(
    df.duplicated().sum()
)


# ============================================================
# PART 2 - SENTIMENT DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("SENTIMENT DISTRIBUTION")
print("=" * 60)


sentiment_counts = (
    df["Sentiment"]
    .value_counts()
)

print(
    sentiment_counts
)


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x="Sentiment"
)

plt.title(
    "Customer Sentiment Distribution"
)

plt.xlabel(
    "Sentiment"
)

plt.ylabel(
    "Number of Reviews"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "01_sentiment_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Sentiment share pie chart
# ------------------------------------------------------------

sentiment_colors = {
    "Positive": "#4CAF50",
    "Neutral": "#FFC107",
    "Negative": "#F44336",
}

plt.figure(
    figsize=(7, 7)
)

plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=[
        sentiment_colors.get(label, "#9E9E9E")
        for label in sentiment_counts.index
    ],
    wedgeprops={"edgecolor": "white"}
)

plt.title(
    "Customer Sentiment Share"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "12_sentiment_pie_chart.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 3 - RATING ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("RATING ANALYSIS")
print("=" * 60)


rating_counts = (
    df["Rating"]
    .value_counts()
    .sort_index()
)

print(
    rating_counts
)


print(
    "\nAverage Rating:",
    round(
        df["Rating"].mean(),
        2
    )
)


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x="Rating"
)

plt.title(
    "Customer Rating Distribution"
)

plt.xlabel(
    "Rating"
)

plt.ylabel(
    "Number of Reviews"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "02_rating_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 4 - TEXT PREPROCESSING
# ============================================================

print("\n")
print("=" * 60)
print("TEXT PREPROCESSING")
print("=" * 60)


# ------------------------------------------------------------
# Stop words
# ------------------------------------------------------------

stop_words = set(
    stopwords.words("english")
)


# ------------------------------------------------------------
# Lemmatizer
# ------------------------------------------------------------

lemmatizer = WordNetLemmatizer()


# ------------------------------------------------------------
# Text preprocessing function
# ------------------------------------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Stop-word removal
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Remove very short words
    words = [
        word
        for word in words
        if len(word) > 2
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return words


# ------------------------------------------------------------
# Apply preprocessing
# ------------------------------------------------------------

df["Tokens"] = (
    df["Review_Text"]
    .fillna("")
    .apply(preprocess_text)
)


# ------------------------------------------------------------
# Create cleaned text
# ------------------------------------------------------------

df["Clean_Review"] = (
    df["Tokens"]
    .apply(
        lambda words: " ".join(words)
    )
)


print("\nOriginal Review:")

print(
    df["Review_Text"].iloc[0]
)


print("\nProcessed Review:")

print(
    df["Clean_Review"].iloc[0]
)


# ------------------------------------------------------------
# Save NLP dataset
# ------------------------------------------------------------

output_path = OUTPUT_DIR / "nlp_reviews.csv"
saved_path = output_path
try:
    df.to_csv(
        output_path,
        index=False
    )
except PermissionError:
    recovery_path = OUTPUT_DIR / (
        f"nlp_reviews_{datetime.now():%Y%m%d_%H%M%S_%f}_recovery.csv"
    )
    try:
        df.to_csv(
            recovery_path,
            index=False
        )
    except PermissionError as recovery_exc:
        raise PermissionError(
            f"Cannot write the NLP dataset to {output_path} or "
            f"{recovery_path}. Close either file if it is open in another "
            "program and check write access to the output folder."
        ) from recovery_exc

    print(
        f"\nWARNING: {output_path} is locked or inaccessible. "
        f"The processed dataset was saved to {recovery_path}. "
        "Close the original file and rerun this script to update the "
        "standard nlp_reviews.csv."
    )
    saved_path = recovery_path


print(
    f"\nNLP processed dataset saved to: {saved_path}"
)


# ============================================================
# PART 5 - POSITIVE REVIEW TRENDS
# ============================================================

print("\n")
print("=" * 60)
print("POSITIVE REVIEW TRENDS")
print("=" * 60)


positive_df = df[
    df["Sentiment"] == "Positive"
]


positive_words = []

for words in positive_df["Tokens"]:

    positive_words.extend(
        words
    )


positive_word_counts = Counter(
    positive_words
)


print(
    "\nTop 20 Positive Words:"
)

for word, count in (
    positive_word_counts
    .most_common(20)
):

    print(
        word,
        ":",
        count
    )


# ------------------------------------------------------------
# Positive word chart
# ------------------------------------------------------------

positive_top = pd.DataFrame(
    positive_word_counts.most_common(15),
    columns=[
        "Word",
        "Frequency"
    ]
)


positive_top = positive_top.sort_values(
    "Frequency"
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    positive_top["Word"],
    positive_top["Frequency"]
)

plt.title(
    "Top Words in Positive Reviews"
)

plt.xlabel(
    "Frequency"
)

plt.ylabel(
    "Word"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "03_positive_words.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 6 - NEGATIVE REVIEW TRENDS
# ============================================================

print("\n")
print("=" * 60)
print("NEGATIVE REVIEW TRENDS")
print("=" * 60)


negative_df = df[
    df["Sentiment"] == "Negative"
]


negative_words = []

for words in negative_df["Tokens"]:

    negative_words.extend(
        words
    )


negative_word_counts = Counter(
    negative_words
)


print(
    "\nTop 20 Negative Words:"
)

for word, count in (
    negative_word_counts
    .most_common(20)
):

    print(
        word,
        ":",
        count
    )


# ------------------------------------------------------------
# Negative word chart
# ------------------------------------------------------------

negative_top = pd.DataFrame(
    negative_word_counts.most_common(15),
    columns=[
        "Word",
        "Frequency"
    ]
)


negative_top = negative_top.sort_values(
    "Frequency"
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    negative_top["Word"],
    negative_top["Frequency"]
)

plt.title(
    "Top Words in Negative Reviews"
)

plt.xlabel(
    "Frequency"
)

plt.ylabel(
    "Word"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "04_negative_words.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 7 - POSITIVE WORD CLOUD
# ============================================================

positive_text = " ".join(
    positive_words
)


if positive_text.strip():

    wordcloud_positive = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100
    ).generate(
        positive_text
    )


    plt.figure(
        figsize=(14, 7)
    )

    plt.imshow(
        wordcloud_positive,
        interpolation="bilinear"
    )

    plt.axis("off")

    plt.title(
        "Positive Review Word Cloud"
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "05_positive_wordcloud.png",
        dpi=300
    )

    plt.show()


# ============================================================
# PART 8 - NEGATIVE WORD CLOUD
# ============================================================

negative_text = " ".join(
    negative_words
)


if negative_text.strip():

    wordcloud_negative = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100
    ).generate(
        negative_text
    )


    plt.figure(
        figsize=(14, 7)
    )

    plt.imshow(
        wordcloud_negative,
        interpolation="bilinear"
    )

    plt.axis("off")

    plt.title(
        "Negative Review Word Cloud"
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "06_negative_wordcloud.png",
        dpi=300
    )

    plt.show()


# ============================================================
# PART 9 - COMPLAINT ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("COMPLAINT ANALYSIS")
print("=" * 60)


complaints = df[
    df["Complaint_Type"]
    != "No Complaint"
]


complaint_counts = (
    complaints["Complaint_Type"]
    .value_counts()
)


print(
    complaint_counts
)


plt.figure(
    figsize=(10, 6)
)

sns.countplot(
    data=complaints,
    y="Complaint_Type",
    order=(
        complaint_counts
        .head(10)
        .index
    )
)

plt.title(
    "Top Customer Complaints"
)

plt.xlabel(
    "Number of Complaints"
)

plt.ylabel(
    "Complaint Type"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "07_top_complaints.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Complaint share pie chart (top 6 + Other)
# ------------------------------------------------------------

top_complaints = complaint_counts.head(6)
other_total = complaint_counts.iloc[6:].sum()

if other_total > 0:
    top_complaints = pd.concat(
        [top_complaints, pd.Series({"Other": other_total})]
    )

plt.figure(
    figsize=(8, 8)
)

plt.pie(
    top_complaints.values,
    labels=top_complaints.index,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white"}
)

plt.title(
    "Share of Customer Complaints"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "13_complaint_pie_chart.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 10 - BRAND SENTIMENT
# ============================================================

print("\n")
print("=" * 60)
print("BRAND SENTIMENT ANALYSIS")
print("=" * 60)


brand_sentiment = pd.crosstab(
    df["Brand"],
    df["Sentiment"]
)


print(
    brand_sentiment
)


brand_sentiment.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Brand-wise Sentiment Distribution"
)

plt.xlabel(
    "Brand"
)

plt.ylabel(
    "Number of Reviews"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "08_brand_sentiment.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 11 - CATEGORY SENTIMENT
# ============================================================

category_sentiment = pd.crosstab(
    df["Category"],
    df["Sentiment"]
)


print(
    "\nCategory Sentiment:"
)

print(
    category_sentiment
)


category_sentiment.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Category-wise Sentiment Distribution"
)

plt.xlabel(
    "Category"
)

plt.ylabel(
    "Number of Reviews"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "09_category_sentiment.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 11B - HEATMAPS
# ============================================================

# ------------------------------------------------------------
# Brand vs Sentiment heatmap (counts)
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

sns.heatmap(
    brand_sentiment,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title(
    "Brand vs Sentiment Heatmap"
)

plt.xlabel(
    "Sentiment"
)

plt.ylabel(
    "Brand"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "14_brand_sentiment_heatmap.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Category vs Sentiment heatmap (row percentages)
# ------------------------------------------------------------

category_sentiment_pct = (
    category_sentiment
    .div(category_sentiment.sum(axis=1), axis=0)
    * 100
)

plt.figure(
    figsize=(9, 6)
)

sns.heatmap(
    category_sentiment_pct,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "% of reviews in category"}
)

plt.title(
    "Category vs Sentiment Heatmap (%)"
)

plt.xlabel(
    "Sentiment"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "15_category_sentiment_heatmap.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Correlation heatmap (numeric columns)
# ------------------------------------------------------------

numeric_cols = [
    col for col in [
        "Rating",
        "Sentiment_Score",
        "purchase_amount",
        "previous_purchases",
    ]
    if col in df.columns
]

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "16_correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 12 - SENTIMENT SCORE
# ============================================================

print("\n")
print("=" * 60)
print("SENTIMENT SCORE ANALYSIS")
print("=" * 60)


print(
    df["Sentiment_Score"].describe()
)


plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    df["Sentiment_Score"],
    bins=30,
    kde=True
)

plt.title(
    "Sentiment Score Distribution"
)

plt.xlabel(
    "Sentiment Score"
)

plt.ylabel(
    "Number of Reviews"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "10_sentiment_score.png",
    dpi=300
)

plt.show()


# ============================================================
# PART 13 - SENTIMENT VS RATING
# ============================================================

sentiment_rating = (
    df.groupby("Sentiment")["Rating"]
    .mean()
)


print(
    "\nAverage Rating by Sentiment:"
)

print(
    sentiment_rating.round(2)
)


# ============================================================
# PART 14 - MONTHLY SENTIMENT TREND
# ============================================================

df["Review_Date"] = pd.to_datetime(
    df["Review_Date"],
    errors="coerce"
)


monthly_sentiment = pd.crosstab(
    df["Review_Date"].dt.to_period("M"),
    df["Sentiment"]
)


monthly_sentiment.plot(
    figsize=(12, 6)
)

plt.title(
    "Monthly Customer Sentiment Trend"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Number of Reviews"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "11_monthly_sentiment.png",
    dpi=300
)

plt.show()


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("STEP 3 COMPLETED")
print("=" * 60)

print(
    "Text preprocessing completed."
)

print(
    "Tokenization completed."
)

print(
    "Stop-word removal completed."
)

print(
    "Positive trends identified."
)

print(
    "Negative trends identified."
)

print(
    "Charts saved in outputs/charts/"
)