import pandas as pd
import mysql.connector
import re
from collections import Counter


# ============================================================
# 1. FILE PATH
# ============================================================

CSV_FILE = r"Data\Ecommerce_Customer_Sentiment_csv.csv"


# ============================================================
# 2. MYSQL SETTINGS
# ============================================================

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "system"
DATABASE_NAME = "ecommerce_sentiment"


# ============================================================
# 3. READ CSV FILE
# ============================================================

print("\nLoading CSV file...")

df = pd.read_csv(CSV_FILE)

print("CSV loaded successfully!")
print("Total rows:", len(df))
print("Total columns:", len(df))

print("\nColumns in dataset:")
print(df.columns.tolist())


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# Convert date column
if "review_date" in df.columns:
    df["review_date"] = pd.to_datetime(
        df["review_date"],
        errors="coerce"
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

# Remove completely empty rows
df = df.dropna(how="all")

# Remove duplicate reviews
if "review_id" in df.columns:
    df = df.drop_duplicates(subset=["review_id"])

# Fill missing text values
text_columns = [
    "review_text",
    "sentiment",
    "customer_type",
    "state",
    "city",
    "purchase_channel",
    "payment_method",
    "complaint_type"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].fillna("Unknown")


# ============================================================
# 6. CONNECT TO MYSQL
# ============================================================

print("\nConnecting to MySQL...")

connection = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD
)

cursor = connection.cursor()

print("MySQL connection successful!")


# ============================================================
# 7. CREATE DATABASE
# ============================================================

cursor.execute(
    f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"
)

print(f"Database '{DATABASE_NAME}' is ready.")

cursor.close()
connection.close()


# ============================================================
# 8. CONNECT TO THE DATABASE
# ============================================================

connection = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=DATABASE_NAME
)

cursor = connection.cursor()

print("Connected to ecommerce_sentiment database.")


# ============================================================
# 9. CREATE PRODUCTS TABLE
# ============================================================

print("\nCreating products table...")

cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    brand VARCHAR(100)
)
""")


# ============================================================
# 10. CREATE REVIEWS TABLE
# ============================================================

print("Creating reviews table...")

cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    product_id VARCHAR(50),
    review_text TEXT,
    review_date DATE,
    sentiment VARCHAR(30),
    sentiment_score DECIMAL(10,6),
    customer_type VARCHAR(50),
    state VARCHAR(100),
    city VARCHAR(100),
    previous_purchases INT,
    purchase_channel VARCHAR(100),
    payment_method VARCHAR(100),

    FOREIGN KEY (product_id)
    REFERENCES products(product_id)
)
""")


# ============================================================
# 11. CREATE RATINGS TABLE
# ============================================================

print("Creating ratings table...")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ratings (
    rating_id INT AUTO_INCREMENT PRIMARY KEY,
    review_id VARCHAR(50),
    product_id VARCHAR(50),
    rating INT,
    purchase_amount DECIMAL(15,2),

    FOREIGN KEY (review_id)
    REFERENCES reviews(review_id),

    FOREIGN KEY (product_id)
    REFERENCES products(product_id)
)
""")


# ============================================================
# 12. CREATE CUSTOMER COMPLAINTS TABLE
# ============================================================

print("Creating customer_complaints table...")

cursor.execute("""
CREATE TABLE IF NOT EXISTS customer_complaints (
    complaint_id INT AUTO_INCREMENT PRIMARY KEY,
    review_id VARCHAR(50),
    customer_id VARCHAR(50),
    complaint_type VARCHAR(150),

    FOREIGN KEY (review_id)
    REFERENCES reviews(review_id)
)
""")


connection.commit()

print("\nAll tables created successfully!")


# ============================================================
# 13. INSERT PRODUCTS
# ============================================================

print("\nInserting products...")

products = df[
    [
        "product_id",
        "product_name",
        "category",
        "brand"
    ]
].drop_duplicates(subset=["product_id"])


product_query = """
INSERT IGNORE INTO products
(
    product_id,
    product_name,
    category,
    brand
)
VALUES (%s, %s, %s, %s)
"""


product_count = 0

for _, row in products.iterrows():

    cursor.execute(
        product_query,
        (
            row["product_id"],
            row["product_name"],
            row["category"],
            row["brand"]
        )
    )

    product_count += 1


print("Products inserted:", product_count)


# ============================================================
# 14. INSERT REVIEWS
# ============================================================

print("\nInserting reviews...")

review_query = """
INSERT IGNORE INTO reviews
(
    review_id,
    customer_id,
    product_id,
    review_text,
    review_date,
    sentiment,
    sentiment_score,
    customer_type,
    state,
    city,
    previous_purchases,
    purchase_channel,
    payment_method
)
VALUES
(
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s
)
"""


review_count = 0


for _, row in df.iterrows():

    review_date = row["review_date"]

    if pd.isna(review_date):
        review_date = None
    else:
        review_date = review_date.date()

    cursor.execute(
        review_query,
        (
            row["review_id"],
            row["customer_id"],
            row["product_id"],
            row["review_text"],
            review_date,
            row["sentiment"],
            row["sentiment_score"],
            row["customer_type"],
            row["state"],
            row["city"],
            row["previous_purchases"],
            row["purchase_channel"],
            row["payment_method"]
        )
    )

    review_count += 1


print("Reviews inserted:", review_count)


# ============================================================
# 15. INSERT RATINGS
# ============================================================

print("\nInserting ratings...")

rating_query = """
INSERT INTO ratings
(
    review_id,
    product_id,
    rating,
    purchase_amount
)
VALUES (%s, %s, %s, %s)
"""


rating_count = 0


for _, row in df.iterrows():

    rating_value = row["rating"]

    if pd.isna(rating_value):
        rating_value = None
    else:
        rating_value = int(rating_value)

    purchase_amount = row["purchase_amount"]

    if pd.isna(purchase_amount):
        purchase_amount = None

    cursor.execute(
        rating_query,
        (
            row["review_id"],
            row["product_id"],
            rating_value,
            purchase_amount
        )
    )

    rating_count += 1


print("Ratings inserted:", rating_count)


# ============================================================
# 16. INSERT CUSTOMER COMPLAINTS
# ============================================================

print("\nInserting customer complaints...")

complaint_query = """
INSERT INTO customer_complaints
(
    review_id,
    customer_id,
    complaint_type
)
VALUES (%s, %s, %s)
"""


complaint_count = 0


for _, row in df.iterrows():

    complaint_type = row["complaint_type"]

    # Insert only actual complaints
    if (
        pd.notna(complaint_type)
        and str(complaint_type).strip() != ""
        and str(complaint_type).lower() != "unknown"
        and str(complaint_type).lower() != "none"
    ):

        cursor.execute(
            complaint_query,
            (
                row["review_id"],
                row["customer_id"],
                complaint_type
            )
        )

        complaint_count += 1


print("Complaints inserted:", complaint_count)


# ============================================================
# 17. COMMIT ALL DATA
# ============================================================

connection.commit()

print("\n==========================================")
print("DATA SUCCESSFULLY STORED IN MYSQL")
print("==========================================")


# ============================================================
# 18. BASIC DATABASE CHECK
# ============================================================

print("\nDATABASE RECORD COUNTS")
print("------------------------------------------")


cursor.execute(
    "SELECT COUNT(*) FROM products"
)

print(
    "Products:",
    cursor.fetchone()[0]
)


cursor.execute(
    "SELECT COUNT(*) FROM reviews"
)

print(
    "Reviews:",
    cursor.fetchone()[0]
)


cursor.execute(
    "SELECT COUNT(*) FROM ratings"
)

print(
    "Ratings:",
    cursor.fetchone()[0]
)


cursor.execute(
    "SELECT COUNT(*) FROM customer_complaints"
)

print(
    "Complaints:",
    cursor.fetchone()[0]
)


# ============================================================
# 19. SENTIMENT ANALYSIS
# ============================================================

print("\nSENTIMENT ANALYSIS")
print("------------------------------------------")


sentiment_query = """
SELECT
    sentiment,
    COUNT(*) AS review_count
FROM reviews
GROUP BY sentiment
ORDER BY review_count DESC
"""


cursor.execute(sentiment_query)

sentiment_results = cursor.fetchall()


for row in sentiment_results:

    print(
        f"{row[0]} : {row[1]}"
    )


# ============================================================
# 20. NEGATIVE REVIEW COUNT
# ============================================================

cursor.execute("""
SELECT COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
""")


negative_count = cursor.fetchone()[0]


print("\nTotal Negative Reviews:", negative_count)


# ============================================================
# 21. MOST COMMON WORDS IN NEGATIVE REVIEWS
# ============================================================

print("\n==========================================")
print("MOST COMMON WORDS IN NEGATIVE REVIEWS")
print("==========================================")


negative_reviews = df[
    df["sentiment"]
    .astype(str)
    .str.lower()
    .str.strip()
    == "negative"
]["review_text"].dropna()


# Combine reviews
all_negative_text = " ".join(
    negative_reviews.astype(str)
)


# Convert to lowercase
all_negative_text = all_negative_text.lower()


# Extract words
words = re.findall(
    r"\b[a-zA-Z]+\b",
    all_negative_text
)


# ============================================================
# 22. STOP WORDS
# ============================================================

stop_words = {
    "the",
    "is",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "this",
    "that",
    "it",
    "was",
    "with",
    "but",
    "not",
    "my",
    "very",
    "i",
    "have",
    "has",
    "had",
    "be",
    "are",
    "were",
    "they",
    "you",
    "we",
    "so",
    "as",
    "from",
    "at",
    "by",
    "product",
    "item",
    "one",
    "would",
    "could",
    "really",
    "just",
    "get",
    "got",
    "like",
    "also",
    "there",
    "their",
    "your",
    "our",
    "me",
    "us",
    "he",
    "she",
    "them",
    "do",
    "does",
    "did",
    "been",
    "can",
    "will",
    "too"
}


# Remove stop words
filtered_words = []

for word in words:

    if word not in stop_words:

        if len(word) > 2:

            filtered_words.append(word)


# ============================================================
# 23. COUNT WORD FREQUENCY
# ============================================================

word_counts = Counter(filtered_words)


# ============================================================
# 24. DISPLAY TOP 20 WORDS
# ============================================================

print("\nTop 20 Negative Review Words:")
print("------------------------------------------")


top_words = word_counts.most_common(20)


for word, count in top_words:

    print(
        f"{word:<20} {count}"
    )


# ============================================================
# 25. SAVE WORD ANALYSIS TO CSV
# ============================================================

word_data = pd.DataFrame(
    top_words,
    columns=[
        "word",
        "frequency"
    ]
)


word_data.to_csv(
    "negative_review_common_words.csv",
    index=False
)


print(
    "\nWord analysis saved to:"
    " negative_review_common_words.csv"
)


# ============================================================
# 26. PRODUCT ANALYSIS
# ============================================================

print("\n==========================================")
print("PRODUCTS WITH MOST NEGATIVE REVIEWS")
print("==========================================")


product_negative_query = """
SELECT
    p.product_name,
    COUNT(r.review_id) AS negative_reviews
FROM reviews r
JOIN products p
    ON r.product_id = p.product_id
WHERE LOWER(r.sentiment) = 'negative'
GROUP BY p.product_id, p.product_name
ORDER BY negative_reviews DESC
LIMIT 10
"""


cursor.execute(product_negative_query)

product_results = cursor.fetchall()


for row in product_results:

    print(
        f"{row[0]} : {row[1]} negative reviews"
    )


# ============================================================
# 27. CATEGORY ANALYSIS
# ============================================================

print("\n==========================================")
print("CATEGORIES WITH MOST NEGATIVE REVIEWS")
print("==========================================")


category_query = """
SELECT
    p.category,
    COUNT(r.review_id) AS negative_reviews
FROM reviews r
JOIN products p
    ON r.product_id = p.product_id
WHERE LOWER(r.sentiment) = 'negative'
GROUP BY p.category
ORDER BY negative_reviews DESC
"""


cursor.execute(category_query)

category_results = cursor.fetchall()


for row in category_results:

    print(
        f"{row[0]} : {row[1]} negative reviews"
    )


# ============================================================
# 28. COMPLAINT ANALYSIS
# ============================================================

print("\n==========================================")
print("CUSTOMER COMPLAINT ANALYSIS")
print("==========================================")


complaint_query = """
SELECT
    complaint_type,
    COUNT(*) AS complaint_count
FROM customer_complaints
GROUP BY complaint_type
ORDER BY complaint_count DESC
"""


cursor.execute(complaint_query)

complaint_results = cursor.fetchall()


for row in complaint_results:

    print(
        f"{row[0]} : {row[1]}"
    )


# ============================================================
# 29. RATING ANALYSIS
# ============================================================

print("\n==========================================")
print("RATING ANALYSIS")
print("==========================================")


rating_analysis_query = """
SELECT
    r.sentiment,
    COUNT(rt.rating) AS total_ratings,
    ROUND(AVG(rt.rating), 2) AS average_rating
FROM reviews r
JOIN ratings rt
    ON r.review_id = rt.review_id
GROUP BY r.sentiment
ORDER BY average_rating
"""


cursor.execute(rating_analysis_query)

rating_results = cursor.fetchall()


for row in rating_results:

    print(
        f"Sentiment: {row[0]} | "
        f"Reviews: {row[1]} | "
        f"Average Rating: {row[2]}"
    )


# ============================================================
# 30. NEGATIVE REVIEWS BY STATE
# ============================================================

print("\n==========================================")
print("NEGATIVE REVIEWS BY STATE")
print("==========================================")


state_query = """
SELECT
    state,
    COUNT(*) AS negative_reviews
FROM reviews
WHERE LOWER(sentiment) = 'negative'
GROUP BY state
ORDER BY negative_reviews DESC
"""


cursor.execute(state_query)

state_results = cursor.fetchall()


for row in state_results:

    print(
        f"{row[0]} : {row[1]}"
    )


# ============================================================
# 31. TOP COMPLAINT TYPES FROM NEGATIVE REVIEWS
# ============================================================

print("\n==========================================")
print("COMPLAINT TYPES FROM NEGATIVE REVIEWS")
print("==========================================")


negative_complaint_query = """
SELECT
    c.complaint_type,
    COUNT(*) AS complaint_count
FROM customer_complaints c
JOIN reviews r
    ON c.review_id = r.review_id
WHERE LOWER(r.sentiment) = 'negative'
GROUP BY c.complaint_type
ORDER BY complaint_count DESC
"""


cursor.execute(negative_complaint_query)

negative_complaint_results = cursor.fetchall()


for row in negative_complaint_results:

    print(
        f"{row[0]} : {row[1]}"
    )


# ============================================================
# 32. CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()


print("\n==========================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("==========================================")
print("\nYour e-commerce sentiment data is now")
print("stored in MySQL and ready for analysis.")