# ==========================================
# CUSTOMER SENTIMENT ANALYSIS
# STEP 4 - LINEAR REGRESSION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("outputs/cleaned_data/nlp_reviews.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. REMOVE EMPTY REVIEWS
# ==========================================

df = df.dropna(subset=["Clean_Review"])

df = df[df["Clean_Review"].str.strip() != ""]

print("Data after cleaning:", df.shape)


# ==========================================
# 3. SELECT INPUT AND TARGET
# ==========================================

X_text = df["Clean_Review"]

y = df["Sentiment_Score"]

print("\nInput: Customer Review")
print("Target: Sentiment Score")


# ==========================================
# 4. CONVERT TEXT INTO NUMBERS
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(X_text)

print("\nTF-IDF completed")
print("TF-IDF Shape:", X.shape)


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape[0])
print("Testing Data:", X_test.shape[0])


# ==========================================
# 6. CREATE LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print("\nModel training completed")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. CHECK MODEL PERFORMANCE
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))


# ==========================================
# 9. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Sentiment Score")
plt.ylabel("Predicted Sentiment Score")

plt.title("Actual vs Predicted Sentiment Score")

plt.tight_layout()

plt.savefig(
    "outputs/charts/actual_vs_predicted.png"
)

plt.show()


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "outputs/models/linear_regression_model.pkl"
)

joblib.dump(
    vectorizer,
    "outputs/models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully")


# ==========================================
# 11. TEST NEW REVIEWS
# ==========================================

reviews = [
    "The product quality is excellent and I am very happy.",
    "Very poor quality and completely disappointed.",
    "The product is average and works normally."
]


# Convert reviews into TF-IDF
review_vectors = vectorizer.transform(reviews)


# Predict sentiment score
predictions = model.predict(review_vectors)


print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")


for review, prediction in zip(reviews, predictions):

    print("\nReview:", review)

    print(
        "Predicted Sentiment Score:",
        round(prediction, 4)
    )


print("\nAnalysis Completed!")
