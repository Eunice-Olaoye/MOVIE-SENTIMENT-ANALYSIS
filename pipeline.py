import pandas as pd
import numpy as np
import string
import joblib
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  1. Load Dataset 
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df


# 2. Clean Text 
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r"\d+", '', text)
    return text.strip()

def preprocess_data(df):
    df['cleaned_review'] = df['review'].apply(clean_text)
    return df


# 3. Vectorization
def vectorize_text(df, max_features=5000, use_bigrams=False):
    ngram = (1, 2) if use_bigrams else (1, 1)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']
    return X, y, vectorizer


#  4. Split Data 
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


#  5. Train Model 
def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


#  6. Evaluate Model 
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    return y_pred


#  7. Error Analysis 
def error_analysis(df, y_test, y_pred):
    results_df = pd.DataFrame({
        'review': df.iloc[y_test.index].reset_index(drop=True)['review'],
        'true_label': y_test.reset_index(drop=True),
        'predicted_label': pd.Series(y_pred)
    })

    false_pos = results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)]
    false_neg = results_df[(results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)]

    print("\n--- False Positives (5) ---")
    print(false_pos['review'].head(5))
    print("\n--- False Negatives (5) ---")
    print(false_neg['review'].head(5))


# 8. Main Runner
def main():
    filepath = r"C:\Users\HP\Desktop\New folder\IMDB Dataset.csv"  # Adjust path if needed

    # Load and clean data
    df = load_data(filepath)
    df = preprocess_data(df)

    # Vectorize text
    X, y, vectorizer = vectorize_text(df, max_features=5000, use_bigrams=True)

    # Split and train
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_model(X_train, y_train)

    # Evaluate
    y_pred = evaluate_model(clf, X_test, y_test)

    # Error analysis
    error_analysis(df, y_test, y_pred)

    # Save model and vectorizer to the same directory as this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "logistic_model.pkl")
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")


#  9. Run 
if __name__ == "__main__":
    main()

# 10. Notes
# - Ensure required libraries are installed: pandas, numpy, scikit-learn, joblib
