import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['title', 'text', 'label'])
    df['label_binary'] = df['label'].map({'REAL': 0, 'FAKE': 1})
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(df):
    df = df.copy()
    df['title_clean'] = df['title'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    df['combined'] = df['title_clean'] + ' ' + df['text_clean']
    return df


def build_and_train(X_train, y_train):
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    return vectorizer, model


def evaluate(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 50)

    return y_pred


def save_model(model, vectorizer, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/logistic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Model and vectorizer saved to '{output_dir}/'")


def load_model(model_dir='model'):
    with open(f'{model_dir}/logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{model_dir}/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_news(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = 'FAKE' if pred == 1 else 'REAL'
    confidence = proba[pred] * 100
    return label, confidence


def main():
    print("Loading dataset...")
    df = load_data('data/fake_or_real_news.csv')
    print(f"Dataset loaded: {len(df)} articles | REAL: {(df['label']=='REAL').sum()} | FAKE: {(df['label']=='FAKE').sum()}")

    print("\nPreprocessing text...")
    df = preprocess(df)

    X = df['combined']
    y = df['label_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("\nTraining Logistic Regression model...")
    vectorizer, model = build_and_train(X_train, y_train)

    print("\nEvaluating model...")
    evaluate(model, vectorizer, X_test, y_test)

    save_model(model, vectorizer)

    print("\n--- Sample Predictions ---")
    samples = [
        "NASA confirms water found on Mars surface in major discovery",
        "BREAKING: Secret deep-state ritual exposed by anonymous insider - government hiding truth about aliens",
        "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
        "Scientists SHOCKED: This one weird trick cures cancer overnight, doctors DON'T want you to know"
    ]

    for headline in samples:
        label, conf = predict_news(headline, model, vectorizer)
        print(f"[{label}] ({conf:.1f}%) → {headline[:70]}...")


if __name__ == '__main__':
    main()
