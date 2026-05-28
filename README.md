# Fake News Detection — Logistic Regression

A Python-based binary text classification system that detects fake news using Logistic Regression and TF-IDF vectorization.

## Project Structure

```
fake-news-detection/
├── data/
│   └── fake_or_real_news.csv       # Dataset (6335 articles)
├── model/
│   ├── logistic_model.pkl          # Saved trained model
│   └── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
├── fake_news_detector.py           # Main script
└── README.md
```

## Dataset

**Size:** 6335 articles — 3171 REAL, 3164 FAKE  
**Columns:** `id`, `title`, `text`, `label`

## Tech Stack

- **Python 3.x**
- **Scikit-learn** — TfidfVectorizer, LogisticRegression
- **Pandas** — data loading and preprocessing
- **Pickle** — model serialization


## Run

```bash
python fake_news_detector.py
```

## Pipeline

1. Load CSV dataset
2. Clean text (lowercase, remove URLs, punctuation, numbers)
3. Combine title + article body
4. TF-IDF vectorization (50k features, bigrams, sublinear TF)
5. Train Logistic Regression (80/20 split, stratified)
6. Evaluate with accuracy, classification report, confusion matrix
7. Save model + vectorizer as `.pkl` files
8. Run sample predictions on new headlines

## Results

| Metric   | Score  |
|----------|--------|
| Accuracy | ~98%   |
| Precision| ~98%   |
| Recall   | ~98%   |
| F1-Score | ~98%   |
