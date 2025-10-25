# src/train_model.py
# Baseline training: TF-IDF + LogisticRegression example (scikit-learn)

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def train(data_path: str, target_col: str = 'medical_specialty', text_col: str = 'transcription'):
    df = pd.read_csv(data_path)
    df = df[[text_col, target_col]].dropna()
    X = df[text_col].astype(str)
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(pipe, 'models/tfidf_lr_baseline.joblib')
    print('Saved model to models/tfidf_lr_baseline.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/mtsamples.csv')
    args = parser.parse_args()
    train(args.data)
