# src/evaluate.py
# Simple evaluation helper to load a saved model and run predictions.

import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate(model_path: str, data_path: str, text_col='transcription', target_col='medical_specialty'):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df = df[[text_col, target_col]].dropna()
    X = df[text_col].astype(str)
    y = df[target_col].astype(str)
    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/tfidf_lr_baseline.joblib'
    data_path = sys.argv[2] if len(sys.argv) > 2 else '../data/mtsamples.csv'
    evaluate(model_path, data_path)
