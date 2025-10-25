"""
setup_project.py
Creates a lean project scaffold for the 'med nlp classification proj' folder.
Run this with your venv active:
    python setup_project.py
"""

import os
import textwrap
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).parent.resolve()

# desired structure
dirs = [
    "data",
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "outputs",
    "logs",
]

files = {
    "README.md": textwrap.dedent(
        """\
        # Med NLP Classification Project

        Short project overview:
        - Goal: classify medical transcription text into medical specialties.
        - Data: `data/mtsamples.csv`
        - Notebooks: `notebooks/01_data_exploration.ipynb`, `notebooks/02_modelling.ipynb`
        - Scripts: `src/preprocess.py`, `src/train_model.py`, `src/evaluate.py`

        ## Quick start
        1. Activate your venv
        2. `pip install -r requirements.txt`
        3. Place `mtsamples.csv` in `data/`
        4. Open `notebooks/01_data_exploration.ipynb` in VS Code or Jupyter
        """
    ),
    ".gitignore": textwrap.dedent(
        """\
        # Python
        __pycache__/
        *.py[cod]
        *.so
        .Python
        venv/
        env/
        .env
        # Jupyter
        .ipynb_checkpoints
        # Data and models
        data/
        models/
        outputs/
        logs/
        """
    ),
    "requirements.txt": textwrap.dedent(
        """\
        pandas
        numpy
        scikit-learn
        matplotlib
        seaborn
        jupyter
        nltk
        spacy
        tqdm
        transformers
        """
    ),
    "src/__init__.py": "",
    "src/preprocess.py": textwrap.dedent(
        """\
        # src/preprocess.py
        # Small helper functions for loading and cleaning the mtsamples.csv file.

        import pandas as pd
        import re

        def load_raw(path: str):
            df = pd.read_csv(path)
            return df

        def basic_clean_text(text: str) -> str:
            if pd.isna(text):
                return ''
            text = text.lower()
            text = re.sub(r'\\s+', ' ', text)
            # remove weird characters but keep basic punctuation
            text = re.sub(r'[^a-z0-9 .,;:\\-()\\n]', '', text)
            return text.strip()

        if __name__ == '__main__':
            import sys
            p = sys.argv[1] if len(sys.argv) > 1 else '../data/mtsamples.csv'
            df = load_raw(p)
            print('Rows:', len(df))
            print('Columns:', df.columns.tolist())
        """
    ),
    "src/train_model.py": textwrap.dedent(
        """\
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
        """
    ),
    "src/evaluate.py": textwrap.dedent(
        """\
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
        """
    ),
    "notebooks/README.txt": "Create Jupyter notebooks here. Start with 01_data_exploration.ipynb",
}

def create_structure():
    for d in dirs:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"created dir: {p}")

    for fname, content in files.items():
        fp = ROOT / fname
        if not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            with fp.open('w', encoding='utf-8') as f:
                f.write(content)
            print(f"created file: {fp}")
        else:
            print(f"already exists: {fp}")

    # move mtsamples.csv into data/ if it exists in the root or elsewhere nearby
    possible_names = ['mtsamples.csv', 'mtsamples.csv.zip', 'mtsamples (1).csv', 'medicaltranscriptions.csv']
    found = False
    for name in possible_names:
        candidate = ROOT / name
        if candidate.exists():
            dest = ROOT / 'data' / name
            shutil.move(str(candidate), str(dest))
            print(f"moved {candidate} -> {dest}")
            found = True
            break

    if not found:
        print("Note: no mtsamples.csv found in project root. Please place the file in data/ or root and re-run this script.")

    print("\nScaffold complete. Next steps (displayed in README):")
    print("- Activate venv, then `pip install -r requirements.txt`")
    print("- Open notebooks/01_data_exploration.ipynb in VS Code and begin EDA")

if __name__ == '__main__':
    create_structure()
