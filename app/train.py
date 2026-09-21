import os

import joblib
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier

DATASETS = {
    "model_1_wine.joblib": load_wine,
    "model_2_breast_cancer.joblib": load_breast_cancer,
    "model_3_digits.joblib": load_digits,
}

for filename, loader in DATASETS.items():
    dataset = loader()
    classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    classifier.fit(dataset.data, dataset.target)
    model_path = os.path.join(os.path.dirname(__file__), filename)
    joblib.dump(classifier, model_path)
    print(f"Saved {dataset.DESCR.splitlines()[0]} model to {model_path}")
