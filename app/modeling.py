"""Shared fixed dataset, split, and classifier configuration."""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_NAME = "Wine"
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
EVALUATION_INPUT_COUNT = 4


def load_split():
    dataset = load_wine()
    indices = list(range(len(dataset.target)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=dataset.target,
    )
    x_train = dataset.data[train_indices]
    y_train = dataset.target[train_indices]
    suite_indices = test_indices[:EVALUATION_INPUT_COUNT]
    x_suite = dataset.data[suite_indices]
    y_suite = dataset.target[suite_indices]
    return x_train, y_train, x_suite, y_suite, list(dataset.feature_names), list(dataset.target_names)


def train_model(random_state: int = SPLIT_RANDOM_STATE):
    x_train, y_train, *_ = load_split()
    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    classifier.fit(x_train, y_train)
    return classifier
