"""Fixed FPL player snapshot, held-out split, and total-points regressor."""

import csv
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

DATASET_NAME = "Fantasy Premier League 2026–2027"
DATASET_PATH = Path(__file__).resolve().parent / "data" / "fpl_2026_2027.csv"
TARGET_NAME = "total_points"
CATEGORICAL_FEATURES = ["position"]
NUMERIC_FEATURES = [
    "now_cost",
    "minutes",
    "starts",
    "expected_goal_involvements",
    "expected_goals",
    "expected_assists",
]
FEATURE_NAMES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
SPLIT_RANDOM_STATE = 42
REPORT_SAMPLE_SIZE = 12


def load_split():
    with DATASET_PATH.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))

    # Retain complete, valid observations for the selected features and target.
    clean_rows = []
    for row in rows:
        try:
            values = {name: float(row[name]) for name in NUMERIC_FEATURES}
            points = float(row[TARGET_NAME])
        except (KeyError, TypeError, ValueError):
            continue
        if not row.get("position", "").strip() or not row.get("id", "").strip():
            continue
        clean_rows.append((row, values, points))

    frame = pd.DataFrame([
        {"position": row["position"], **values}
        for row, values, _ in clean_rows
    ], columns=FEATURE_NAMES)
    target = [points for _, _, points in clean_rows]
    indices = list(range(len(clean_rows)))
    positions = [row[0]["position"] for row in clean_rows]
    train_indices, remainder_indices = train_test_split(
        indices,
        test_size=0.3,
        random_state=SPLIT_RANDOM_STATE,
        stratify=positions,
    )
    validation_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=0.5,
        random_state=SPLIT_RANDOM_STATE + 1,
        stratify=[positions[index] for index in remainder_indices],
    )
    x_train = frame.iloc[train_indices].reset_index(drop=True)
    y_train = [target[index] for index in train_indices]
    x_validation = frame.iloc[validation_indices].reset_index(drop=True)
    y_validation = [target[index] for index in validation_indices]
    x_test = frame.iloc[test_indices].reset_index(drop=True)
    y_test = [target[index] for index in test_indices]
    validation_players = [
        {"id": clean_rows[index][0]["id"], "name": clean_rows[index][0].get("web_name", "Player")}
        for index in validation_indices
    ]
    test_players = [
        {"id": clean_rows[index][0]["id"], "name": clean_rows[index][0].get("web_name", "Player")}
        for index in test_indices
    ]
    return x_train, y_train, x_validation, y_validation, x_test, y_test, FEATURE_NAMES, validation_players, test_players


def build_model(random_state: int = SPLIT_RANDOM_STATE):
    preprocessing = ColumnTransformer(
        transformers=[
            ("position", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def train_model(random_state: int = SPLIT_RANDOM_STATE):
    x_train, y_train, *_ = load_split()
    model = build_model(random_state=random_state)
    model.fit(x_train, y_train)
    return model


def model_matches_dataset(model):
    names = getattr(model, "feature_names_in_", None)
    return names is not None and list(names) == FEATURE_NAMES
