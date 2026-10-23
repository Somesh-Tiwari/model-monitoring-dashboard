"""Train a candidate, compare held-out point error with production, and gate deployment."""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.modeling import (
    DATASET_NAME,
    REPORT_SAMPLE_SIZE,
    SPLIT_RANDOM_STATE,
    load_split,
    model_matches_dataset,
    train_model,
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_PATH = ARTIFACT_DIR / "candidate.joblib"
IMAGE_MODEL_PATH = ARTIFACT_DIR / "production.joblib"
REPORT_PATH = ARTIFACT_DIR / "deployment_report.json"
PRODUCTION_PATH = Path(
    os.environ.get(
        "PRODUCTION_MODEL_PATH",
        str(Path(os.environ.get("JENKINS_HOME", str(APP_DIR.parent))) / "model-dashboard" / "production.joblib"),
    )
)
PRODUCTION_VERSION_PATH = PRODUCTION_PATH.with_name("production-version.json")
ACTIVE_VERSION_PATH = ARTIFACT_DIR / "production-version.json"


def score(model, x_suite, y_suite, players):
    predictions = [float(value) for value in model.predict(x_suite)]
    metrics = {
        "mae": round(float(mean_absolute_error(y_suite, predictions)), 3),
        "rmse": round(float(mean_squared_error(y_suite, predictions) ** 0.5), 3),
        "r2_score": round(float(r2_score(y_suite, predictions)), 4),
        "evaluated_inputs": len(y_suite),
    }
    examples = [
        {
            "player_id": players[index]["id"],
            "player_name": players[index]["name"],
            "actual_total_points": round(float(y_suite[index]), 2),
            "predicted_total_points": round(predictions[index], 2),
            "absolute_error": round(abs(float(y_suite[index]) - predictions[index]), 2),
        }
        for index in range(min(REPORT_SAMPLE_SIZE, len(y_suite)))
    ]
    return metrics, examples


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    _, _, x_validation, y_validation, x_test, y_test, _, validation_players, test_players = load_split()
    previous_model = None
    if PRODUCTION_PATH.exists():
        try:
            stored_model = joblib.load(PRODUCTION_PATH)
            if model_matches_dataset(stored_model):
                previous_model = stored_model
        except (OSError, EOFError, ValueError):
            previous_model = None
    bootstrap = previous_model is None
    if bootstrap:
        previous_model = train_model(random_state=SPLIT_RANDOM_STATE)
        previous_version = "fpl-bootstrap-seed-42"
        candidate_seed = SPLIT_RANDOM_STATE
    else:
        try:
            previous_version = json.loads(PRODUCTION_VERSION_PATH.read_text(encoding="utf-8"))["version"]
        except (OSError, json.JSONDecodeError, KeyError):
            previous_version = "previous-production"
        try:
            candidate_seed = int(os.environ.get("BUILD_NUMBER", "43"))
        except ValueError:
            candidate_seed = 43

    candidate_version = f"build-{os.environ.get('BUILD_NUMBER', candidate_seed)}"
    candidate_model = train_model(random_state=candidate_seed)
    previous_validation, _ = score(previous_model, x_validation, y_validation, validation_players)
    candidate_validation, _ = score(candidate_model, x_validation, y_validation, validation_players)
    previous_metrics, previous_inputs = score(previous_model, x_test, y_test, test_players)
    candidate_metrics, candidate_inputs = score(candidate_model, x_test, y_test, test_players)
    approved = candidate_validation["mae"] <= previous_validation["mae"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "validation_size": len(y_validation),
        "test_size": len(y_test),
        "decision": "deploy" if approved else "keep-production",
        "deployment_approved": approved,
        "deployment_status": "approved-pending-deployment" if approved else "production-retained",
        "gate_metric": "mae",
        "reason": "Candidate validation MAE is no higher than production. Final metrics are reported on a separate held-out test split." if approved else "Candidate validation MAE is higher than production; production was retained. Final test metrics are still recorded.",
        "previous_production": {"version": previous_version, "validation_metrics": previous_validation, "metrics": previous_metrics, "inputs": previous_inputs},
        "candidate": {"version": candidate_version, "validation_metrics": candidate_validation, "metrics": candidate_metrics, "inputs": candidate_inputs},
    }

    # Keep Jenkins' production copy and the model packaged into the image in sync only on approval.
    if approved:
        joblib.dump(candidate_model, CANDIDATE_PATH)
        shutil.copy2(CANDIDATE_PATH, IMAGE_MODEL_PATH)
        active_version = candidate_version
    else:
        if bootstrap:
            joblib.dump(previous_model, IMAGE_MODEL_PATH)
        else:
            shutil.copy2(PRODUCTION_PATH, IMAGE_MODEL_PATH)
        active_version = previous_version

    save_json(ACTIVE_VERSION_PATH, {"version": active_version})
    save_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2))
    print(f"Evaluation report: {REPORT_PATH}")
    if not approved:
        print("Deployment skipped: the existing production model remains active.")
        return 0
    print("Deployment approved: candidate MAE meets the production gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
