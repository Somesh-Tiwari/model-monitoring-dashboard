"""Train a candidate, compare it with production, and promote only on a pass."""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.modeling import DATASET_NAME, SPLIT_RANDOM_STATE, load_split, train_model

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


def score(model, x_suite, y_suite, class_names):
    predictions = model.predict(x_suite)
    metrics = {
        "accuracy": round(float(accuracy_score(y_suite, predictions)), 4),
        "precision": round(float(precision_score(y_suite, predictions, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_suite, predictions, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_suite, predictions, average="macro", zero_division=0)), 4),
        "evaluated_inputs": len(y_suite),
    }
    examples = [
        {
            "input_id": f"wine-test-{index + 1}",
            "expected_class": class_names[int(actual)],
            "predicted_class": class_names[int(predicted)],
            "correct": bool(actual == predicted),
        }
        for index, (actual, predicted) in enumerate(zip(y_suite, predictions))
    ]
    return metrics, examples


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    _, _, x_suite, y_suite, _, class_names = load_split()
    bootstrap = not PRODUCTION_PATH.exists()
    if bootstrap:
        previous_model = train_model(random_state=SPLIT_RANDOM_STATE)
        previous_version = "bootstrap-seed-42"
        candidate_seed = SPLIT_RANDOM_STATE
    else:
        previous_model = joblib.load(PRODUCTION_PATH)
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
    previous_metrics, previous_inputs = score(previous_model, x_suite, y_suite, class_names)
    candidate_metrics, candidate_inputs = score(candidate_model, x_suite, y_suite, class_names)
    approved = candidate_metrics["accuracy"] >= previous_metrics["accuracy"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "evaluation_suite_size": len(y_suite),
        "decision": "deploy" if approved else "keep-production",
        "deployment_approved": approved,
        "deployment_status": "approved-pending-deployment" if approved else "production-retained",
        "reason": "Candidate accuracy meets or exceeds production accuracy on the fixed evaluation suite." if approved else "Candidate accuracy is lower than production accuracy on the fixed evaluation suite; production was retained.",
        "previous_production": {"version": previous_version, "metrics": previous_metrics, "inputs": previous_inputs},
        "candidate": {"version": candidate_version, "metrics": candidate_metrics, "inputs": candidate_inputs},
    }

    # Keep Jenkins' production copy and the model packaged into the image in sync only on approval.
    if approved:
        joblib.dump(candidate_model, CANDIDATE_PATH)
        shutil.copy2(CANDIDATE_PATH, IMAGE_MODEL_PATH)
        active_version = candidate_version
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
    print("Deployment approved: candidate meets the production accuracy gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
