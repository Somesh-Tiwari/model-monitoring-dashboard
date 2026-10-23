"""Persist the candidate as production after Jenkins has deployed successfully."""

import os
import shutil
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CANDIDATE_PATH = APP_DIR / "artifacts" / "candidate.joblib"
ACTIVE_VERSION_PATH = APP_DIR / "artifacts" / "production-version.json"
PRODUCTION_PATH = Path(
    os.environ.get(
        "PRODUCTION_MODEL_PATH",
        str(Path(os.environ.get("JENKINS_HOME", str(APP_DIR.parent))) / "model-dashboard" / "production.joblib"),
    )
)

if not CANDIDATE_PATH.exists():
    raise SystemExit("Candidate model artifact is missing; production model was not changed.")

PRODUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary_path = PRODUCTION_PATH.with_suffix(".joblib.tmp")
shutil.copy2(CANDIDATE_PATH, temporary_path)
temporary_path.replace(PRODUCTION_PATH)
if ACTIVE_VERSION_PATH.exists():
    production_version_path = PRODUCTION_PATH.with_name("production-version.json")
    temporary_version_path = production_version_path.with_suffix(".json.tmp")
    temporary_version_path.write_text(ACTIVE_VERSION_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    temporary_version_path.replace(production_version_path)
print(f"Promoted candidate to production model: {PRODUCTION_PATH}")
