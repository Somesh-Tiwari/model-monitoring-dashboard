"""Create the deterministic first production model for local development."""

import json
from pathlib import Path

import joblib

from app.modeling import train_model

model_path = Path(__file__).resolve().parent / "artifacts" / "production.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(train_model(), model_path)
model_path.with_name("production-version.json").write_text(
    json.dumps({"version": "fpl-bootstrap-seed-42"}, indent=2), encoding="utf-8"
)
print(f"Saved FPL total-points model to {model_path}")
