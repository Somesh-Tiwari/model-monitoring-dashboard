"""Mark the comparison report as deployed once Jenkins starts the candidate image."""

import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPORT_PATH = APP_DIR / "artifacts" / "deployment_report.json"
report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
if not report.get("deployment_approved"):
    raise SystemExit("Refusing to mark a rejected candidate as deployed.")
report["deployment_status"] = "deployed"
temporary = REPORT_PATH.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
temporary.replace(REPORT_PATH)
print("Deployment status recorded.")
