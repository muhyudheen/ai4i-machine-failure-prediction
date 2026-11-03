"""Load the trained model and score new machine runs.

    pip install -r requirements.txt      # exact versions: the model was saved with them
    python predict_example.py

The saved model (models/final_model.joblib) is a complete scikit-learn pipeline:
physics features -> encoding/scaling -> Random Forest. Give it the six raw inputs
and it returns P(failure). It needs the `src/` folder next to it, because the
pipeline's first step (PhysicsFeatures) is our own code.

Input columns (one row per run):
    type            product quality: "L", "M" or "H"
    air_temp_k      air temperature [K]
    process_temp_k  process temperature [K]
    rot_speed_rpm   rotational speed [rpm]
    torque_nm       torque [Nm]
    tool_wear_min   tool wear [min]
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))           # lets the saved pipeline find src.features

from src.config import FAILURE_MODE_NAMES  # noqa: E402  (after the path fix on purpose)

model = joblib.load(ROOT / "models" / "final_model.joblib")
mode_models = joblib.load(ROOT / "models" / "mode_models.joblib")
meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
THRESHOLD = meta["threshold"]           # 0.056: chosen to catch >= 90% of failures
RISK_BUDGET = 0.01                      # design limit used for the operating ranges

runs = pd.DataFrame([
    # a typical, healthy setting
    {"type": "L", "air_temp_k": 300.1, "process_temp_k": 310.1, "rot_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100},
    # worn tool under heavy load -> overstrain
    {"type": "L", "air_temp_k": 298.4, "process_temp_k": 308.2, "rot_speed_rpm": 1282, "torque_nm": 60.7, "tool_wear_min": 216},
    # small temperature gap at low speed -> heat dissipation
    {"type": "L", "air_temp_k": 302.4, "process_temp_k": 310.2, "rot_speed_rpm": 1331, "torque_nm": 45.5, "tool_wear_min": 53},
    # very high power (high torque at low speed) -> power failure
    {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1250, "torque_nm": 72.0, "tool_wear_min": 30},
])

p_fail = model.predict_proba(runs)[:, 1]
p_modes = pd.DataFrame({m: mm.predict_proba(runs)[:, 1] for m, mm in mode_models.items()})

out = runs.copy()
out["P(failure)"] = p_fail.round(4)
out["alarm"] = p_fail >= THRESHOLD
out["status"] = ["ALARM" if p >= THRESHOLD else "elevated" if p > RISK_BUDGET else "within budget" for p in p_fail]
out["most likely mode"] = [FAILURE_MODE_NAMES[row.idxmax()] if row.max() >= RISK_BUDGET else "none expected"
                           for _, row in p_modes.iterrows()]

print(f"Model: {meta['selected']}  |  CV PR-AUC {meta['cv_headline_pr_auc'][0]:.3f} ± {meta['cv_headline_pr_auc'][1]:.3f}"
      f"  |  alarm threshold {THRESHOLD:.3f}\n")
pd.set_option("display.width", 200)
print(out.to_string(index=False))
