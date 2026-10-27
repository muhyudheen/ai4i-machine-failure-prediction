"""Project-wide constants: paths, column names, random seed.

Everything that more than one notebook / the dashboard needs lives here so a
change (e.g. the seed) propagates everywhere.
"""
from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_CSV = DATA_DIR / "ai4i2020.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

UCI_ID = 601

# The raw column names differ between sources:
#   * ucimlrepo (id=601)         -> "UID", "Air temperature", ...   (no units)
#   * original UCI ai4i2020.csv  -> "UDI", "Air temperature [K]", ...
# We normalise both to short snake_case names with the unit kept in the name.
RENAME = {
    "UID": "uid",
    "UDI": "uid",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature": "air_temp_k",
    "Air temperature [K]": "air_temp_k",
    "Process temperature": "process_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed": "rot_speed_rpm",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque": "torque_nm",
    "Torque [Nm]": "torque_nm",
    "Tool wear": "tool_wear_min",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

TARGET = "machine_failure"
ID_COLS = ["uid", "product_id"]
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_MODE_NAMES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

CAT_COLS = ["type"]
TYPE_ORDER = ["L", "M", "H"]  # product quality variant: low < medium < high
NUM_COLS = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]

PRETTY = {
    "type": "Type",
    "air_temp_k": "Air temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "rot_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "machine_failure": "Machine failure",
    "power_w": "Power [W]",
    "temp_diff_k": "Temp. difference [K]",
    "overstrain_nm_min": "Tool wear × Torque [min·Nm]",
}
