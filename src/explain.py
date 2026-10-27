"""SHAP helpers shared by notebook 03 and the dashboard.

SHAP runs on the fitted tree model (the pipeline's last step) using the
*transformed* matrix the trees actually see (scaled numerics). For display we
swap in the *unscaled* values (real K, rpm, Nm, min, W), so plots read in
physical units. Each SHAP value is the same either way, because scaling is a
per-column monotone map.
"""
import numpy as np
import pandas as pd
import shap

from src.config import NUM_COLS, PRETTY, TYPE_ORDER
from src.features import ENGINEERED, add_physics_features

MODEL_FEATURES = ["type"] + NUM_COLS + ENGINEERED


def display_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Model features in physical units, `type` as 0/1/2, in the model's column order."""
    D = add_physics_features(X)
    D["type"] = D["type"].map({t: i for i, t in enumerate(TYPE_ORDER)})
    return D[MODEL_FEATURES]


def make_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model.named_steps["clf"])


def explain(model, X: pd.DataFrame, explainer=None) -> shap.Explanation:
    """SHAP values for P(failure) (class 1), with physical-unit display data."""
    explainer = explainer or make_explainer(model)
    Xt = model[:-1].transform(X)
    e = explainer(Xt)
    values, base = e.values, np.asarray(e.base_values)
    if values.ndim == 3:            # RandomForest: (n, features, classes)
        values, base = values[:, :, 1], base[:, 1]
    return shap.Explanation(
        values=values, base_values=base, data=display_frame(X).values,
        feature_names=[PRETTY[f] for f in MODEL_FEATURES],
    )
