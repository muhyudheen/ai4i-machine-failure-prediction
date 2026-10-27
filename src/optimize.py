"""Optimisation layer: risk surfaces and recommended operating ranges.

The decision variables are the controllable process parameters, rotational
speed and torque (the operating point), plus tool wear (set by the
tool-change interval). Air and process temperature are treated as ambient
conditions and fixed at their medians unless stated otherwise.

Method (exhaustive grid, no black-box optimiser needed in 2-D):
1. Predict P(failure) on a dense rpm × torque grid for each product Type.
2. Keep only grid cells **supported by data**: at least `min_count` training runs
   within a small radius. Outside the observed band (e.g. 2,800 rpm at 70 Nm)
   a tree model just extrapolates its edge leaves, so we do not recommend
   anything there.
3. A cell counts as "safe" if P(failure) <= RISK_BUDGET at every tool-wear level
   from 0 up to the recommended tool-change limit.

Why a risk budget and not the alarm threshold (0.056)? They answer different
questions. The alarm threshold is tuned to catch 90% of failures run by run,
and it sits on the ~5% plateau created by random tool-wear failures. Using it
as a design limit would label 230 min of tool wear "acceptable". The budget
asks a design question instead: which settings keep risk near the healthy
baseline (~0.1%)? Notebook 03 shows the conclusions are the same for budgets
between 0.5% and 2%.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import NearestNeighbors

from src.config import TYPE_ORDER

RISK_BUDGET = 0.01
RPM_GRID = np.linspace(1150, 2900, 141)      # 12.5 rpm steps
TORQUE_GRID = np.linspace(3, 78, 151)        # 0.5 Nm steps
WEAR_GRID = np.arange(0, 254, 2)


def power_w(rpm, torque):
    return np.asarray(torque) * np.asarray(rpm) * 2 * np.pi / 60


def conditions(X: pd.DataFrame) -> dict:
    """Default (median) values for the parameters not being varied."""
    med = X.median(numeric_only=True)
    return {"air_temp_k": med.air_temp_k, "process_temp_k": med.process_temp_k,
            "rot_speed_rpm": med.rot_speed_rpm, "torque_nm": med.torque_nm,
            "tool_wear_min": med.tool_wear_min}


def predict_settings(model, typ: str, **cols) -> np.ndarray:
    """P(failure) for every row of a broadcast set of settings (scalars or arrays)."""
    n = max(np.size(v) for v in cols.values())
    frame = pd.DataFrame({"type": [typ] * n, **{k: np.broadcast_to(v, n) for k, v in cols.items()}})
    frame = frame[["type", "air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]]
    return model.predict_proba(frame)[:, 1]


def risk_grid(model, typ: str, base: dict, rpm=RPM_GRID, torque=TORQUE_GRID, **overrides) -> np.ndarray:
    """P(failure) on the rpm × torque grid -> array of shape (len(torque), len(rpm))."""
    R, T = np.meshgrid(rpm, torque)
    cols = {**base, **overrides, "rot_speed_rpm": R.ravel(), "torque_nm": T.ravel()}
    return predict_settings(model, typ, **cols).reshape(R.shape)


def support_mask(X: pd.DataFrame, rpm=RPM_GRID, torque=TORQUE_GRID,
                 radius: float = 0.1, min_count: int = 3) -> np.ndarray:
    """Grid cells with >= min_count training runs within `radius` (in std units of rpm, torque)."""
    pts = X[["rot_speed_rpm", "torque_nm"]].to_numpy(float)
    sd = pts.std(axis=0)
    R, T = np.meshgrid(rpm, torque)
    nn = NearestNeighbors(radius=radius).fit(pts / sd)
    idx = nn.radius_neighbors(np.c_[R.ravel(), T.ravel()] / sd, return_distance=False)
    return (np.array([len(i) for i in idx]) >= min_count).reshape(R.shape)


def torque_at_power(rpm, power: float) -> np.ndarray:
    """Torque that keeps power constant at a given speed (moves along an iso-power curve)."""
    return power / (np.asarray(rpm) * 2 * np.pi / 60)


def tool_wear_limit(model, typ: str, base: dict, budget: float = RISK_BUDGET, wears=WEAR_GRID):
    """Largest tool wear whose risk at the base setpoint stays <= budget (and the risk curve)."""
    p = predict_settings(model, typ, **{**base, "tool_wear_min": wears})
    over = np.flatnonzero(p > budget)
    limit = wears[over[0] - 1] if over.size and over[0] > 0 else (wears[-1] if not over.size else 0)
    return int(limit), p


def safe_region(model, typ: str, base: dict, support: np.ndarray, wear_max: float,
                budget: float = RISK_BUDGET, n_wear: int = 9):
    """Boolean grid: supported AND P <= budget for every tool wear in [0, wear_max]."""
    safe = support.copy()
    for w in np.linspace(0, wear_max, n_wear):
        safe &= risk_grid(model, typ, base, tool_wear_min=w) <= budget
    return safe


def power_band(safe: np.ndarray, support: np.ndarray, rpm=RPM_GRID, torque=TORQUE_GRID,
               step: float = 250.0, min_frac: float = 0.5):
    """Power interval in which at least `min_frac` of supported cells are safe."""
    R, T = np.meshgrid(rpm, torque)
    pw = power_w(R, T)[support]
    s = safe[support]
    edges = np.arange(0, pw.max() + step, step)
    frac = pd.Series(s).groupby(pd.cut(pw, edges), observed=True).mean()
    ok = [iv for iv, f in frac.items() if f >= min_frac]
    return (ok[0].left, ok[-1].right) if ok else (np.nan, np.nan)


def robust_setpoint(safe: np.ndarray, rpm=RPM_GRID, torque=TORQUE_GRID, scale=(180.0, 10.0)):
    """Safe cell farthest from any unsafe/unsupported cell (distance in std units of rpm, torque)."""
    d = distance_transform_edt(safe, sampling=(np.diff(torque)[0] / scale[1], np.diff(rpm)[0] / scale[0]))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return float(rpm[j]), float(torque[i]), float(d[i, j])


def min_safe_rpm(model, typ: str, base: dict, X: pd.DataFrame, temp_diff: float,
                 budget: float = RISK_BUDGET, rpm=np.arange(1170, 1900, 5)):
    """Lowest speed that is safe at a given temp difference.

    Torque is adjusted to hold power at its median, so that moving along the
    speed axis does not also cross the power-failure band; this isolates the
    heat-dissipation mechanism."""
    p_med = float(np.median(power_w(X.rot_speed_rpm, X.torque_nm)))
    cols = {**base, "process_temp_k": base["air_temp_k"] + temp_diff,
            "rot_speed_rpm": rpm, "torque_nm": torque_at_power(rpm, p_med)}
    p = predict_settings(model, typ, **cols)
    ok = np.flatnonzero(p <= budget)
    return (float(rpm[ok[0]]) if ok.size else np.nan), p


def operating_ranges(model, X: pd.DataFrame, budget: float = RISK_BUDGET, support=None) -> pd.DataFrame:
    """One row per Type: the model-implied safe operating envelope."""
    base = conditions(X)
    support = support_mask(X) if support is None else support
    R, T = np.meshgrid(RPM_GRID, TORQUE_GRID)
    rows = []
    for typ in TYPE_ORDER:
        w_lim, _ = tool_wear_limit(model, typ, base, budget)
        safe = safe_region(model, typ, base, support, w_lim, budget)
        p_lo, p_hi = power_band(safe, support)
        sp_rpm, sp_tq, _ = robust_setpoint(safe)
        rpm_hdf, _ = min_safe_rpm(model, typ, base, X, temp_diff=8.0, budget=budget)
        rows.append({
            "type": typ,
            "tool_wear_max_min": w_lim,
            "power_min_w": p_lo, "power_max_w": p_hi,
            "rpm_min": R[safe].min(), "rpm_max": R[safe].max(),
            "torque_min_nm": T[safe].min(), "torque_max_nm": T[safe].max(),
            "torque_x_wear_max": T[safe].max() * w_lim,
            "min_rpm_if_temp_diff_8K": rpm_hdf,
            "setpoint_rpm": sp_rpm, "setpoint_torque_nm": sp_tq,
            "setpoint_power_w": float(power_w(sp_rpm, sp_tq)),
            "safe_share_of_supported": safe.sum() / support.sum(),
        })
    return pd.DataFrame(rows)
