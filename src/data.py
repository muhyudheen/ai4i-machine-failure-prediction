"""Data loading and the feature/target split.

`load_raw()` returns the full table (incl. IDs and failure-mode flags) for EDA.
`get_X_y()` returns the modelling matrix with leakage and ID columns removed.
"""
import pandas as pd

from src.config import (
    CAT_COLS, DATA_CSV, FAILURE_MODES, ID_COLS, NUM_COLS, RENAME, TARGET, UCI_ID,
)


def _fetch_from_uci() -> pd.DataFrame:
    """Download via ucimlrepo and cache to data/ai4i2020.csv."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_ID)
    df = ds.data.original
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    return df


def load_raw() -> pd.DataFrame:
    """Load the AI4I 2020 table with normalised column names.

    Uses the cached CSV if present, otherwise downloads from UCI. If the
    network is blocked, place the original UCI `ai4i2020.csv` in data/ — the
    rename map accepts both naming conventions.
    """
    df = pd.read_csv(DATA_CSV) if DATA_CSV.exists() else _fetch_from_uci()
    df = df.rename(columns=RENAME)

    expected = set(ID_COLS + CAT_COLS + NUM_COLS + [TARGET] + FAILURE_MODES)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    return df


def get_X_y(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix X and binary target y.

    GRADED DECISION #1 — LEAKAGE:
    The five failure-mode flags (TWF, HDF, PWF, OSF, RNF) are *outcomes*, not
    process inputs: `machine_failure` is (almost exactly) their logical OR, so
    any model given them scores ~100% without learning anything about the
    process. They are never available at prediction time, so they are dropped
    here — the single place X is built — and never re-enter the feature set.

    `uid` / `product_id` are row identifiers (unique per row) with no physical
    meaning; `product_id`'s letter prefix duplicates `type`, which we keep.
    """
    if df is None:
        df = load_raw()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, *FAILURE_MODES, *ID_COLS])
    assert not set(FAILURE_MODES) & set(X.columns), "leakage columns in X"
    return X[CAT_COLS + NUM_COLS], y
