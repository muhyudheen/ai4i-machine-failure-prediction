"""Cross-validation harness shared by every model / strategy / feature set.

One explicit loop instead of `cross_validate`, so that a single pass gives us
both the per-fold metrics *and* the out-of-fold (OOF) probabilities needed for
PR/ROC curves, confusion matrices and threshold analysis.

All configurations use the same `CV` splitter (fixed seed), so they are
scored on identical folds and per-fold differences between configurations are
paired comparisons.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import RANDOM_STATE

N_SPLITS = 5
CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# GRADED DECISION #3 — PRIMARY METRIC = PR-AUC (average precision).
# With 3.4% positives, ROC-AUC looks flattering: its x-axis (false-positive
# rate) is divided by ~9,700 negatives, so hundreds of false alarms barely move
# it. PR-AUC is built only from precision and recall on the failure class, and
# its chance level is the 3.4% prevalence, not 0.5. It is also threshold-free.
# That matters here because class weighting and SMOTE both shift the predicted
# probabilities, so F1/precision/recall at a fixed 0.5 cut-off do not compare
# fairly across strategies. Accuracy is reported only to show how misleading
# it is: predicting "no failure" for everything already scores 96.6%.
PRIMARY = "pr_auc"
METRICS = ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]
METRIC_NAMES = {
    "pr_auc": "PR-AUC (primary)", "roc_auc": "ROC-AUC", "f1": "F1 (failure)",
    "recall": "Recall", "precision": "Precision", "accuracy": "Accuracy (misleading)",
}


def fold_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


@dataclass
class CVResult:
    folds: pd.DataFrame   # one row per fold, metrics at threshold 0.5
    oof: np.ndarray       # out-of-fold P(failure) for every row

    def summary(self) -> pd.Series:
        return self.folds[METRICS].agg(["mean", "std"]).T.stack()


def run_cv(pipeline, X: pd.DataFrame, y: pd.Series, cv=CV) -> CVResult:
    """Fit a fresh clone of `pipeline` on each training fold, score the held-out fold."""
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        est = clone(pipeline)
        est.fit(X.iloc[tr], y.iloc[tr])          # any resampling happens here, on tr only
        p = est.predict_proba(X.iloc[te])[:, 1]  # te is never resampled
        oof[te] = p
        rows.append({"fold": k, **fold_metrics(y.iloc[te], p)})
    return CVResult(pd.DataFrame(rows), oof)


def mean_std(df: pd.DataFrame, metrics=METRICS, digits: int = 3) -> pd.Series:
    return pd.Series({m: f"{df[m].mean():.{digits}f} ± {df[m].std():.{digits}f}" for m in metrics})


# ---------------------------------------------------------------- thresholds
def threshold_for_recall(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target (= best precision at that recall)."""
    _, recall, thr = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= target_recall          # recall[i] belongs to thr[i]
    return float(thr[ok].max()) if ok.any() else float(thr.min())


def threshold_for_f1(y_true, proba) -> float:
    precision, recall, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def nested_threshold_cv(pipeline, X, y, target_recall: float, outer=CV,
                        inner_splits: int = N_SPLITS) -> pd.DataFrame:
    """Honest estimate of performance at a *tuned* threshold.

    If you pick the threshold on the same OOF predictions you then score, the
    result is optimistic. Instead, for every outer fold:
      1. run an inner stratified CV on the outer-training part only, giving
         inner OOF probabilities;
      2. choose the threshold that reaches `target_recall` on those;
      3. refit on the whole outer-training part and score the untouched outer
         test fold at that threshold.
    """
    rows = []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE + 1 + k)
        inner_oof = run_cv(pipeline, X_tr, y_tr, cv=inner).oof
        t = threshold_for_recall(y_tr, inner_oof, target_recall)

        est = clone(pipeline).fit(X_tr, y_tr)
        p = est.predict_proba(X.iloc[te])[:, 1]
        rows.append({"fold": k, "threshold": t, **fold_metrics(y.iloc[te], p, threshold=t)})
    return pd.DataFrame(rows)


# --------------------------------------------------------- hyper-parameters
def nested_tuning_cv(pipeline, param_distributions: dict, X, y, n_iter: int = 25,
                     outer=CV, inner_splits: int = 3):
    """Nested CV for hyper-parameter search: the inner search picks params on the
    outer-training part only, and the outer fold scores the result. Returns the
    per-fold outer metrics and the params each outer fold chose."""
    rows, chosen = [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        search = RandomizedSearchCV(
            pipeline, param_distributions, n_iter=n_iter, scoring="average_precision",
            cv=StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE + 1 + k),
            random_state=RANDOM_STATE, n_jobs=1, refit=True,
        )
        search.fit(X.iloc[tr], y.iloc[tr])
        p = search.predict_proba(X.iloc[te])[:, 1]
        rows.append({"fold": k, "inner_best_pr_auc": search.best_score_, **fold_metrics(y.iloc[te], p)})
        chosen.append(search.best_params_)
    return pd.DataFrame(rows), chosen
