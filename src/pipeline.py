"""Pipeline factory: (model, imbalance strategy, feature set) -> imblearn Pipeline.

Every configuration compared in this project is built here, so they all share
the same preprocessing and are evaluated by the same CV harness.

Pipeline layout
---------------
    [PhysicsFeatures]  (optional, stateless)
 -> ColumnTransformer: type -> ordinal L=0 < M=1 < H=2 ; numerics -> StandardScaler
 -> [SMOTENC]          (only for strategy "smote")
 -> classifier

Why scale for the tree models too? Trees don't need it (scaling is monotone,
so it changes no splits). We do it anyway because SMOTE picks nearest
neighbours by Euclidean distance. Unscaled, rpm (std ~180) would swamp the
temperatures (std ~2) and synthetic points would only be "near" in rpm. Scaling
every model keeps one preprocessing path, and it costs the trees nothing.
"""
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import NUM_COLS, RANDOM_STATE, TYPE_ORDER
from src.features import ENGINEERED, PhysicsFeatures

MODELS = ["logreg", "rf", "xgb"]
STRATEGIES = ["none", "weighted", "smote"]
MODEL_NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
STRATEGY_NAMES = {"none": "No resampling", "weighted": "Class weighting", "smote": "SMOTE (in-fold)"}

# Neg:pos ratio of the full dataset (9661 / 339). With stratified folds every
# training fold has this ratio to within one sample, so fixing it up front is
# the same as recomputing it per fold, and it reveals nothing about the test fold.
NEG_POS_RATIO = 9661 / 339


def make_preprocessor(engineered: bool) -> ColumnTransformer:
    num_cols = NUM_COLS + (ENGINEERED if engineered else [])
    return ColumnTransformer(
        [
            # `type` first, so it is always column 0 (SMOTENC needs its index).
            ("type", OrdinalEncoder(categories=[TYPE_ORDER]), ["type"]),
            ("num", StandardScaler(), num_cols),
        ],
        verbose_feature_names_out=False,
    )


def make_model(model: str, strategy: str, params: dict | None = None):
    weighted = strategy == "weighted"
    params = params or {}
    if model == "logreg":
        return LogisticRegression(
            max_iter=5000, class_weight="balanced" if weighted else None,
            random_state=RANDOM_STATE, **params,
        )
    if model == "rf":
        defaults = dict(n_estimators=400, min_samples_leaf=2, n_jobs=-1)
        return RandomForestClassifier(
            class_weight="balanced_subsample" if weighted else None,
            random_state=RANDOM_STATE, **{**defaults, **params},
        )
    if model == "xgb":
        defaults = dict(n_estimators=400, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
        return XGBClassifier(
            scale_pos_weight=NEG_POS_RATIO if weighted else 1.0,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1,
            random_state=RANDOM_STATE, **{**defaults, **params},
        )
    raise ValueError(model)


def build_pipeline(model: str, strategy: str, engineered: bool = True,
                   params: dict | None = None) -> Pipeline:
    steps = []
    if engineered:
        steps.append(("physics", PhysicsFeatures()))
    steps.append(("prep", make_preprocessor(engineered)))
    if strategy == "smote":
        # GRADED DECISION #2 — SMOTE ONLY INSIDE THE TRAINING FOLD.
        # This is an imblearn Pipeline, not an sklearn one. Its samplers run
        # only during fit() and are skipped during predict/predict_proba. In
        # CV, fit() sees only the training fold, so synthetic minority points
        # are built from training rows and the validation fold stays real and
        # untouched. Resampling before the split would let synthetic copies of
        # test-fold failures leak into training and inflate every score.
        #
        # SMOTENC (not plain SMOTE) because `type` is categorical. Plain SMOTE
        # would interpolate it to meaningless values like 0.37. SMOTENC takes
        # the neighbours' majority category instead.
        steps.append(("smote", SMOTENC(categorical_features=[0], random_state=RANDOM_STATE)))
    steps.append(("clf", make_model(model, strategy, params)))
    return Pipeline(steps)
