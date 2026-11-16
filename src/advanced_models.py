"""Advanced models (boosting + tuned forests) with random-search tuning.

This file ADDS models. The old models (dummy, logistic regression, decision tree,
random forest, SVM) stay in src/pipeline.py and their notebooks are unchanged.

Every model = median imputer + estimator, same as the old models.
Settings are chosen by random search on the VALIDATION set only.
The test set is never used for tuning or for choosing the model.

Optional models: XGBoost and LightGBM are used only if the library is installed
(pip install xgboost lightgbm). If not installed they are skipped.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from . import config as C
from .pipeline import MODELS as BASE_MODELS
from .pipeline import build_model as build_base_model

RS = C.RANDOM_STATE

# name -> (estimator class, default settings, search space)
ADV_MODELS = {
    # histogram gradient boosting = same algorithm family as LightGBM / XGBoost (hist)
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier,
        dict(random_state=RS),
        dict(learning_rate=[0.02, 0.03, 0.05, 0.08, 0.1],
             max_iter=[100, 200, 300, 400],
             max_depth=[3, 4, 5, 6],
             max_leaf_nodes=[15, 31, 63],
             min_samples_leaf=[50, 100, 200, 400],
             l2_regularization=[0.0, 1.0, 5.0, 10.0, 30.0],
             class_weight=[None, "balanced"]),
    ),
    # Random Forest with a wider search than the old notebook (fair comparison)
    "random_forest_tuned": (
        RandomForestClassifier,
        dict(class_weight="balanced_subsample", n_jobs=-1, random_state=RS),
        dict(n_estimators=[100, 150, 200],
             max_depth=[6, 8, 10, 12, 16],
             min_samples_leaf=[20, 50, 100, 200],
             max_features=["sqrt", 0.5],
             max_samples=[0.3, 0.5]),
    ),
    "extra_trees": (
        ExtraTreesClassifier,
        dict(n_jobs=-1, random_state=RS),
        dict(n_estimators=[100, 200],
             max_depth=[8, 10, 14],
             min_samples_leaf=[20, 50, 100],
             max_features=["sqrt", 0.5],
             class_weight=["balanced_subsample", None]),
    ),
}

# ---- optional libraries (skipped when not installed) -------------------------------
try:
    from xgboost import XGBClassifier
    ADV_MODELS["xgboost"] = (
        XGBClassifier,
        dict(tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=RS),
        dict(n_estimators=[200, 400, 600],
             max_depth=[3, 4, 5, 6],
             learning_rate=[0.02, 0.05, 0.1],
             subsample=[0.7, 0.9, 1.0],
             colsample_bytree=[0.7, 0.9, 1.0],
             min_child_weight=[1, 5, 20],
             reg_lambda=[1.0, 5.0, 10.0],
             scale_pos_weight=[1, 10, 44]),   # 44 = non-sepsis rows / sepsis rows
    )
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    ADV_MODELS["lightgbm"] = (
        LGBMClassifier,
        dict(subsample_freq=1, n_jobs=-1, random_state=RS, verbosity=-1),
        dict(n_estimators=[200, 400, 600],
             num_leaves=[15, 31, 63],
             learning_rate=[0.02, 0.05, 0.1],
             min_child_samples=[20, 50, 100, 200],
             subsample=[0.7, 0.9, 1.0],
             colsample_bytree=[0.7, 0.9, 1.0],
             reg_lambda=[0.0, 5.0, 10.0],
             scale_pos_weight=[1, 10, 44]),
    )
except ImportError:
    pass

OPTIONAL_MISSING = [n for n in ("xgboost", "lightgbm") if n not in ADV_MODELS]


def build_any(name: str, **overrides) -> Pipeline:
    """Build the pipeline of ANY model (old or advanced)."""
    if name in BASE_MODELS:
        return build_base_model(name, **overrides)
    cls, params, _ = ADV_MODELS[name]
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", cls(**{**params, **overrides}))])


def tune_model(name, X_train, y_train, X_val, y_val, n_iter=20, seed=RS, verbose=True):
    """Random search. Score of a trial = mean(val ROC-AUC, val PR-AUC).

    Returns (best fitted pipeline, best params dict, tuning table as DataFrame).
    Only the best pipeline is kept in memory.
    """
    space = ADV_MODELS[name][2]
    n_possible = int(np.prod([len(v) for v in space.values()]))
    trials = list(ParameterSampler(space, n_iter=min(n_iter, n_possible), random_state=seed))
    rows, best_pipe, best_score, best_params = [], None, -1.0, None
    for i, params in enumerate(trials, 1):
        t = time.time()
        pipe = build_any(name, **params)
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_val)[:, 1]
        roc, pr = roc_auc_score(y_val, p), average_precision_score(y_val, p)
        score = (roc + pr) / 2
        rows.append({**params, "val_roc_auc": round(roc, 4), "val_pr_auc": round(pr, 4),
                     "val_score": round(score, 4), "seconds": round(time.time() - t, 1)})
        if score > best_score:
            best_pipe, best_score, best_params = pipe, score, params
        if verbose:
            print(f"[{name}] trial {i}/{len(trials)}  ROC {roc:.3f}  PR {pr:.3f}  "
                  f"score {score:.4f}  best {best_score:.4f}", flush=True)
    table = pd.DataFrame(rows).sort_values("val_score", ascending=False).reset_index(drop=True)
    return best_pipe, best_params, table


def table_records(table: pd.DataFrame) -> list:
    """Tuning table -> list of dicts that is valid JSON (missing values become null)."""
    import json
    return json.loads(table.to_json(orient="records"))
