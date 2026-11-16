"""Final pipeline: train ONLY the selected model and save its pickle.

Reads models/best_model_config.json (made by select_best_model.py),
cleans data (if needed), trains the chosen model with the saved settings,
tunes the alert threshold on validation, scores the test set and saves
models/best_model.pkl. This is the only model pickle in the project.

Run:  python -m src.final_pipeline
"""
import json
import pickle

import pandas as pd
import sklearn

from . import config as C
from .cleaning import run_cleaning
from .evaluation import best_threshold, score_at, threshold_for_recall
from .features import get_xy
from .advanced_models import build_any
from .pipeline import predict_proba


def training_sample(name, X, y):
    """SVM is slow, so it uses the same balanced sample as its notebook."""
    if name != "svm":
        return X, y
    pos = y[y == 1].sample(min(6000, int(y.sum())), random_state=C.RANDOM_STATE).index
    neg = y[y == 0].sample(len(pos) * 3, random_state=C.RANDOM_STATE).index
    idx = pos.union(neg)
    return X.loc[idx], y.loc[idx]


def train_final(verbose=True) -> dict:
    config = json.load(open(C.BEST_CONFIG_PATH))
    name = config["model_name"]
    if not C.CLEAN_PATH.exists():
        run_cleaning(verbose=False)
    df = pd.read_parquet(C.CLEAN_PATH)
    X_train, y_train, _ = get_xy(df, "train")
    X_val, y_val, _ = get_xy(df, "val")
    X_test, y_test, _ = get_xy(df, "test")

    pipe = build_any(name, **config["tuned_params"])
    Xs, ys = training_sample(name, X_train, y_train)
    pipe.fit(Xs, ys)

    p_val, p_test = predict_proba(pipe, X_val), predict_proba(pipe, X_test)
    thr = best_threshold(y_val, p_val)                     # balanced mode (F2)
    thr_hr = threshold_for_recall(y_val, p_val)            # high-sensitivity mode
    test = score_at(y_test, p_test, thr)
    test_hr = score_at(y_test, p_test, thr_hr)
    bundle = {
        "model_name": name,
        "pipeline": pipe,
        "threshold": thr,
        "threshold_high_recall": thr_hr,
        "features": list(X_train.columns),
        "metrics": {f"test_{k}": v for k, v in test.items()},
        "metrics_high_recall": {f"test_{k}": v for k, v in test_hr.items()},
        "sklearn_version": sklearn.__version__,
    }
    with open(C.BEST_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    config["final_threshold"] = thr
    config["final_test_metrics"] = test
    config["final_threshold_high_recall"] = thr_hr
    config["final_test_metrics_high_recall"] = test_hr
    config["sklearn_version"] = sklearn.__version__
    with open(C.BEST_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)
    if verbose:
        print(f"Saved {C.BEST_MODEL_PATH.name}: {name}, threshold {thr:.3f} "
              f"(high-sensitivity threshold {thr_hr:.3f})")
        print("balanced mode   :", {k: round(v, 3) for k, v in test.items() if isinstance(v, float)})
        print("high sensitivity:", {k: round(v, 3) for k, v in test_hr.items() if isinstance(v, float)})
    return bundle


if __name__ == "__main__":
    train_final()
