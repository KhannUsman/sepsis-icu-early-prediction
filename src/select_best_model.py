"""Compare all models and save the winner's settings in ONE json file.

Rule: highest average of validation ROC-AUC, PR-AUC and F2 (dummy ignored).
All models (old + advanced) take part. Old models keep their saved results.
Sepsis is rare (about 2% of rows), so PR-AUC and F2 are used with ROC-AUC.
Only validation data is used, the test set is never used to choose.

Output: models/best_model_config.json  (no model is pickled here)
Run:    python -m src.select_best_model
"""
import json

import pandas as pd

from . import config as C
from .advanced_models import ADV_MODELS, build_any

BASE_NAMES = ["dummy", "logistic_regression", "decision_tree", "random_forest", "svm"]
ADV_NAMES = ["hist_gradient_boosting", "random_forest_tuned", "extra_trees",
             "xgboost", "lightgbm"]          # xgboost / lightgbm only if installed and run
MODEL_NAMES = BASE_NAMES + ADV_NAMES


def load_results() -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        path = C.RESULTS_DIR / f"{name}.json"
        if not path.exists():
            if name in ADV_NAMES and name not in ADV_MODELS:
                print(f"skipped: {name} (library not installed)")
            else:
                print(f"missing result: {name} (run its notebook first)")
            continue
        r = json.load(open(path))
        row = {"model": name}
        row.update({f"val_{k}": v for k, v in r["val"].items()
                    if k in ("roc_auc", "pr_auc", "f2")})
        row.update({f"test_{k}": v for k, v in r["test"].items()
                    if k in ("roc_auc", "pr_auc", "precision", "recall",
                             "specificity", "f1", "f2", "balanced_accuracy")})
        row["threshold"] = r["test"]["threshold"]
        hr = r.get("high_recall")
        if hr:
            row.update({"hr_recall": hr["test"]["recall"], "hr_precision": hr["test"]["precision"],
                        "hr_specificity": hr["test"]["specificity"]})
        row["params"] = json.dumps(r.get("info", {}).get("params", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def select_best(verbose=True) -> dict:
    table = load_results()
    table["val_score"] = table[["val_roc_auc", "val_pr_auc", "val_f2"]].mean(axis=1)
    table.to_csv(C.RESULTS_DIR / "comparison.csv", index=False)
    best = table[table["model"] != "dummy"].sort_values(
        "val_score", ascending=False).iloc[0]
    name = best["model"]
    res = json.load(open(C.RESULTS_DIR / f"{name}.json"))
    tuned = res["info"]["params"]
    full = build_any(name, **tuned).named_steps["model"].get_params()
    config = {
        "model_name": name,
        "selection_rule": "highest mean of validation ROC-AUC, PR-AUC and F2",
        "tuned_params": tuned,
        "all_model_params": full,
        "pipeline_steps": [n for n, _ in build_any(name, **tuned).steps],
        "features": C.FEATURES,
        "threshold": res["test"]["threshold"],
        "random_state": C.RANDOM_STATE,
        "tuning_trials": res["info"].get("tuning", []),
        "validation_metrics": res["val"],
        "test_metrics": res["test"],
        "models_compared": {r["model"]: round(float(r["val_score"]), 4)
                            for _, r in table.iterrows()},
    }
    C.MODEL_DIR.mkdir(exist_ok=True)
    with open(C.BEST_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)
    if verbose:
        print(table.drop(columns="params").round(3).to_string(index=False))
        print(f"\nBEST MODEL: {name} -> settings saved in {C.BEST_CONFIG_PATH}")
    return config


if __name__ == "__main__":
    select_best()
