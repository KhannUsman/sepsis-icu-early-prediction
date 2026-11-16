"""Load the saved best model and score one patient."""
import pickle
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .features import add_features


def load_bundle(path=C.BEST_MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    import sklearn
    saved = bundle.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(f"Model was saved with scikit-learn {saved}, you have {sklearn.__version__}. "
                      "If loading or predicting fails, run: python -m src.final_pipeline")
    return bundle


def risk_level(prob: float, threshold: float) -> str:
    """Low / Elevated / High, relative to the model alert threshold."""
    if prob >= threshold:
        return "High"
    if prob >= 0.8 * threshold:
        return "Elevated"
    return "Low"


def predict_patient(values: dict, bundle: dict, mode: str = "balanced") -> dict:
    """values: raw inputs (vitals, labs, patient). Missing labs can be None.

    mode: "balanced" (F2 threshold) or "high_sensitivity" (catches more sepsis cases,
    gives more false alarms).
    """
    row = pd.DataFrame([values]).astype(float)   # None -> NaN
    row = add_features(row)
    X = row.reindex(columns=bundle["features"])
    prob = float(bundle["pipeline"].predict_proba(X)[0, 1])
    thr = bundle["threshold"]
    if mode == "high_sensitivity":
        thr = bundle.get("threshold_high_recall", thr)
    return {"probability": prob, "threshold": thr,
            "alert": prob >= thr, "level": risk_level(prob, thr)}


if __name__ == "__main__":
    b = load_bundle()
    print(b["model_name"], round(b["threshold"], 3))
