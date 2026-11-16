"""Metrics, threshold tuning and plots shared by all notebooks."""
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from . import config as C


def best_threshold(y_true, y_prob, beta=2.0) -> float:
    """Pick the threshold with the best F-beta score (beta=2 favours recall).

    In an ICU a missed sepsis case is worse than a false alarm.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[np.argmax(f)])


def threshold_for_recall(y_true, y_prob, target=C.HIGH_RECALL_TARGET) -> float:
    """Highest threshold that still catches at least `target` of the sepsis cases.

    Used for the "high sensitivity" alert mode: more cases found, more false alarms.
    """
    _, rec, thr = precision_recall_curve(y_true, y_prob)
    ok = np.where(rec[:-1] >= target)[0]          # recall only falls when threshold rises
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def score_at(y_true, y_prob, thr) -> dict:
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(thr),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_and_save(name, y_val, p_val, y_test, p_test, extra=None,
                      threshold=None) -> dict:
    """Tune threshold on validation, score on validation + test, save files.

    Pass threshold to skip tuning (used for the dummy baseline).
    """
    thr = best_threshold(y_val, p_val) if threshold is None else threshold
    out = {"model": name, "val": score_at(y_val, p_val, thr),
           "test": score_at(y_test, p_test, thr)}
    if extra:
        out["info"] = extra
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key = name.lower().replace(" ", "_")
    with open(C.RESULTS_DIR / f"{key}.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(C.RESULTS_DIR / f"preds_{key}.npz",
                        y_test=np.asarray(y_test), p_test=np.asarray(p_test))
    return out


def add_high_recall_point(name, y_val, p_val, y_test, p_test, target=C.HIGH_RECALL_TARGET):
    """Add the high-sensitivity operating point to results/<name>.json (thr from validation)."""
    key = name.lower().replace(" ", "_")
    path = C.RESULTS_DIR / f"{key}.json"
    out = json.load(open(path))
    thr = threshold_for_recall(y_val, p_val, target)
    out["high_recall"] = {"target": target, "val": score_at(y_val, p_val, thr),
                          "test": score_at(y_test, p_test, thr)}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out["high_recall"]


def plot_curves(y_test, p_test, name, thr):
    """ROC, precision-recall and confusion matrix on the test set."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fpr, tpr, _ = roc_curve(y_test, p_test)
    ax[0].plot(fpr, tpr, color="#c0392b")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title=f"ROC curve - AUC {roc_auc_score(y_test, p_test):.3f}",
              xlabel="False positive rate", ylabel="True positive rate")
    p, r, _ = precision_recall_curve(y_test, p_test)
    ax[1].plot(r, p, color="#2c3e50")
    ax[1].set(title=f"PR curve - AP {average_precision_score(y_test, p_test):.3f}",
              xlabel="Recall", ylabel="Precision")
    cm = confusion_matrix(y_test, (p_test >= thr).astype(int))
    ax[2].imshow(cm, cmap="Reds")
    for (i, j), v in np.ndenumerate(cm):
        ax[2].text(j, i, f"{v:,}", ha="center", va="center")
    ax[2].set(title=f"Confusion matrix (thr {thr:.2f})",
              xticks=[0, 1], yticks=[0, 1],
              xticklabels=["No sepsis", "Sepsis"],
              yticklabels=["No sepsis", "Sepsis"],
              xlabel="Predicted", ylabel="Actual")
    fig.suptitle(name, fontweight="bold")
    fig.tight_layout()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name.lower().replace(' ', '_')}_curves.png",
                dpi=130)
    plt.show()
