# 07 - Evaluation and Best Model Selection

**Owner:** Muhammad Usman  |  **Files:** `src/evaluation.py`, `src/select_best_model.py`, notebooks `07_Model_Comparison.ipynb` (original, 5 models) and `09_Final_Comparison.ipynb` (all 8 models)

## Metrics (`evaluation.py`)
| Metric | Meaning in this project |
|---|---|
| ROC-AUC | How well the model ranks sepsis rows above normal rows (0.5 = guessing) |
| PR-AUC | Same, but focused on the rare class. A random model scores 0.02 here, so 0.10 is 5x better |
| Recall | Share of real sepsis rows that raise an alert. A missed case is the worst error |
| Precision | Share of alerts that are real. Low precision = false-alarm fatigue |
| Specificity | Share of healthy rows left alone |
| F1 / F2 | Mix of precision and recall. F2 counts recall twice as important |
| Balanced accuracy | Average of recall and specificity |
Plots: ROC, PR curve and confusion matrix (`plot_curves`).

## Threshold
A model gives a score from 0 to 1. The alert threshold decides what counts as sepsis.
- **Balanced mode:** `best_threshold` picks the threshold with the best **F2** on the **validation** set.
- **High-sensitivity mode:** `threshold_for_recall` picks the highest threshold that still keeps recall at 70% or more on validation
  (`HIGH_RECALL_TARGET` in `config.py`).
Both are chosen on validation, never on test.

## Selecting the best model
```
val_score = mean(val_ROC-AUC, val_PR-AUC, val_F2)      # dummy ignored
```
| Model | val_score |
|---|---|
| **Histogram Gradient Boosting** | **0.3875** |
| Random Forest (tuned, notebook 08) | 0.3827 |
| Random Forest (original, notebook 05) | 0.3774 |
| Extra Trees | 0.3751 |
| Decision Tree | 0.3527 |
| SVM | 0.3529 |
| Logistic Regression | 0.3455 |
Dummy 0.1739 (ignored).

The winner changed from Random Forest to Histogram Gradient Boosting. The gap is small (0.3875 vs 0.3774), which is honest:
with single time-point features, the data limits every model.

## Results on the test set (balanced mode)
| Model | ROC-AUC | PR-AUC | Recall | Precision | F2 |
|---|---|---|---|---|---|
| Dummy | 0.500 | 0.020 | 0.00 | 0.00 | 0.000 |
| Logistic Regression | 0.721 | 0.094 | 0.35 | 0.11 | 0.243 |
| Decision Tree | 0.735 | 0.083 | 0.36 | 0.15 | 0.278 |
| Random Forest (original) | 0.783 | 0.102 | 0.39 | 0.14 | 0.287 |
| SVM (RBF) | 0.755 | 0.079 | 0.38 | 0.09 | 0.231 |
| Random Forest (tuned) | 0.780 | 0.103 | 0.42 | 0.12 | 0.279 |
| Extra Trees | 0.781 | 0.104 | 0.35 | 0.16 | 0.283 |
| **Histogram Gradient Boosting** | **0.785** | **0.104** | **0.42** | 0.13 | **0.289** |

## Two alert modes (advanced models, test set)
| Model | Balanced recall | Balanced precision | High-sens. recall | High-sens. precision |
|---|---|---|---|---|
| Histogram Gradient Boosting | 0.42 | 0.13 | 0.73 | 0.05 |
| Random Forest (tuned) | 0.42 | 0.12 | 0.72 | 0.05 |
| Extra Trees | 0.35 | 0.16 | 0.72 | 0.05 |

In high-sensitivity mode the model finds about 7 of 10 sepsis rows, but only about 1 alert in 20 is real
(specificity about 0.70). This is the real trade-off between catching patients and false alarms.
