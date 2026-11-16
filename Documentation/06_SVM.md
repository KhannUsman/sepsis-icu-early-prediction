# 06 - Support Vector Machine (RBF)

**Owner:** Ghulam Mustafa  |  **Notebook:** `06_SVM.ipynb`

## How it works
An SVM finds the boundary that separates the two classes with the widest margin. The RBF kernel lets the boundary
bend, so it can learn non-linear patterns.

## Challenge: speed
An RBF SVM needs time that grows with the square of the row count. 384,000 rows would take hours. Solution:

- Train on a **balanced sample**: up to 6,000 sepsis rows + 3x non-sepsis rows (24,000 rows, 25% sepsis).
- Validation and test sets stay **full and untouched**, so the scores are honest.
- The same sampling is reused in `final_pipeline.training_sample` if SVM ever wins.

## Pipeline and tuning
- Median imputer -> `StandardScaler` (SVM needs scaling) -> `SVC(kernel="rbf", probability=True, class_weight="balanced")`.
- `probability=True` gives a risk score, but makes training slower (internal cross-validation).
- **Tuning:** `C` in [0.5, 2.0]. Best `C = 0.5` (validation PR-AUC 0.0823 vs 0.0788). Each trial about 2 minutes.

## Results (test set)
ROC-AUC 0.755, PR-AUC 0.079, recall 0.38, precision 0.09, F2 0.231.

## Conclusion
SVM is competitive on recall but has the lowest precision of the strong models, and it is the slowest to train
and to score. It was not the best model.
