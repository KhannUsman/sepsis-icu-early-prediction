# 04 - Baseline Models (Dummy and Logistic Regression)

**Owner:** Muhammad Mubashir Shafique  |  **Notebooks:** `02_Dummy_Classifier.ipynb`, `03_Logistic_Regression.ipynb`  |  **Settings:** `src/pipeline.py`

## Why baselines
A baseline tells us how hard the problem is. If a complex model does not beat these two, it is not worth using.
These notebooks are **not changed** by the later improvement work.

## Common pipeline
Every model is a scikit-learn `Pipeline`: `SimpleImputer(median)` -> (`StandardScaler` if needed) -> estimator.
Built in `pipeline.build_model(name, **settings)`.

## Dummy classifier
- `DummyClassifier(strategy="prior")` always predicts the most common class ("no sepsis").
- Result: ROC-AUC 0.500, PR-AUC 0.020, recall 0.00. Accuracy would look like 98% but it finds no patient.
- Meaning: this is the floor. Any real model must beat PR-AUC 0.020 and recall 0.

## Logistic Regression
- Pipeline: median imputer -> `StandardScaler` -> `LogisticRegression(class_weight="balanced", max_iter=500)`.
- **Scaling** is needed because HR (about 80) and Lactate (about 1.5) have very different sizes.
- **`class_weight="balanced"`** makes a missed sepsis row cost about 45x more than a missed healthy row, to
  handle the 2% imbalance.
- **Tuning:** `C` in [0.01, 0.1, 1.0] chosen by validation PR-AUC. All three gave the same PR-AUC (0.0951), so the
  most regularised one (`C = 0.01`) was kept.
- Coefficient chart: `results/figures/logreg_coefficients.png`.

## Results (test set, threshold tuned on validation for F2)
| Model | ROC-AUC | PR-AUC | Recall | Precision | F2 |
|---|---|---|---|---|---|
| Dummy | 0.500 | 0.020 | 0.00 | 0.00 | 0.000 |
| Logistic Regression | 0.721 | 0.094 | 0.35 | 0.11 | 0.243 |

## What we learned
A straight-line model already reaches ROC-AUC 0.72, so there is real signal. But it cannot capture combinations
(for example "high heart rate AND low pressure AND long stay"), which is why tree models do better.
