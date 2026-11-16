# 10 - Advanced Models and Hyperparameter Tuning

**Owner:** Ghulam Mustafa  |  **Files:** `src/advanced_models.py`, `notebooks/08_Advanced_Models_Tuning.ipynb`

## Why this part exists
The original models reached ROC-AUC 0.72-0.78. The aim here was to try stronger machine-learning models on the **same
data, same 13 features and same patient split**, tune them properly, and replace the Random Forest if one is
clearly better. The old notebooks (02-07) are unchanged.

## Models
| Model | Idea |
|---|---|
| Histogram Gradient Boosting | Trees are built one after another, each one fixing the mistakes of the last. Same family as XGBoost and LightGBM |
| Random Forest (tuned) | The original forest only tried 2 settings. Here it gets the same wide search, so the comparison is fair |
| Extra Trees | A forest with random split points. Less overfit, more random |
| XGBoost, LightGBM | Code is ready. They are used **only if installed** (`pip install xgboost lightgbm`) and are skipped otherwise |

XGBoost and LightGBM could not be run while this project was prepared (no internet to install them), so they have
**no results yet**. After installing them, re-run notebook 08 and then 09: they join the comparison automatically.

## How tuning works (`tune_model`)
1. Random search: settings are drawn at random from a search space (`ADV_MODELS` in `advanced_models.py`).
2. Each trial trains a full pipeline (median imputer + model) on the **train** set.
3. Each trial is scored on the **validation** set: mean of ROC-AUC and PR-AUC.
4. The best trial is kept. The **test** set is never used for tuning.
5. The alert thresholds (balanced and high-sensitivity) are then chosen on validation.

| Model | Trials | Settings searched |
|---|---|---|
| Hist Gradient Boosting | 30 | learning rate, iterations, depth, leaves, min leaf size, L2, class weight |
| Random Forest (tuned) | 10 | trees, depth, min leaf, max features, rows per tree |
| Extra Trees | 10 | trees, depth, min leaf, max features, class weight |

## Winner settings (Histogram Gradient Boosting)
`max_iter 300, learning_rate 0.02, max_depth 6, max_leaf_nodes 15, min_samples_leaf 100, l2_regularization 10, class_weight balanced`.
A small learning rate with strong regularisation and small trees won: with a noisy, rare-class target, simple trees
overfit less.

## Results (test set)
| Model | ROC-AUC | PR-AUC | Recall (balanced) | Precision | Recall (high sens.) | Precision (high sens.) |
|---|---|---|---|---|---|---|
| Hist Gradient Boosting | 0.785 | 0.104 | 0.42 | 0.13 | 0.73 | 0.05 |
| Random Forest (tuned) | 0.780 | 0.103 | 0.42 | 0.12 | 0.72 | 0.05 |
| Extra Trees | 0.781 | 0.104 | 0.35 | 0.16 | 0.72 | 0.05 |
| Random Forest (original) | 0.783 | 0.102 | 0.39 | 0.14 | - | - |

## Honest reading of the results
- Boosting beat the original Random Forest on validation (0.3875 vs 0.3774) and on test recall (0.42 vs 0.39).
  The gain is **small**. All strong models land at ROC-AUC 0.78 because they see only 13 single-time-point features.
- Recall can be raised to about 0.73, but then only 1 alert in 20 is real. That is a property of the data, not a bug.
- The next real improvement is not another algorithm but **history features** (changes over the last hours, rolling
  means, how often labs were measured). The app would then need the patient's recent values as input.

## Feature importance
`results/figures/<model>_importance.png`: permutation importance (shuffle one feature on validation data and see how much
PR-AUC drops). It works for every model type.
