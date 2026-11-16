# 05 - Decision Tree and Random Forest

**Owner:** Ghulam Mustafa  |  **Notebooks:** `04_Decision_Tree.ipynb`, `05_Random_Forest.ipynb`  |  **Settings:** `src/pipeline.py`

These two notebooks are the original tree models and are **not changed**. Stronger and wider-tuned models are in
file 10 (`08_Advanced_Models_Tuning.ipynb`).

## Decision Tree
- A tree asks yes/no questions ("is lactate above 2?") and ends in a prediction. Easy to explain to doctors.
- No scaling needed. `class_weight="balanced"`.
- **Tuning grid:** `max_depth` [4, 6, 8, 10] x `min_samples_leaf` [50, 200] = 8 trials, picked by validation PR-AUC.
- Best: depth 4, leaf 50. Deeper trees got worse on validation (overfitting): PR-AUC 0.083 at depth 4 vs 0.066 at depth 10.
- Test: ROC-AUC 0.735, recall 0.36.

## Random Forest
- Many trees vote together. Each tree sees a random part of the rows and features, so mistakes cancel out.
- `class_weight="balanced_subsample"`, `max_samples=0.4` (each tree uses 40% of rows: faster and less overfit).
- **Tuning:** 2 settings (depth 8 / leaf 50 and depth 12 / leaf 30), 80 trees each. The grid was kept small so it
  runs on a free machine.
- Best: 80 trees, depth 8, leaf 50. Test ROC-AUC **0.783**, recall 0.39.
- Feature importance chart: `results/figures/random_forest_importance.png`.

## Results (test set)
| Model | ROC-AUC | PR-AUC | Recall | Precision | F2 |
|---|---|---|---|---|---|
| Decision Tree | 0.735 | 0.083 | 0.36 | 0.15 | 0.278 |
| Random Forest | 0.783 | 0.102 | 0.39 | 0.14 | 0.287 |

## Challenge
Free-machine run time forced a small tuning grid for the forest. Because of that, file 10 re-tests Random Forest
with a wider search, so the comparison with boosting models is fair.
