# 00 - Overview

**Goal:** warn ICU staff about sepsis up to 6 hours before the clinical diagnosis, from hourly vitals and labs.

## The whole pipeline in one page
1. `src/cleaning.py` cleans the raw CSV and saves `data/processed/clean_data.parquet` (file 01).
2. `src/features.py` adds 1 clinical feature, `ShockIndex` (file 03).
3. Notebooks `02` to `06` train the original models (Dummy, Logistic Regression, Decision Tree, Random Forest, SVM).
   Notebook `08` trains and tunes the advanced models (Histogram Gradient Boosting, tuned Random Forest, Extra Trees,
   and XGBoost / LightGBM when installed). Each notebook saves only `results/<model>.json` and
   `results/preds_<model>.npz`. No model pickles are made for these test models.
4. `src/select_best_model.py` (called from notebook `09`) compares **all** models and saves every setting of the
   winner in `models/best_model_config.json`.
5. `src/final_pipeline.py` reads that file, trains ONLY the winner and saves the **single pickle** `models/best_model.pkl`.
6. `app/streamlit_app.py` loads the pickle through `src/predict.py`. Prediction runs only when the button is pressed.

## Who did what
| Member | Parts |
|---|---|
| Muhammad Mubashir Shafique | Files 01-04: cleaning, EDA, features, Dummy and Logistic Regression |
| Ghulam Mustafa | Files 05, 06, 10: Decision Tree, Random Forest, SVM, advanced models and tuning |
| Muhammad Usman | Files 07-09: evaluation, model selection, pickle, Streamlit app, report |

## Key choices
| Choice | Why |
|---|---|
| Split by patient | A patient has about 39 similar rows. A random row split would leak data. |
| PR-AUC, recall, F2 | Only 2% of rows are sepsis. Accuracy is misleading (97.8% for "never sepsis"). |
| Threshold tuned on validation | The test set stays clean. |
| Same data and 13 features for every model | Scores are directly comparable. |
| Model chosen on validation only | The test set is used only to report the final numbers. |
| One shared `config.py` | Change a setting in one place only. |

## Final model
Tuned **Histogram Gradient Boosting** (selected automatically). It replaced Random Forest as the best model.
Test ROC-AUC 0.785, recall 0.42 in balanced mode and 0.73 in high-sensitivity mode (see files 07 and 10).
