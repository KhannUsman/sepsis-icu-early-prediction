# 08 - Best Model Settings and Pickle

**Owner:** Muhammad Usman  |  **Files:** `src/select_best_model.py`, `src/final_pipeline.py`, `src/predict.py`

Only **one** model is pickled. All tested models have no pickle.

## Step 1: settings file (`select_best_model.py`)
After comparing all models it saves `models/best_model_config.json`:

| Key | Meaning |
|---|---|
| `model_name` | `hist_gradient_boosting` |
| `tuned_params` | `max_iter 300, max_depth 6, max_leaf_nodes 15, learning_rate 0.02, min_samples_leaf 100, l2_regularization 10, class_weight balanced` |
| `all_model_params` | every parameter of the estimator |
| `pipeline_steps` | imputer, model |
| `features` | 13 feature names in order |
| `tuning_trials` | all settings tried and their validation scores |
| `validation_metrics`, `test_metrics` | scores |
| `models_compared` | selection score of every model |
| `final_threshold`, `final_threshold_high_recall` | the two alert thresholds (about 0.668 and 0.395) |
| `final_test_metrics`, `final_test_metrics_high_recall` | test scores of both modes |
| `sklearn_version` | scikit-learn version used to save the pickle |

## Step 2: final pipeline (`final_pipeline.py`)
`python -m src.final_pipeline`
1. Reads the settings file.
2. Loads the cleaned data and the train / val / test split.
3. Builds the pipeline with `build_any(name, **tuned_params)` (works for old and advanced models) and trains it on train data.
4. Tunes both thresholds on validation, scores the test set.
5. Saves `models/best_model.pkl` and writes the final metrics back into the json.

## Pickle content (a Python dict)
`model_name`, `pipeline` (scikit-learn Pipeline), `threshold`, `threshold_high_recall`, `features`, `metrics`,
`metrics_high_recall`, `sklearn_version`.

## Load and predict
```python
from src.predict import load_bundle, predict_patient
bundle = load_bundle()
result = predict_patient({"HR": 118, "Resp": 24, "Temp": 38.4, "SBP": 96, "MAP": 64,
    "O2Sat": 95, "Lactate": 2.8, "WBC": 14, "Creatinine": 1.6, "Platelets": 140,
    "Age": 70, "ICULOS": 92}, bundle, mode="balanced")   # or mode="high_sensitivity"
print(result)   # probability, threshold, alert, level
```
`predict_patient` adds ShockIndex itself.

## Version note
A pickle only loads safely with the scikit-learn version that saved it (stored in `sklearn_version`). `load_bundle`
warns if the versions differ. If loading fails on another computer, run `python -m src.final_pipeline` once to
recreate the pickle there.
