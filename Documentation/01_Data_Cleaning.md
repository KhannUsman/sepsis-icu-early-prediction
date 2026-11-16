# 01 - Data Cleaning

**Owner:** Muhammad Mubashir Shafique  |  **File:** `src/cleaning.py`  |  **Run:** `python -m src.cleaning`

## What this part does
The raw hospital data cannot be given to a model directly. It has a broken row, columns that are almost empty,
impossible values (a heart rate of 900) and many gaps. This part turns `data/raw/Dataset.csv` into one clean
file, `data/processed/clean_data.parquet`, that every model uses. All models read the **same** clean file, so
their scores can be compared fairly.

## The raw data in numbers
| Item | Value |
|---|---|
| Rows (one row = one patient, one hour) | 546,123 |
| Columns | 44 (vitals, labs, demographics, ICU stay, `SepsisLabel`) |
| Patients | 14,057 (about 39 hours of data per patient) |
| Rows with sepsis label | 2.17% (very imbalanced) |

## Step by step (function names are in `cleaning.py`)
| # | Step | Function | How it is done | Result |
|---|---|---|---|---|
| 1 | Remove index column | `remove_duplicates` | Drop `Unnamed: 0` (it is a copy of the row number) | 44 -> 43 columns |
| 2 | Remove broken rows | `remove_duplicates` | `dropna(subset=[Patient_ID, SepsisLabel, ICULOS])` | 1 half-empty row at the end of the file removed |
| 3 | Remove duplicates | `remove_duplicates` | `drop_duplicates()` and again on (`Patient_ID`, `ICULOS`) | 0 exact duplicates found. Check is kept for safety |
| 4 | Drop sparse columns | `drop_sparse_columns` | Drop columns with more than 95% missing, except 4 key sepsis labs | 10 columns dropped: EtCO2, SaO2, AST, Alkalinephos, Calcium, Bilirubin_direct, Bilirubin_total, TroponinI, PTT, Fibrinogen |
| 5 | Select columns | `select_columns` | Keep only id, target, ICULOS and the chosen raw features | 11 raw features kept |
| 6 | Fix impossible values | `remove_impossible_values` | Ranges in `config.VALID_RANGES`. Outside range becomes `NaN` | 575 values fixed (Resp 166, MAP 153, WBC 78, O2Sat 59, SBP 45 ...) |
| 7 | Fill missing | `fill_missing` | Forward fill **inside each patient** (last known value) | Vitals now 2-7% empty |
| 8 | Add feature | `features.add_features` | `ShockIndex = HR / SBP` | see file 03 |
| 9 | Split | `split_by_patient` | `GroupShuffleSplit`, 70 / 15 / 15, seed 42 | 9,839 / 2,109 / 2,109 patients |

## Why these choices
- **Keep Lactate, WBC, Creatinine, Platelets even though they are 92-97% empty.** They are core sepsis markers
  (SOFA score). Doctors do not measure labs every hour, so the gaps are normal, not bad data.
- **Forward fill, not a global average.** In an ICU the last measured value is still the best guess for the
  next hour. Filling with an average would hide that a patient is getting worse.
- **Range check instead of deleting rows.** Only the bad value becomes missing, the rest of the row is kept.
- **Split by patient, not by row.** One patient has about 39 rows that look alike. A random row split would put
  the same patient in train and test, and the model would "remember" patients (data leakage). With a patient
  split, the test patients are completely new, like in a real hospital.

## Output
`data/processed/clean_data.parquet`: 546,122 rows, columns `Patient_ID`, `SepsisLabel`, `split` and 13 features.

| Split | Rows | Patients | Sepsis rate |
|---|---|---|---|
| train | 384,548 | 9,839 | 2.20% |
| val | 81,129 | 2,109 | 2.17% |
| test | 80,445 | 2,109 | 2.02% |

## Missing values that remain
After forward fill, vitals are almost complete but labs are still often empty (Lactate 64%, WBC / Creatinine /
Platelets about 18%). The model pipeline therefore starts with `SimpleImputer(strategy="median")`. It learns the
median from the **train** rows only, so no information from validation or test leaks in.

## Challenges
- Very high missing rate in labs: solved with forward fill + train-median imputer, not with deleting rows.
- Hidden leakage risk (many rows per patient): solved with `GroupShuffleSplit`.
- Impossible values (sensor errors): solved with clinical ranges.
