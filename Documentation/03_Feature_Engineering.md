# 03 - Feature Engineering

**Owner:** Muhammad Mubashir Shafique  |  **Files:** `src/features.py`, feature list in `src/config.py`

## What this part does
The raw data has 44 columns. This part decides which ones the models see and creates one new feature. The
same list is used by training, the saved pickle and the Streamlit app, so the app asks the user for exactly the
inputs the model was trained on.

## Why only 13 features
Two rules decided the list:
1. The sign is part of a standard sepsis score (**SIRS**, **qSOFA** or **SOFA**) that doctors already use.
2. The sign was near the top of Random Forest feature importance in the first version.

Fewer inputs also mean the app stays short for a nurse, and the model sees less noise.

## Final features
| Group | Features | Why |
|---|---|---|
| Vitals (6) | HR, Resp, Temp, SBP, MAP, O2Sat | SIRS / qSOFA signs: fast heart, fast breathing, fever, low blood pressure |
| Labs (4) | Lactate, WBC, Creatinine, Platelets | Lactate = poor oxygen delivery, WBC = infection response, Creatinine = kidney, Platelets = clotting (SOFA) |
| Patient (2) | Age, ICULOS | Older patients and longer ICU stay carry more risk |
| New (1) | ShockIndex | HR / SBP, see below |

## The new feature: ShockIndex
```python
df["ShockIndex"] = df["HR"] / df["SBP"]
df = df.replace([np.inf, -np.inf], np.nan)     # divide by zero -> missing -> imputer fills it
```
A normal value is about 0.5-0.7. A higher value means the heart beats fast while the pressure falls, an early
sign of circulation failure. The app computes it automatically (`predict_patient` calls `add_features`), so the
user never types it.

## Not used
| Column | Reason |
|---|---|
| Other labs and blood gases | 85-100% empty |
| `Hour` | almost identical to `ICULOS` |
| `Unit1`, `Unit2`, `Gender` | not part of the SIRS / qSOFA / SOFA signs |
| `Patient_ID` | only used to split patients, never as a feature |
| `SepsisLabel` | the target |

## Importance in the original Random Forest
ICULOS 0.417, Temp 0.103, Lactate 0.099, Resp 0.066, HR 0.053, Creatinine 0.052, WBC 0.048, ShockIndex 0.040,
Age 0.039, Platelets 0.037, SBP 0.021, MAP 0.015, O2Sat 0.010.

## Important note for the viva
`ICULOS` is the strongest feature in this dataset: the longer a patient stays, the higher the sepsis chance. It
is a real pattern in the data but also means part of the score comes from "time in ICU" and not only from
vitals. The Streamlit app is honest about this and is described as a screening aid.
