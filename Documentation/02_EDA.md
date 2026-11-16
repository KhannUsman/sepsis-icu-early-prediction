# 02 - Exploratory Data Analysis (EDA)

**Owner:** Muhammad Mubashir Shafique  |  **Notebook:** `notebooks/01_Preprocessing_EDA.ipynb`  |  **Charts:** `results/figures/eda_*.png`

## What this part does
Before building models we look at the data to understand it: how much is missing, how rare sepsis is, which
signs differ between sepsis and non-sepsis rows. The findings decided the metrics, the split and the features.

## Charts and what we learned
| Chart (file) | How it is made | What we learned | Decision it caused |
|---|---|---|---|
| Missing values (`eda_missing.png`) | `raw.isna().mean()` as a bar plot with the 95% line | Many labs are 90%+ empty, vitals are 8-15% empty (Temp 66%) | Drop 10 sparse columns, forward fill, median imputer |
| Class balance (`eda_class_balance.png`) | `value_counts()` | Only **2.17%** of rows are sepsis. About 8.8% of patients get sepsis at some point | Do not use accuracy. Use PR-AUC, recall, F2 and class weights |
| Box plots (`eda_boxplots.png`) | `sns.boxplot(x=SepsisLabel)` for 10 signs | Sepsis rows have slightly higher HR, Resp, Temp, ShockIndex. The difference is small and overlaps a lot | One sign alone cannot predict sepsis, models must combine signs |
| Correlation (`eda_correlation.png`) | `df.corr()` heat map | No strong single feature. Highest with the label is `ICULOS` (0.151), then Resp 0.055, HR 0.049, Temp 0.046 | Expect modest scores, use non-linear models |
| Sepsis by ICU hour (`eda_icu_hours.png`) | Group `ICULOS` in 5-hour bins | Sepsis rate changes with time in ICU | Keep `ICULOS` as a feature |

## Why accuracy is a trap here
A model that always answers "no sepsis" is right 97.8% of the time and finds **zero** patients. That is exactly
what the Dummy classifier does (notebook 02). So all notebooks report **recall** (how many sepsis rows are found),
**precision** (how many alarms are real), **PR-AUC** and **F2** (recall counts more than precision).

## Final dataset summary (from the notebook)
| Split | Rows | Patients | Sepsis rate |
|---|---|---|---|
| train | 384,548 | 9,839 | 2.20% |
| val | 81,129 | 2,109 | 2.17% |
| test | 80,445 | 2,109 | 2.02% |

Missing values left in the 13 features after cleaning: 783,315 cells. The model imputer fills them.

## Challenge
The signal is weak and noisy: single vitals barely separate the two classes. This is why the project expects
ROC-AUC around 0.7-0.8 and not 0.95, and why recall is paid for with false alarms.
