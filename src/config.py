"""Central settings for the whole project. Change values here only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Dataset.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean_data.parquet"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
BEST_CONFIG_PATH = MODEL_DIR / "best_model_config.json"

# High-sensitivity alert mode: threshold that keeps at least this recall on validation
HIGH_RECALL_TARGET = 0.70

TARGET = "SepsisLabel"
GROUP = "Patient_ID"
RANDOM_STATE = 42

# Columns with more than this share of missing values are dropped
MISSING_DROP_THRESHOLD = 0.95
# Important sepsis markers that we keep even if they are mostly missing
ALWAYS_KEEP = ["Lactate", "WBC", "Creatinine", "Platelets"]

# Only the most important sepsis signs (SIRS / qSOFA / SOFA + model importance)
VITALS = ["HR", "Resp", "Temp", "SBP", "MAP", "O2Sat"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets"]
PATIENT = ["Age", "ICULOS"]
ENGINEERED = ["ShockIndex"]
FEATURES = VITALS + LABS + PATIENT + ENGINEERED

# Physiologically possible ranges. Values outside become missing (NaN).
VALID_RANGES = {
    "HR": (20, 250), "O2Sat": (40, 100), "Temp": (30, 43),
    "SBP": (40, 300), "MAP": (20, 200), "DBP": (20, 200), "Resp": (4, 60),
    "BaseExcess": (-30, 30), "HCO3": (5, 50), "FiO2": (0.21, 1.0),
    "pH": (6.8, 7.8), "PaCO2": (15, 100), "Glucose": (20, 800),
    "Lactate": (0.3, 25), "Magnesium": (0.5, 6), "Phosphate": (0.5, 15),
    "Potassium": (1.5, 9), "Hct": (10, 65), "Hgb": (3, 20), "BUN": (2, 200),
    "Chloride": (70, 130), "Creatinine": (0.2, 20), "WBC": (0.5, 100),
    "Platelets": (5, 1000), "Age": (18, 90), "HospAdmTime": (-1000, 24),
}

# Patient level split (no patient is in two sets)
TEST_SIZE = 0.15
VAL_SIZE = 0.15

FIG_DIR.mkdir(parents=True, exist_ok=True)
