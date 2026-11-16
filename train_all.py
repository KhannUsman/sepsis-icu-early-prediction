"""Run the full pipeline.

cleaning -> notebooks 02-06 (one model each) + 08 (advanced models, tuned)
-> select best (config json)
-> final pipeline (trains ONLY the best model, saves best_model.pkl)

Usage:  python train_all.py     (SVM notebook is slow, about 5 minutes; notebook 08 about 25 minutes)
"""
import subprocess
import sys

from src.cleaning import run_cleaning
from src.final_pipeline import train_final
from src.select_best_model import select_best

NOTEBOOKS = ["02_Dummy_Classifier", "03_Logistic_Regression", "04_Decision_Tree",
             "05_Random_Forest", "06_SVM",
             "08_Advanced_Models_Tuning"]

if __name__ == "__main__":
    run_cleaning()
    for nb in NOTEBOOKS:
        print("running", nb)
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", f"notebooks/{nb}.ipynb"], check=True)
    select_best()
    train_final()
