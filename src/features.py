"""Feature engineering and feature selection helpers."""
import numpy as np
import pandas as pd

from . import config as C


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ShockIndex (HR / SBP). Works on one row or many rows."""
    df = df.copy()
    df["ShockIndex"] = df["HR"] / df["SBP"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_available_features(df: pd.DataFrame) -> list:
    """Return the final feature list that exists in the dataframe."""
    return [c for c in C.FEATURES if c in df.columns]


def get_xy(df: pd.DataFrame, split: str):
    """Return X, y for 'train', 'val' or 'test'."""
    part = df[df["split"] == split]
    feats = get_available_features(df)
    return part[feats], part[C.TARGET].astype(int), part[C.GROUP]
