"""Data cleaning: duplicates, impossible values, missing values, split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C
from .features import add_features


def load_raw(path=C.RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the index column, exact duplicate rows and repeated patient hours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    # rows without patient id or label are broken (the file ends with one)
    df = df.dropna(subset=[C.GROUP, C.TARGET, "ICULOS"])
    df[C.GROUP] = df[C.GROUP].astype(int)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[C.GROUP, "ICULOS"], keep="first")
    return df.reset_index(drop=True)


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are almost always empty (keep key sepsis markers)."""
    miss = df.isna().mean()
    drop = [c for c in miss[miss > C.MISSING_DROP_THRESHOLD].index
            if c not in C.ALWAYS_KEEP]
    return df.drop(columns=drop)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only id, target, ICULOS and the chosen raw features."""
    raw_feats = [c for c in C.FEATURES if c not in C.ENGINEERED]
    keep = [C.GROUP, C.TARGET, "ICULOS"] + [c for c in raw_feats if c != "ICULOS"]
    return df[keep]


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Set values outside the possible clinical range to NaN."""
    df = df.copy()
    for col, (lo, hi) in C.VALID_RANGES.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill inside each patient (last known value), then use median.

    Median is computed on the full data only for the cleaned file. The model
    pipeline has its own imputer that learns the median from train data only.
    """
    df = df.sort_values([C.GROUP, "ICULOS"]).copy()
    measured = [c for c in C.VITALS + C.LABS if c in df.columns]
    df[measured] = df.groupby(C.GROUP)[measured].ffill()
    return df.reset_index(drop=True)


def split_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'split' column: train / val / test (grouped by patient)."""
    ids = df[C.GROUP].unique()
    frame = pd.DataFrame({C.GROUP: ids})
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                            random_state=C.RANDOM_STATE)
    rest_idx, test_idx = next(gss.split(frame, groups=frame[C.GROUP]))
    rest = frame.iloc[rest_idx]
    val_frac = C.VAL_SIZE / (1 - C.TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac,
                             random_state=C.RANDOM_STATE)
    tr_idx, va_idx = next(gss2.split(rest, groups=rest[C.GROUP]))
    mapping = {}
    mapping.update({p: "train" for p in rest.iloc[tr_idx][C.GROUP]})
    mapping.update({p: "val" for p in rest.iloc[va_idx][C.GROUP]})
    mapping.update({p: "test" for p in frame.iloc[test_idx][C.GROUP]})
    df = df.copy()
    df["split"] = df[C.GROUP].map(mapping)
    return df


def run_cleaning(verbose=True) -> pd.DataFrame:
    """Full cleaning pipeline. Saves data/processed/clean_data.parquet."""
    df = load_raw()
    log = {"raw_rows": len(df)}
    df = remove_duplicates(df)
    log["after_duplicates"] = len(df)
    df = drop_sparse_columns(df)
    df = select_columns(df)
    log["columns_kept"] = df.shape[1]
    df = remove_impossible_values(df)
    df = fill_missing(df)
    df = add_features(df)
    df = split_by_patient(df)
    keep = [C.GROUP, C.TARGET, "split"] + [c for c in C.FEATURES if c in df.columns]
    df = df[keep]
    C.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.CLEAN_PATH, index=False)
    if verbose:
        print(log)
        print(df["split"].value_counts())
    return df


if __name__ == "__main__":
    run_cleaning()
