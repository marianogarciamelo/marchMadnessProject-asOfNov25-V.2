"""Feature engineering utilities for building combined team + season datasets. 
   that we can use for the model in marchMadness.py"""

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DATA_DIR = Path(__file__).with_name("march+madness+data")
OUTPUT_PATH = Path(__file__).with_name("combined_features.parquet")

# Core table that provides one row per team + season
BASE_SPEC = {
    "filename": "TeamRankings.csv",
    "keys": ["YEAR", "TEAM"],
    "columns": [
        "TEAM_NO",
        "SEED",
        "ROUND",
        "TR_RANK",
        "TR_RATING",
        "SOS_RANK",
        "SOS_RATING",
        "LUCK_RANK",
        "LUCK_RATING",
    ],
    "name": "TEAMRANKINGS",
    "prefix": "",
}

# Lightweight feature sources we can safely merge at the (YEAR, TEAM) grain.
FEATURE_SPECS = [
    {
        "name": "FIVETHIRTYEIGHT",
        "filename": "538 Ratings.csv",
        "keys": ["YEAR", "TEAM"],
        "columns": ["POWER_RATING", "POWER_RATING_RANK"],
        "prefix": "FTE_",
    },
    {
        "name": "KENPOM_PRE",
        "filename": "KenPom Preseason.csv",
        "keys": ["YEAR", "TEAM"],
        "columns": [
            "PRESEASON_KADJ_EM",
            "PRESEASON_KADJ_O",
            "PRESEASON_KADJ_D",
            "PRESEASON_KADJ_T",
        ],
        "prefix": "KP_PRE_",
    },
    {
        "name": "HEAT_CHECK",
        "filename": "Heat Check Ratings.csv",
        "keys": ["YEAR", "TEAM"],
        "columns": ["EASY_DRAW", "TOUGH_DRAW", "DARK_HORSE", "UPSET_ALERT", "CINDERELLA"],
        "prefix": "HC_",
    },
]

# Helper functions for loading and cleaning feature data
def normalize_column(name: str) -> str:
    """Convert raw CSV headers into snake-case, model-friendly names."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().upper())
    return cleaned.strip("_")

# Normalize all column names in a DataFrame
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column(col) for col in df.columns]
    return df

# Standardize team labels to avoid merge mismatches
def standardize_team_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and uppercase team identifiers to avoid merge mismatches."""
    if "TEAM" not in df.columns:
        return df
    df = df.copy()
    df["TEAM"] = df["TEAM"].astype(str).str.strip().str.upper()
    return df

# Convert string columns to numerics or binary flags where appropriate
def coerce_feature_columns(df: pd.DataFrame, skip: Iterable[str]) -> pd.DataFrame:
    """Attempt to convert string columns into numerics or binary flags."""
    df = df.copy()
    for col in df.columns:
        if col in skip:
            continue
        try:
            numeric_col = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            numeric_col = df[col]
        else:
            df[col] = numeric_col
        if df[col].dtype == object:
            upper_vals = (
                df[col].dropna().astype(str).str.upper().unique().tolist()
            )
            if upper_vals and set(upper_vals).issubset({"TRUE", "FALSE"}):
                df[col] = (
                    df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
                )
    return df

# Ensure the data directory exists
def validate_data_dir() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Expected data directory at {DATA_DIR}, but it does not exist."
        )

# Load, clean, and subset one CSV described by `spec`
def read_spec_table(spec: dict) -> pd.DataFrame:
    """Load, clean, and subset one CSV described by `spec`."""
    path = DATA_DIR / spec["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = standardize_team_labels(df)

    required_cols = list(dict.fromkeys(spec["keys"] + spec.get("columns", [])))
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"{spec['name']} missing expected columns: {', '.join(missing)}"
        )

    df = df[required_cols]
    df = coerce_feature_columns(df, spec["keys"])
    df = df.drop_duplicates(subset=spec["keys"])

    prefix = spec.get("prefix", "")
    if prefix:
        rename_map = {
            col: f"{prefix}{col}" for col in df.columns if col not in spec["keys"]
        }
        df = df.rename(columns=rename_map)
    return df

# Build the combined feature table by merging all specified sources
def build_combined_team_table(
    feature_specs: List[dict] = FEATURE_SPECS,
) -> pd.DataFrame:
    """Merge base TeamRankings data with a handful of feature sources."""
    validate_data_dir()
    base_df = read_spec_table(BASE_SPEC)

    combined = base_df.copy()
    for spec in feature_specs:
        feature_df = read_spec_table(spec)
        combined = combined.merge(
            feature_df,
            on=spec["keys"],
            how="left",
            validate="m:1",
        )
    combined = combined.dropna(axis=1, how="all")
    combined = combined.sort_values(["YEAR", "TEAM"]).reset_index(drop=True)
    return combined

# Materialize the combined feature table to disk
def materialize_dataset(output_path: Path = OUTPUT_PATH) -> Path:
    """Build the combined feature table and persist it for downstream modeling."""
    combined = build_combined_team_table()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    combined.to_csv(csv_path, index=False)
    print(
        f"Wrote {len(combined):,} rows with {combined.shape[1]} columns "
        f"to {output_path.name} and {csv_path.name}"
    )
    return output_path


if __name__ == "__main__":
    materialize_dataset()
