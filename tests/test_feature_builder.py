import pandas as pd

from feature_builder import (
    build_combined_team_table,
    coerce_feature_columns,
    normalize_column,
    normalize_columns,
    standardize_team_labels,
)


def test_normalize_column_strips_and_uppercases():
    assert normalize_column(" power rating ") == "POWER_RATING"


def test_normalize_column_collapses_non_alnum_runs():
    assert normalize_column("Win % (adj)") == "WIN_ADJ"


def test_normalize_columns_renames_all_headers():
    df = pd.DataFrame(columns=["Team Name", "win pct"])
    result = normalize_columns(df)
    assert list(result.columns) == ["TEAM_NAME", "WIN_PCT"]


def test_standardize_team_labels_trims_and_uppercases():
    df = pd.DataFrame({"TEAM": [" Duke ", "unc"]})
    result = standardize_team_labels(df)
    assert result["TEAM"].tolist() == ["DUKE", "UNC"]


def test_standardize_team_labels_noop_without_team_column():
    df = pd.DataFrame({"YEAR": [2024]})
    result = standardize_team_labels(df)
    assert "TEAM" not in result.columns


def test_coerce_feature_columns_converts_numeric_strings():
    df = pd.DataFrame({"YEAR": [2024], "RATING": ["12.5"]})
    result = coerce_feature_columns(df, skip=["YEAR"])
    assert result["RATING"].dtype.kind == "f"


def test_coerce_feature_columns_maps_true_false_to_binary():
    df = pd.DataFrame({"YEAR": [2024, 2025], "FLAG": ["TRUE", "FALSE"]})
    result = coerce_feature_columns(df, skip=["YEAR"])
    assert result["FLAG"].tolist() == [1, 0]


def test_build_combined_team_table_has_one_row_per_team_season():
    combined = build_combined_team_table()
    assert not combined.empty
    assert combined.duplicated(subset=["YEAR", "TEAM"]).sum() == 0
