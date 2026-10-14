import pandas as pd
import pytest

from src.features.engineering import create_features


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Administrative": [2, 0],
            "Administrative_Duration": [60.0, 0.0],
            "Informational": [1, 0],
            "Informational_Duration": [30.0, 0.0],
            "ProductRelated": [7, 0],
            "ProductRelated_Duration": [210.0, 0.0],
        }
    )


def test_adds_expected_columns(sample_df):
    out = create_features(sample_df)
    for col in ["TotalPagesViewed", "TotalDuration", "AvgTimePerPage"]:
        assert col in out.columns


def test_engineered_values_are_correct(sample_df):
    out = create_features(sample_df)
    assert out.loc[0, "TotalPagesViewed"] == 10
    assert out.loc[0, "TotalDuration"] == 300.0
    assert out.loc[0, "AvgTimePerPage"] == 30.0


def test_zero_pages_does_not_divide_by_zero(sample_df):
    out = create_features(sample_df)
    assert out.loc[1, "AvgTimePerPage"] == 0.0


def test_input_dataframe_is_not_modified(sample_df):
    columns_before = list(sample_df.columns)
    create_features(sample_df)
    assert list(sample_df.columns) == columns_before
