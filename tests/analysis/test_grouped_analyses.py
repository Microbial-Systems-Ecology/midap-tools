import pytest
import pandas as pd
import numpy as np
from analysis.grouped_analyses import grouped_rank, grouped_metric


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "frame":   [1, 1, 1, 2, 2, 2],
        "chamber": ["A", "A", "A", "A", "A", "A"],
        "trackID": [1, 2, 3, 1, 2, 3],
        "x":       [10.0, 30.0, 20.0, 15.0, 5.0, 25.0],
    })


# ----- grouped_rank -----

def test_grouped_rank_descending(sample_df):
    result = grouped_rank(sample_df, group_columns=["frame", "chamber"], value_column="x",
                          rank_column="rank", ascending=False)
    frame1 = result[result["frame"] == 1].sort_values("x")
    # x=[10,20,30] descending → ranks [3, 2, 1]
    assert list(frame1["rank"]) == [3, 2, 1]


def test_grouped_rank_ascending(sample_df):
    result = grouped_rank(sample_df, group_columns=["frame", "chamber"], value_column="x",
                          rank_column="rank", ascending=True)
    frame1 = result[result["frame"] == 1].sort_values("x")
    # x=[10,20,30] ascending → ranks [1, 2, 3]
    assert list(frame1["rank"]) == [1, 2, 3]


def test_grouped_rank_output_length(sample_df):
    result = grouped_rank(sample_df, group_columns=["frame", "chamber"], value_column="x")
    assert len(result) == len(sample_df)


def test_grouped_rank_does_not_mutate(sample_df):
    original_cols = list(sample_df.columns)
    grouped_rank(sample_df, group_columns=["frame", "chamber"], value_column="x")
    assert list(sample_df.columns) == original_cols


def test_grouped_rank_single_group_column(sample_df):
    result = grouped_rank(sample_df, group_columns="frame", value_column="x", ascending=True)
    assert "rank" in result.columns


# ----- grouped_metric -----

def test_grouped_metric_mean(sample_df):
    result = grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x", mode="mean")
    # frame=1: mean([10,30,20]) = 20.0
    assert (result[result["frame"] == 1]["x_mean"] == 20.0).all()


def test_grouped_metric_min(sample_df):
    result = grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x",
                            mode="min", metric_column="x_min_val")
    assert (result[result["frame"] == 1]["x_min_val"] == 10.0).all()


def test_grouped_metric_max(sample_df):
    result = grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x", mode="max")
    assert (result[result["frame"] == 1]["x_max"] == 30.0).all()


def test_grouped_metric_sd(sample_df):
    result = grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x", mode="sd")
    expected_sd = np.std([10.0, 30.0, 20.0], ddof=1)
    assert np.allclose(result[result["frame"] == 1]["x_sd"].values, expected_sd)


def test_grouped_metric_default_column_name(sample_df):
    result = grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x", mode="mean")
    assert "x_mean" in result.columns


def test_grouped_metric_invalid_mode(sample_df):
    with pytest.raises(ValueError, match="mode must be one of"):
        grouped_metric(sample_df, group_columns=["frame"], value_column="x", mode="variance")


def test_grouped_metric_does_not_mutate(sample_df):
    original_cols = list(sample_df.columns)
    grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x")
    assert list(sample_df.columns) == original_cols


def test_grouped_metric_output_length(sample_df):
    result = grouped_metric(sample_df, group_columns=["frame", "chamber"], value_column="x")
    assert len(result) == len(sample_df)
