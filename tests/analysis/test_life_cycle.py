import pytest
import pandas as pd
import numpy as np
from analysis.life_cycle import compute_life_cycle, compute_time_to_split


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "trackID": [1, 1, 1, 2, 2],
        "frame":   [0, 1, 2, 1, 3],
    })


@pytest.fixture
def float_frame_df():
    return pd.DataFrame({
        "trackID": [1, 1, 1, 2, 2],
        "frame":   [0.0, 0.5, 1.0, 0.5, 1.5],
    })


# ----- compute_life_cycle -----

def test_life_cycle_int_frames(sample_df):
    result = compute_life_cycle(sample_df, life_cycle_column="lc")
    # track 1: max=2 − min=0 = 2; track 2: max=3 − min=1 = 2
    assert all(result[result["trackID"] == 1]["lc"] == 2)
    assert all(result[result["trackID"] == 2]["lc"] == 2)


def test_life_cycle_float_frames(float_frame_df):
    result = compute_life_cycle(float_frame_df, life_cycle_column="lc")
    # track 1: 1.0 − 0.0 = 1.0; track 2: 1.5 − 0.5 = 1.0
    assert np.allclose(result[result["trackID"] == 1]["lc"].values, 1.0)
    assert np.allclose(result[result["trackID"] == 2]["lc"].values, 1.0)


def test_life_cycle_output_length(sample_df):
    result = compute_life_cycle(sample_df)
    assert len(result) == len(sample_df)


def test_life_cycle_does_not_mutate(sample_df):
    original_cols = list(sample_df.columns)
    compute_life_cycle(sample_df)
    assert list(sample_df.columns) == original_cols


def test_life_cycle_non_numeric_raises(sample_df):
    df = sample_df.copy()
    df["frame"] = df["frame"].astype(str)
    with pytest.raises(AssertionError):
        compute_life_cycle(df)


def test_life_cycle_custom_column_name(sample_df):
    result = compute_life_cycle(sample_df, life_cycle_column="duration")
    assert "duration" in result.columns


# ----- compute_time_to_split -----

def test_time_to_split_int_frames(sample_df):
    result = compute_time_to_split(sample_df, time_to_split_column="tts")
    # track 1 last=2: frame=0 → 2, frame=1 → 1, frame=2 → 0
    t1 = result[result["trackID"] == 1].sort_values("frame")
    assert list(t1["tts"]) == [2, 1, 0]


def test_time_to_split_float_frames(float_frame_df):
    result = compute_time_to_split(float_frame_df, time_to_split_column="tts")
    # track 1 last=1.0: frame=0.0 → 1.0, frame=0.5 → 0.5, frame=1.0 → 0.0
    t1 = result[result["trackID"] == 1].sort_values("frame")
    assert list(t1["tts"]) == pytest.approx([1.0, 0.5, 0.0])


def test_time_to_split_last_row_is_zero(sample_df):
    result = compute_time_to_split(sample_df)
    for _, group in result.groupby("trackID"):
        last_row = group.sort_values("frame").iloc[-1]
        assert last_row["time_to_split"] == pytest.approx(0.0)


def test_time_to_split_output_length(sample_df):
    result = compute_time_to_split(sample_df)
    assert len(result) == len(sample_df)


def test_time_to_split_does_not_mutate(sample_df):
    original_cols = list(sample_df.columns)
    compute_time_to_split(sample_df)
    assert list(sample_df.columns) == original_cols


def test_time_to_split_non_numeric_raises(sample_df):
    df = sample_df.copy()
    df["frame"] = df["frame"].astype(str)
    with pytest.raises(AssertionError):
        compute_time_to_split(df)


def test_time_to_split_nonnegative(sample_df):
    result = compute_time_to_split(sample_df)
    assert (result["time_to_split"] >= 0).all()
