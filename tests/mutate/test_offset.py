import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from mutate.offset import apply_frame_offset


@pytest.fixture
def tracking_df():
    """Minimal tracking DataFrame with all frame-index columns present."""
    return pd.DataFrame({
        "frame":       [0, 1, 2, 3, 4],
        "trackID":     [1, 1, 1, 2, 2],
        "first_frame": [0, 0, 0, 3, 3],
        "last_frame":  [2, 2, 2, 4, 4],
        "split":       [0, 0, 1, 0, 0],   # boolean flag, not a frame index
        "area":        [100, 110, 120, 90, 95],
    })


# ---------------------------------------------------------------------------
# Basic shifting behaviour
# ---------------------------------------------------------------------------

def test_frame_column_shifted(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["frame"], tracking_df["frame"] + 3, check_names=False)


def test_first_frame_shifted(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["first_frame"], tracking_df["first_frame"] + 3, check_names=False)


def test_last_frame_shifted(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["last_frame"], tracking_df["last_frame"] + 3, check_names=False)


def test_split_flag_not_modified(tracking_df):
    """split is a boolean flag (0/1), not a frame index — must be left unchanged."""
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["split"], tracking_df["split"], check_names=False)


def test_non_frame_columns_not_modified(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["area"], tracking_df["area"], check_names=False)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_offset_is_noop(tracking_df):
    result = apply_frame_offset(tracking_df, offset=0)
    assert_series_equal(result["frame"], tracking_df["frame"], check_names=False)
    assert_series_equal(result["first_frame"], tracking_df["first_frame"], check_names=False)
    assert_series_equal(result["last_frame"], tracking_df["last_frame"], check_names=False)


def test_negative_offset_raises(tracking_df):
    with pytest.raises(ValueError, match="Offset must be >= 0"):
        apply_frame_offset(tracking_df, offset=-1)


def test_missing_first_last_frame_columns():
    """Should work fine when first_frame / last_frame columns are absent."""
    df = pd.DataFrame({"frame": [0, 1, 2], "trackID": [1, 1, 1]})
    result = apply_frame_offset(df, offset=5)
    assert list(result["frame"]) == [5, 6, 7]
    assert "first_frame" not in result.columns
    assert "last_frame" not in result.columns


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_does_not_modify_input(tracking_df):
    original_frames = tracking_df["frame"].copy()
    apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(tracking_df["frame"], original_frames, check_names=False)


def test_row_count_preserved(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert len(result) == len(tracking_df)
