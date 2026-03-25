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
        "split":       [0, 0, 1, 0, 0],
        "area":        [100, 110, 120, 90, 95],
    })


# ---------------------------------------------------------------------------
# Default new-column mode (out_column="frame_offset")
# ---------------------------------------------------------------------------

def test_new_column_created(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert "frame_offset" in result.columns


def test_new_column_values_shifted(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["frame_offset"], tracking_df["frame"] + 3, check_names=False)


def test_original_frame_column_unchanged(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["frame"], tracking_df["frame"], check_names=False)


def test_first_last_frame_unchanged_in_new_column_mode(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(result["first_frame"], tracking_df["first_frame"], check_names=False)
    assert_series_equal(result["last_frame"], tracking_df["last_frame"], check_names=False)


def test_custom_out_column_name(tracking_df):
    result = apply_frame_offset(tracking_df, offset=2, out_column="my_frames")
    assert "my_frames" in result.columns
    assert "frame_offset" not in result.columns


# ---------------------------------------------------------------------------
# Overwrite mode (out_column == frame_column)
# ---------------------------------------------------------------------------

def test_overwrite_mode_shifts_frame(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3, out_column="frame")
    assert_series_equal(result["frame"], tracking_df["frame"] + 3, check_names=False)


def test_overwrite_mode_shifts_first_last_frame(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3, out_column="frame")
    assert_series_equal(result["first_frame"], tracking_df["first_frame"] + 3, check_names=False)
    assert_series_equal(result["last_frame"], tracking_df["last_frame"] + 3, check_names=False)


def test_overwrite_mode_split_unchanged(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3, out_column="frame")
    assert_series_equal(result["split"], tracking_df["split"], check_names=False)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_offset_new_column_equals_frame(tracking_df):
    result = apply_frame_offset(tracking_df, offset=0)
    assert_series_equal(result["frame_offset"], tracking_df["frame"], check_names=False)


def test_negative_offset_raises(tracking_df):
    with pytest.raises(ValueError, match="Offset must be >= 0"):
        apply_frame_offset(tracking_df, offset=-1)


def test_does_not_modify_input(tracking_df):
    original_frames = tracking_df["frame"].copy()
    apply_frame_offset(tracking_df, offset=3)
    assert_series_equal(tracking_df["frame"], original_frames, check_names=False)


def test_row_count_preserved(tracking_df):
    result = apply_frame_offset(tracking_df, offset=3)
    assert len(result) == len(tracking_df)
