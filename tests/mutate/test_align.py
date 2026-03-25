import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from mutate.align import cut_to_frame_range


@pytest.fixture
def tracking_df():
    """DataFrame spanning frames 0–5."""
    return pd.DataFrame({
        "frame":       [0, 1, 2, 3, 4, 5],
        "trackID":     [1, 1, 1, 1, 1, 1],
        "first_frame": [0, 0, 0, 0, 0, 0],
        "last_frame":  [5, 5, 5, 5, 5, 5],
        "split":       [0, 0, 0, 0, 0, 1],
        "area":        [10, 11, 12, 13, 14, 15],
    })


# ---------------------------------------------------------------------------
# Default new-column mode (out_column="frame_aligned")
# ---------------------------------------------------------------------------

def test_new_column_created(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4)
    assert "frame_aligned" in result.columns


def test_original_frame_column_unchanged(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4)
    assert_series_equal(result["frame"], tracking_df["frame"], check_names=False)


def test_rows_inside_range_get_reindexed_values(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4)
    inside = result[result["frame"].isin([2, 3, 4])]
    assert list(inside["frame_aligned"]) == [0.0, 1.0, 2.0]


def test_rows_outside_range_get_nan(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4)
    outside = result[~result["frame"].isin([2, 3, 4])]
    assert outside["frame_aligned"].isna().all()


def test_all_rows_kept_in_new_column_mode(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4)
    assert len(result) == len(tracking_df)


def test_first_last_frame_unchanged_in_new_column_mode(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4)
    assert_series_equal(result["first_frame"], tracking_df["first_frame"], check_names=False)
    assert_series_equal(result["last_frame"], tracking_df["last_frame"], check_names=False)


def test_custom_out_column_name(tracking_df):
    result = cut_to_frame_range(tracking_df, start=1, end=4, out_column="my_aligned")
    assert "my_aligned" in result.columns
    assert "frame_aligned" not in result.columns


# ---------------------------------------------------------------------------
# Overwrite mode (out_column == frame_column)
# ---------------------------------------------------------------------------

def test_overwrite_mode_drops_rows_outside_range(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4, out_column="frame")
    assert set(result["frame"]) == {0, 1, 2}


def test_overwrite_mode_reindexes_from_zero(tracking_df):
    result = cut_to_frame_range(tracking_df, start=3, end=5, out_column="frame")
    assert list(result["frame"]) == [0, 1, 2]


def test_overwrite_mode_row_count_matches_range(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4, out_column="frame")
    assert len(result) == 3


def test_overwrite_mode_first_frame_clamped():
    """Cell started before cut start: first_frame should clamp to 0."""
    df = pd.DataFrame({
        "frame":       [3, 4, 5],
        "trackID":     [1, 1, 1],
        "first_frame": [1, 1, 1],
        "last_frame":  [5, 5, 5],
    })
    result = cut_to_frame_range(df, start=3, end=5, out_column="frame")
    assert (result["first_frame"] == 0).all()


def test_overwrite_mode_last_frame_clamped():
    """Cell ends after cut end: last_frame should clamp to new_end."""
    df = pd.DataFrame({
        "frame":       [0, 1, 2],
        "trackID":     [1, 1, 1],
        "first_frame": [0, 0, 0],
        "last_frame":  [8, 8, 8],
    })
    result = cut_to_frame_range(df, start=0, end=2, out_column="frame")
    assert (result["last_frame"] == 2).all()


def test_overwrite_mode_split_unchanged(tracking_df):
    result = cut_to_frame_range(tracking_df, start=3, end=5, out_column="frame")
    assert list(result["split"]) == [0, 0, 1]


# ---------------------------------------------------------------------------
# Shared: non-frame columns and immutability
# ---------------------------------------------------------------------------

def test_non_frame_columns_preserved(tracking_df):
    result = cut_to_frame_range(tracking_df, start=2, end=4, out_column="frame")
    assert list(result["area"]) == [12, 13, 14]


def test_missing_first_last_frame_columns_new_mode():
    df = pd.DataFrame({"frame": [0, 1, 2, 3, 4], "trackID": [1] * 5})
    result = cut_to_frame_range(df, start=2, end=4)
    assert "first_frame" not in result.columns
    assert result["frame_aligned"].notna().sum() == 3


def test_missing_first_last_frame_columns_overwrite_mode():
    df = pd.DataFrame({"frame": [0, 1, 2, 3, 4], "trackID": [1] * 5})
    result = cut_to_frame_range(df, start=2, end=4, out_column="frame")
    assert list(result["frame"]) == [0, 1, 2]


def test_does_not_modify_input(tracking_df):
    original_frames = list(tracking_df["frame"])
    cut_to_frame_range(tracking_df, start=2, end=4)
    assert list(tracking_df["frame"]) == original_frames
