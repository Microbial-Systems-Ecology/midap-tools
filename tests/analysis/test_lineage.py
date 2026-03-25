import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from analysis.lineage import create_new_lineage


@pytest.fixture
def lineage_df():
    """
    Minimal lineage tree:

        trackID=1  (frames 0-4, root at start_frame=2)
        ├── trackID=2  (frames 3-5, no split)
        └── trackID=3  (frames 3-5, splits into 4 and 5)
                ├── trackID=4  (frames 4-5)
                └── trackID=5  (frames 4-5)

    trackID=10  (frames 4-5, orphan — no path back to start_frame)
    """
    rows = [
        # trackID=1: spans start_frame=2, daughters are 2 and 3
        {"frame": 0, "trackID": 1, "trackID_d1": 2, "trackID_d2": 3},
        {"frame": 1, "trackID": 1, "trackID_d1": 2, "trackID_d2": 3},
        {"frame": 2, "trackID": 1, "trackID_d1": 2, "trackID_d2": 3},
        {"frame": 3, "trackID": 1, "trackID_d1": 2, "trackID_d2": 3},
        {"frame": 4, "trackID": 1, "trackID_d1": 2, "trackID_d2": 3},
        # trackID=2: daughter of 1, no split (NaN = no daughters)
        {"frame": 3, "trackID": 2, "trackID_d1": np.nan, "trackID_d2": np.nan},
        {"frame": 4, "trackID": 2, "trackID_d1": np.nan, "trackID_d2": np.nan},
        {"frame": 5, "trackID": 2, "trackID_d1": np.nan, "trackID_d2": np.nan},
        # trackID=3: daughter of 1, splits into 4 and 5
        {"frame": 3, "trackID": 3, "trackID_d1": 4, "trackID_d2": 5},
        {"frame": 4, "trackID": 3, "trackID_d1": 4, "trackID_d2": 5},
        {"frame": 5, "trackID": 3, "trackID_d1": 4, "trackID_d2": 5},
        # trackID=4: granddaughter via 3, no split
        {"frame": 4, "trackID": 4, "trackID_d1": np.nan, "trackID_d2": np.nan},
        {"frame": 5, "trackID": 4, "trackID_d1": np.nan, "trackID_d2": np.nan},
        # trackID=5: granddaughter via 3, no split
        {"frame": 4, "trackID": 5, "trackID_d1": np.nan, "trackID_d2": np.nan},
        {"frame": 5, "trackID": 5, "trackID_d1": np.nan, "trackID_d2": np.nan},
        # trackID=10: orphan — appears after start_frame with no lineage connection
        {"frame": 4, "trackID": 10, "trackID_d1": np.nan, "trackID_d2": np.nan},
        {"frame": 5, "trackID": 10, "trackID_d1": np.nan, "trackID_d2": np.nan},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

def test_row_count_preserved(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    assert len(result) == len(lineage_df)


def test_output_column_added(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    assert "lineageID_postfix" in result.columns


def test_custom_output_column_name(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2, output_column="my_lineage")
    assert "my_lineage" in result.columns
    assert "lineageID_postfix" not in result.columns


def test_does_not_modify_input(lineage_df):
    original_columns = list(lineage_df.columns)
    create_new_lineage(lineage_df, start_frame=2)
    assert list(lineage_df.columns) == original_columns


# ---------------------------------------------------------------------------
# Pre-start_frame rows → NaN
# ---------------------------------------------------------------------------

def test_pre_start_frame_rows_are_nan(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    pre = result[result["frame"] < 2]
    assert pre["lineageID_postfix"].isna().all()


# ---------------------------------------------------------------------------
# Roots: cells present at start_frame become their own lineage root
# ---------------------------------------------------------------------------

def test_root_at_start_frame(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    at_start = result[(result["frame"] == 2) & (result["trackID"] == 1)]
    assert (at_start["lineageID_postfix"] == 1).all()


# ---------------------------------------------------------------------------
# Daughters and granddaughters inherit the root's trackID
# ---------------------------------------------------------------------------

def test_daughters_inherit_root(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    daughters = result[result["trackID"].isin([2, 3])]
    assert (daughters["lineageID_postfix"] == 1).all()


def test_granddaughters_inherit_root(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    granddaughters = result[result["trackID"].isin([4, 5])]
    assert (granddaughters["lineageID_postfix"] == 1).all()


# ---------------------------------------------------------------------------
# Orphans: trackIDs at frame >= start_frame with no lineage connection
# ---------------------------------------------------------------------------

def test_orphan_self_rooted_by_default(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2)
    orphan_rows = result[result["trackID"] == 10]
    assert (orphan_rows["lineageID_postfix"] == 10).all()


def test_orphan_is_nan_when_disabled(lineage_df):
    result = create_new_lineage(lineage_df, start_frame=2, orphans_as_root=False)
    orphan_rows = result[result["trackID"] == 10]
    assert orphan_rows["lineageID_postfix"].isna().all()


def test_orphan_flag_does_not_affect_known_lineage(lineage_df):
    """Disabling orphans_as_root must not change assignments for cells with known ancestry."""
    result_default = create_new_lineage(lineage_df, start_frame=2, orphans_as_root=True)
    result_no_orphan = create_new_lineage(lineage_df, start_frame=2, orphans_as_root=False)
    known_ids = [1, 2, 3, 4, 5]
    for tid in known_ids:
        vals_default = result_default.loc[result_default["trackID"] == tid, "lineageID_postfix"]
        vals_no_orphan = result_no_orphan.loc[result_no_orphan["trackID"] == tid, "lineageID_postfix"]
        assert_series_equal(vals_default.reset_index(drop=True),
                            vals_no_orphan.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Track spanning start_frame: pre rows NaN, post rows assigned
# ---------------------------------------------------------------------------

def test_spanning_track_split_at_start_frame(lineage_df):
    """trackID=1 spans frames 0-4; only frames >= 2 should be assigned."""
    result = create_new_lineage(lineage_df, start_frame=2)
    track1 = result[result["trackID"] == 1].sort_values("frame")
    assert track1.loc[track1["frame"] < 2, "lineageID_postfix"].isna().all()
    assert (track1.loc[track1["frame"] >= 2, "lineageID_postfix"] == 1).all()


# ---------------------------------------------------------------------------
# NaN daughters (no division): trackID_d1/d2 = NaN treated as no daughter
# ---------------------------------------------------------------------------

def test_nan_daughters_not_followed(lineage_df):
    """Cells with NaN trackID_d1/d2 must not produce extra entries in lineageID_postfix."""
    result = create_new_lineage(lineage_df, start_frame=2)
    # Only trackIDs that actually exist in the data should appear as lineage roots
    valid_ids = set(lineage_df["trackID"].unique())
    assigned = set(result["lineageID_postfix"].dropna().unique())
    assert assigned.issubset(valid_ids)
