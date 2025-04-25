import pytest
import pandas as pd
from mutate.fuse import fuse_track_output


@pytest.fixture
def sample_dataframes():
    df1 = pd.DataFrame({
        "trackID": [1, 2, 3],
        "lineageID": [1, 1, 1],
        "something_else": [1, 1, 1],
        "other": ["a", "b", "a"]
    })
    df2 = pd.DataFrame({
        "trackID": [1, 2],
        "lineageID": [1, 1],
        "something_else": [1, 1],
        "other": ["c", "d"]
    })
    df3 = pd.DataFrame({
        "trackID": [1, 2],
        "lineageID": [1, 1],
        "something_else": [1, 1],
        "other": ["e", "f"]
    })
    return [df1, df2, df3]


def test_fuse_preserves_row_order(sample_dataframes):
    result = fuse_track_output(sample_dataframes, increasing_columns=["trackID", "lineageID"], increasing_reference="trackID")
    assert len(result) == 7
    assert list(result["other"]) == ["a", "b", "a", "c", "d", "e", "f"]


def test_track_id_increments_correctly(sample_dataframes):
    result = fuse_track_output(sample_dataframes, increasing_columns=["trackID", "lineageID"], increasing_reference="trackID")
    assert list(result["trackID"]) == [1, 2, 3, 4, 5, 6, 7]


def test_other_column_is_untouched(sample_dataframes):
    result = fuse_track_output(sample_dataframes, increasing_columns=["trackID", "lineageID"], increasing_reference="trackID")
    assert list(result["something_else"]) == [1, 1, 1, 1, 1, 1, 1]


def test_lineage_id_increments_if_included(sample_dataframes):
    result = fuse_track_output(sample_dataframes, increasing_columns=["trackID", "lineageID"], increasing_reference="trackID")
    assert list(result["lineageID"]) == [1, 1, 1, 4, 4, 6, 6]


def test_missing_reference_column_raises_error(sample_dataframes):
    with pytest.raises(KeyError, match="is not a vaild column"):
        _ = fuse_track_output(sample_dataframes, increasing_reference="nonexistent_column")
