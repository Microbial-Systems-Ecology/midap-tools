import pandas as pd
from mutate.fuse import fuse_track_output

def test_fuse_track_output_basic():
    df1 = pd.DataFrame({
        "trackID": [1, 2, 3],
        "lineageID": [1, 1, 1],
        "other": ["a", "b", "a"]
    })
    df2 = pd.DataFrame({
        "trackID": [1, 2],
        "lineageID": [1, 1],
        "other": ["c", "d"]
    })
    df3 = pd.DataFrame({
        "trackID": [1, 2],
        "lineageID": [1, 1],
        "other": ["e", "f"]
    })

    result = fuse_track_output([df1, df2, df3], increasing_columns=["trackID", "lineageID"], increasing_reference="trackID")
    # Check length and ordering
    assert len(result) == 7
    assert list(result["other"]) == ["a", "b", "a", "c", "d", "e", "f"]
    # Check that trackID is offset correctly
    assert list(result["trackID"]) == [1, 2, 3, 4, 5, 6, 7]
    # Check that lineageID is offset similarly
    assert list(result["lineageID"]) == [1, 1, 1, 4, 4, 6 ,6]

