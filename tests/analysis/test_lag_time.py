import pytest
import numpy as np
import pandas as pd
from analysis.life_cycle import compute_lag_time


@pytest.fixture
def sample_df():
    # Lineage 1: two baseline tracks (frames 0-4 and 5-9, cycle=4 each)
    #            one posterior track (frames 12-18, cycle=6)
    # Lineage 2: one baseline track (frames 0-6, cycle=6)
    #            one posterior track (frames 12-14, cycle=2)
    rows = []
    for f in range(5):   # track 1, lineage 1, cycle=4
        rows.append({"trackID": 1, "lineageID": 1, "frame": float(f)})
    for f in range(5, 10):  # track 2, lineage 1, cycle=4
        rows.append({"trackID": 2, "lineageID": 1, "frame": float(f)})
    for f in range(12, 19):  # track 3, lineage 1, cycle=6
        rows.append({"trackID": 3, "lineageID": 1, "frame": float(f)})
    for f in range(7):   # track 4, lineage 2, cycle=6
        rows.append({"trackID": 4, "lineageID": 2, "frame": float(f)})
    for f in range(12, 15):  # track 5, lineage 2, cycle=2
        rows.append({"trackID": 5, "lineageID": 2, "frame": float(f)})
    return pd.DataFrame(rows)


def test_lag_time_posterior_positive(sample_df):
    # boundary=10: baseline mean lineage1=4, posterior track3 cycle=6 → lag=2
    result = compute_lag_time(sample_df, boundary_time=10.0)
    t3 = result[result["trackID"] == 3]["lag_time"].iloc[0]
    assert pytest.approx(t3) == 2.0


def test_lag_time_posterior_negative(sample_df):
    # boundary=10: baseline mean lineage2=6, posterior track5 cycle=2 → lag=-4
    result = compute_lag_time(sample_df, boundary_time=10.0)
    t5 = result[result["trackID"] == 5]["lag_time"].iloc[0]
    assert pytest.approx(t5) == -4.0


def test_lag_time_baseline_mean_zero(sample_df):
    # baseline tracks in lineage 1 (tracks 1 and 2) should have mean lag = 0
    result = compute_lag_time(sample_df, boundary_time=10.0)
    baseline_lags = result[result["trackID"].isin([1, 2])]["lag_time"]
    assert pytest.approx(baseline_lags.mean()) == 0.0


def test_lag_time_allow_negative_false(sample_df):
    result = compute_lag_time(sample_df, boundary_time=10.0, allow_negative=False)
    assert (result["lag_time"].dropna() >= 0).all()


def test_lag_time_allow_negative_true(sample_df):
    result = compute_lag_time(sample_df, boundary_time=10.0, allow_negative=True)
    t5 = result[result["trackID"] == 5]["lag_time"].iloc[0]
    assert t5 < 0


def test_lag_time_median_agg(sample_df):
    # with median: lineage1 baseline tracks both have cycle=4, median=4 → same as mean here
    result = compute_lag_time(sample_df, boundary_time=10.0, agg="median")
    t3 = result[result["trackID"] == 3]["lag_time"].iloc[0]
    assert pytest.approx(t3) == 2.0


def test_lag_time_no_baseline_gives_nan():
    # lineage with no baseline tracks → NaN
    df = pd.DataFrame({
        "trackID":    [1, 1, 2, 2],
        "lineageID":  [1, 1, 1, 1],
        "frame":      [10.0, 12.0, 14.0, 17.0],
    })
    result = compute_lag_time(df, boundary_time=10.0)
    assert result["lag_time"].isna().all()


def test_lag_time_broadcast_per_track(sample_df):
    result = compute_lag_time(sample_df, boundary_time=10.0)
    for tid, grp in result.groupby("trackID"):
        assert grp["lag_time"].nunique(dropna=False) == 1


def test_lag_time_output_length(sample_df):
    result = compute_lag_time(sample_df, boundary_time=10.0)
    assert len(result) == len(sample_df)


def test_lag_time_does_not_mutate(sample_df):
    original_cols = list(sample_df.columns)
    compute_lag_time(sample_df, boundary_time=10.0)
    assert list(sample_df.columns) == original_cols


def test_lag_time_non_numeric_frame_raises(sample_df):
    df = sample_df.copy()
    df["frame"] = df["frame"].astype(str)
    with pytest.raises(AssertionError):
        compute_lag_time(df, boundary_time=10.0)


def test_lag_time_invalid_agg_raises(sample_df):
    with pytest.raises(ValueError, match="agg must be"):
        compute_lag_time(sample_df, boundary_time=10.0, agg="variance")


def test_lag_time_custom_column_name(sample_df):
    result = compute_lag_time(sample_df, boundary_time=10.0, lag_time_column="my_lag")
    assert "my_lag" in result.columns
