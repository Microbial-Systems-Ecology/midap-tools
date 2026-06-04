from typing import Literal

import numpy as np
import pandas as pd


def compute_life_cycle(
    df: pd.DataFrame,
    id_column: str = "trackID",
    frame_column: str = "frame",
    life_cycle_column: str = "life_cycle",
) -> pd.DataFrame:
    """
    Adds a life-cycle duration column: the total span of each track in frame_column units
    (max − min per id_column group), broadcast to every row of that track.

    Works with integer or float frame columns (e.g. "frame" or "time_hour").

    Args:
        df (pd.DataFrame): input tracking DataFrame
        id_column (str): column identifying individual tracks. Defaults to "trackID"
        frame_column (str): numeric time column (int or float). Defaults to "frame"
        life_cycle_column (str): name of the new duration column. Defaults to "life_cycle"

    Returns:
        pd.DataFrame: DataFrame with life_cycle_column added
    """
    assert pd.api.types.is_numeric_dtype(df[frame_column]), (
        f"frame_column '{frame_column}' must be numeric (int or float)"
    )
    df = df.copy()
    grp = df.groupby(id_column)[frame_column]
    df[life_cycle_column] = grp.transform("max") - grp.transform("min")
    return df


def compute_time_to_split(
    df: pd.DataFrame,
    id_column: str = "trackID",
    frame_column: str = "frame",
    time_to_split_column: str = "time_to_split",
) -> pd.DataFrame:
    """
    Adds a time-to-split column: for each row, the remaining time until the track's last
    observed frame in frame_column units (max − current value), broadcast to every row.

    At any snapshot time t, filtering to rows where frame_column == t gives the remaining
    lifetime for every track alive at t, enabling survival and duration-distribution analyses.

    Works with integer or float frame columns (e.g. "frame" or "time_hour").

    Args:
        df (pd.DataFrame): input tracking DataFrame
        id_column (str): column identifying individual tracks. Defaults to "trackID"
        frame_column (str): numeric time column (int or float). Defaults to "frame"
        time_to_split_column (str): name of the new column. Defaults to "time_to_split"

    Returns:
        pd.DataFrame: DataFrame with time_to_split_column added
    """
    assert pd.api.types.is_numeric_dtype(df[frame_column]), (
        f"frame_column '{frame_column}' must be numeric (int or float)"
    )
    df = df.copy()
    df[time_to_split_column] = (
        df.groupby(id_column)[frame_column].transform("max") - df[frame_column]
    )
    return df


def compute_lag_time(
    df: pd.DataFrame,
    boundary_time: float,
    frame_column: str = "frame",
    id_column: str = "trackID",
    lineage_column: str = "lineageID",
    lag_time_column: str = "lag_time",
    agg: Literal["mean", "median"] = "mean",
    allow_negative: bool = True,
) -> pd.DataFrame:
    """
    Adds a lag-time column: for each track, the difference between its actual cycle
    duration and the reference cycle duration derived from baseline tracks in the same
    lineage (those whose last frame is before boundary_time).

    All tracks receive a lag_time value. Baseline tracks will scatter around 0
    (exactly zero in aggregate when agg="mean"). Tracks in lineages with no baseline
    data receive NaN.

    Args:
        df (pd.DataFrame): input tracking DataFrame
        boundary_time (float): frame value separating baseline from posterior.
            Tracks whose last observed frame is strictly less than boundary_time
            contribute to the per-lineage reference cycle time.
        frame_column (str): numeric time column. Defaults to "frame"
        id_column (str): column identifying individual tracks. Defaults to "trackID"
        lineage_column (str): column identifying lineage groups. Defaults to "lineageID"
        lag_time_column (str): name of the new lag-time column. Defaults to "lag_time"
        agg (str): aggregation used for the per-lineage reference — "mean" or "median".
            Defaults to "mean"
        allow_negative (bool): if False, lag values below 0 are clipped to 0.
            Defaults to True

    Returns:
        pd.DataFrame: DataFrame with lag_time_column added
    """
    assert pd.api.types.is_numeric_dtype(df[frame_column]), (
        f"frame_column '{frame_column}' must be numeric (int or float)"
    )
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")

    df = df.copy()

    # Per-trackID summary: cycle duration, lineage, and whether fully in baseline
    grp = df.groupby(id_column)
    track_stats = pd.DataFrame({
        "_first": grp[frame_column].min(),
        "_last":  grp[frame_column].max(),
        lineage_column: grp[lineage_column].first(),
    })
    track_stats["_cycle"] = track_stats["_last"] - track_stats["_first"]
    track_stats["_is_baseline"] = track_stats["_last"] < boundary_time

    # Per-lineage reference from baseline tracks only
    baseline = track_stats[track_stats["_is_baseline"]]
    if agg == "mean":
        lineage_ref = baseline.groupby(lineage_column)["_cycle"].mean()
    else:
        lineage_ref = baseline.groupby(lineage_column)["_cycle"].median()

    track_stats["_ref_cycle"] = track_stats[lineage_column].map(lineage_ref)
    track_stats[lag_time_column] = track_stats["_cycle"] - track_stats["_ref_cycle"]

    if not allow_negative:
        track_stats[lag_time_column] = track_stats[lag_time_column].clip(lower=0)

    df[lag_time_column] = df[id_column].map(track_stats[lag_time_column])
    return df
