from collections import deque

import numpy as np
import pandas as pd


def create_new_lineage(df: pd.DataFrame,
                       start_frame: int,
                       output_column: str = "lineageID_postfix",
                       orphans_as_root: bool = True) -> pd.DataFrame:
    """
    Rebuilds cell lineage starting from a given frame.

    Cells present at start_frame become the roots of new lineages. All descendants
    (daughters, granddaughters, ...) inherit the root's trackID as lineageID_postfix.
    Rows at frames before start_frame receive NA.

    TrackIDs that appear at or after start_frame but have no traceable path back to
    a cell at start_frame (e.g. cells entering the field of view, broken tracks) are
    treated according to orphans_as_root:
        - True (default): they become their own lineage root (lineageID_postfix = trackID)
        - False: they receive NA, same as pre-start_frame rows

    Args:
        df (pd.DataFrame): tracking DataFrame with columns frame, trackID, trackID_d1, trackID_d2
        start_frame (int): frame index from which to rebuild lineages (0-based)
        output_column (str): name of the output column. Defaults to "lineageID_postfix"
        orphans_as_root (bool): if True, unresolved trackIDs at frame >= start_frame are
                                self-rooted. Defaults to True.

    Returns:
        pd.DataFrame: copy of df with output_column added
    """
    # Build daughter map: trackID -> {daughter trackIDs}
    # One entry per unique trackID is sufficient since d1/d2 are constant within a track
    track_info = df[["trackID", "trackID_d1", "trackID_d2"]].drop_duplicates(subset="trackID")

    daughter_map: dict[int, set[int]] = {}
    for row in track_info.itertuples(index=False):
        daughters: set[int] = set()
        for d in (row.trackID_d1, row.trackID_d2):
            if pd.notna(d) and d != 0:
                daughters.add(int(d))
        daughter_map[int(row.trackID)] = daughters

    # Seed BFS with all trackIDs present at start_frame
    roots = df.loc[df["frame"] == start_frame, "trackID"].dropna().unique()

    trackid_to_root: dict[int, int] = {}
    queue: deque[int] = deque()
    for r in roots:
        r = int(r)
        trackid_to_root[r] = r
        queue.append(r)

    # BFS: propagate each root trackID to all descendants
    while queue:
        current = queue.popleft()
        root = trackid_to_root[current]
        for daughter in daughter_map.get(current, set()):
            if daughter not in trackid_to_root:
                trackid_to_root[daughter] = root
                queue.append(daughter)

    # Apply: NA before start_frame, mapped root at/after start_frame
    df = df.copy()
    df[output_column] = pd.NA
    mask_post = df["frame"] >= start_frame
    df.loc[mask_post, output_column] = df.loc[mask_post, "trackID"].map(trackid_to_root)

    # Orphans: trackIDs at frame >= start_frame with no resolved root
    if orphans_as_root:
        mask_orphan = mask_post & df[output_column].isna()
        df.loc[mask_orphan, output_column] = df.loc[mask_orphan, "trackID"]

    df[output_column] = df[output_column].astype("Int64")
    return df
