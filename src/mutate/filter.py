import pandas as pd
import numpy as np
from skimage.measure import regionprops
from typing import Union, Tuple
from multiprocessing import Pool, cpu_count

def filter_by_column(
    df: pd.DataFrame,
    column: str,
    min_occurences: int = 0,
    min_value: Union[float, None] = None,
    max_value: Union[float, None] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Filters a track output file (loaded as pd.DataFrame) based on:
    - minimum number of occurrences of values in the specified column
    - minimum and maximum value thresholds

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to filter with. i.e "trackID" for min occurences or "area" for min / max values.
        min_occurences (int, optional): Minimum number of occurrences to retain. Defaults to 0.
        min_value (float, optional): Minimum value threshold. Defaults to None.
        max_value (float, optional): Maximum value threshold. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, dict]: Filtered DataFrame and summary dictionary.
    """
    if df.empty:
        return df, {}
    
    df = df.copy()
    total_rows_before = len(df)

    # Track unique values before filtering (only if min_occurences > 0)
    unique_before = df[column].nunique()

    # Filter by occurrence count
    if min_occurences > 0:
        counts = df[column].value_counts()
        valid_ids = counts[counts >= min_occurences].index
        df = df[df[column].isin(valid_ids)]

    
    # Filter by value thresholds
    if min_value is not None:
        df = df[df[column] >= min_value]
    if max_value is not None:
        df = df[df[column] <= max_value]

    unique_after = df[column].nunique()
    total_rows_after = len(df)

    rate_rows = (total_rows_before - total_rows_after) / total_rows_before * 100
    rate_unique = (unique_before - unique_after) / unique_before * 100
    
    summary = {
        "column_filtered": column,
        "min_occurences": min_occurences,
        "min_value": min_value if min_value is not None else '',
        "max_value": max_value if max_value is not None else '',
        "unique_values_before": unique_before,
        "unique_values_after": unique_after,
        "rows_before": total_rows_before,
        "rows_after": total_rows_after,
        "filter_rate_rows": int(rate_rows),
        "filter_rate_unique_values": int(rate_unique)
    }

    return df.reset_index(drop=True), summary




def filter_tracks_custom(
    df: pd.DataFrame,
    column: str,
    filter_nth_track: int,
    min_occurences: int = 0,
    min_value: Union[float, None] = None,
    max_value: Union[float, None] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Custom function that filters every nth value (i.e frame). example function that shows how a user modified function may look like
    """
    df = df.copy()
    total_rows_before = len(df) 
    df = df[df[column] % filter_nth_track == 0]
    total_rows_after = len(df)
    rate_rows = (total_rows_before - total_rows_after) / total_rows_before * 100

    #our custom summary
    summary = {
        "column_filtered": column,
        "filter_nth_track": filter_nth_track,
        "rows_before": total_rows_before,
        "rows_after": total_rows_after,
        "filter_rate_rows": int(rate_rows),
    }

    return df.reset_index(drop=True), summary


def _check_segment_smoothness(args):
    frame, track_id, mask, smoothness = args
    segment = (mask == track_id)
    
    if not np.any(segment):
        return None  # No segment
    
    region_props = regionprops(segment.astype(np.uint8))
    if not region_props:
        return None

    region = region_props[0]
    if region.area < 5:
        return None

    if region.solidity >= smoothness:
        return (frame, track_id)

    return None


def filter_by_segment_shape_parallel(data: pd.DataFrame,
                            mask: np.ndarray,
                            smoothness: float = 0.8,
                            frame_column: str = "frame",
                            track_id_column: str = "trackID") -> pd.DataFrame:
    """ Filter segments in a tracking dataset based on their shape smoothness.
    This function checks each segment in the provided mask against a smoothness threshold.
    Segments with solidity below the threshold are removed.
    Parameters:
        data (pd.DataFrame): Tracking data with columns for frame and track ID.
        mask (np.ndarray): 3D numpy array representing the segmentation mask.
        smoothness (float): Solidity threshold for segment filtering. Default is 0.8.
        frame_column (str): Column name for frame index in the DataFrame. Default is "frame".
        track_id_column (str): Column name for track ID in the DataFrame. Default is "trackID".
    Returns:
        pd.DataFrame: Filtered DataFrame containing only segments that meet the smoothness criteria.
        dict: Summary statistics including unique values before and after filtering, total rows before and after filtering, and filter rates.
    """
    tasks = []
    grouped = data.groupby(frame_column)

    for frame, frame_data in grouped:
        labels_in_frame = frame_data[track_id_column].unique()
        frame_mask = mask[frame]
        for label_id in labels_in_frame:
            tasks.append((frame, label_id, frame_mask, smoothness))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_check_segment_smoothness, tasks)

    retained_pairs = set(filter(None, results))

    # Efficient filtering using vectorized join logic
    mask_df = pd.DataFrame(list(retained_pairs), columns=[frame_column, track_id_column])
    retained_df = data.merge(mask_df, on=[frame_column, track_id_column], how='inner')

    # Summary statistics
    total_rows_before = len(data)
    unique_before = data[track_id_column].nunique()
    total_rows_after = len(retained_df)
    unique_after = retained_df[track_id_column].nunique()

    summary = {
        "cell_smoothness": smoothness,
        "unique_values_before": unique_before,
        "unique_values_after": unique_after,
        "rows_before": total_rows_before,
        "rows_after": total_rows_after,
        "filter_rate_rows": int((total_rows_before - total_rows_after) / total_rows_before * 100),
        "filter_rate_unique_values": int((unique_before - unique_after) / unique_before * 100)
    }

    return retained_df, summary