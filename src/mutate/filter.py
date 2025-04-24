import pandas as pd
from typing import Union, Tuple

def filter_tracks(
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
        column (str, optional): Column to filter. Defaults to "trackID".
        min_occurences (int, optional): Minimum number of occurrences to retain. Defaults to 0.
        min_value (float, optional): Minimum value threshold. Defaults to None.
        max_value (float, optional): Maximum value threshold. Defaults to None.
        silent (bool, optional): If False, prints summary statistics. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, dict]: Filtered DataFrame and summary dictionary.
    """
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
        "min_value": min_value,
        "max_value": max_value,
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
    Custom function that filters every nth value (i.e frame)
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