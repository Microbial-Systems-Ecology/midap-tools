import pandas as pd
from typing import List

def fuse_track_output(dat: List[pd.DataFrame], increasing_columns=["trackID", "lineageID", "trackID_d1", "trackID_d2", "trackID_mother"], increasing_reference="trackID"):
    """
    Concatenates a list of DataFrames while ensuring that specified identifier columns
    are adjusted to maintain unique, increasing values across the combined result.

    Parameters:
    ----------
    dat : List[pd.DataFrame]
        A list of pandas DataFrames to be concatenated.
    increasing_columns : List[str], optional
        A list of column names whose values should be incremented based on the
        maximum of the `increasing_reference` column from the previous DataFrames.
        Default includes common identifier columns: ["trackID", "lineageID", "trackID_d1", "trackID_d2", "trackID_mother"].
    increasing_reference : str, optional
        The column used to determine the offset for incrementing the values in `increasing_columns`.
        This should be a column present in all relevant DataFrames with numeric, increasing values.
        Default is "trackID".

    Returns:
    -------
    pd.DataFrame
        A single concatenated DataFrame with adjusted values in the specified `increasing_columns`
        to ensure unique and continuous identifiers across the inputs.

    Notes:
    -----
    - Assumes all relevant columns are numeric. non-numeric values will result in errors.
    - The offset is computed using the maximum value of the `increasing_reference` column after each DataFrame.
    """
    fused_dat = []
    offset = 0

    for df in dat:
        df_copy = df.copy()
        for col in increasing_columns:
            if col in df_copy.columns:
                df_copy[col] += offset
        fused_dat.append(df_copy)
        if increasing_reference in df_copy.columns:
            offset = df_copy[increasing_reference].max()
        else:
            raise KeyError(f"{increasing_reference} is not a vaild column that can serve as reference")
    result = pd.concat(fused_dat, ignore_index=True)
    return result


