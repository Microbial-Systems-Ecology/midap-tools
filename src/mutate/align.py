import numpy as np
import pandas as pd


def cut_to_frame_range(df: pd.DataFrame,
                       start: int,
                       end: int,
                       frame_column: str = "frame",
                       out_column: str = "frame_aligned") -> pd.DataFrame:
    """
    Reindexes frames relative to start and writes the result to out_column.

    By default a new column "frame_aligned" is added, leaving the original "frame"
    column (used for bitmap/HDF5 indexing) untouched. Rows outside [start, end]
    receive NaN in the new column and are kept in the DataFrame.

    Set out_column=frame_column to overwrite in place; in that mode rows outside
    [start, end] are dropped and first_frame / last_frame are shifted and clamped
    to [0, end - start].

    Args:
        df (pd.DataFrame): tracking DataFrame
        start (int): first frame of the shared range (inclusive)
        end (int): last frame of the shared range (inclusive)
        frame_column (str): column to read frame values from. Defaults to "frame"
        out_column (str): column to write aligned values to. Defaults to "frame_aligned"

    Returns:
        pd.DataFrame: copy of df with out_column added or updated
    """
    df = df.copy()
    mask = (df[frame_column] >= start) & (df[frame_column] <= end)

    if out_column == frame_column:
        # Overwrite mode: drop rows outside range and update first_frame / last_frame
        df = df[mask].copy()
        df[frame_column] = df[frame_column] - start
        new_end = end - start
        for col in ("first_frame", "last_frame"):
            if col in df.columns:
                df[col] = (df[col] - start).clip(lower=0, upper=new_end)
    else:
        # New-column mode: keep all rows, NaN outside range
        df[out_column] = np.nan
        df.loc[mask, out_column] = df.loc[mask, frame_column] - start

    return df
