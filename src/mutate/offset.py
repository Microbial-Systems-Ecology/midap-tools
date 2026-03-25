import pandas as pd


def apply_frame_offset(df: pd.DataFrame,
                       offset: int,
                       frame_column: str = "frame",
                       out_column: str = "frame_offset") -> pd.DataFrame:
    """
    Shifts frame values forward by offset and writes the result to out_column.

    By default a new column "frame_offset" is added, leaving the original "frame"
    column (used for bitmap/HDF5 indexing) untouched. Set out_column=frame_column
    to overwrite in place; in that case first_frame and last_frame are also shifted.

    Args:
        df (pd.DataFrame): tracking DataFrame
        offset (int): number of frames to add (must be >= 0)
        frame_column (str): column to read frame values from. Defaults to "frame"
        out_column (str): column to write shifted values to. Defaults to "frame_offset"

    Returns:
        pd.DataFrame: copy of df with out_column added or updated
    """
    if offset < 0:
        raise ValueError(f"Offset must be >= 0, got {offset}.")

    df = df.copy()
    df[out_column] = df[frame_column] + offset

    # Only update first_frame / last_frame when overwriting the source column,
    # so bitmap-indexed frame values remain valid in the default new-column mode.
    if out_column == frame_column:
        for col in ("first_frame", "last_frame"):
            if col in df.columns:
                df[col] = df[col] + offset
    return df
