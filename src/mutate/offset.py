import pandas as pd


def apply_frame_offset(df: pd.DataFrame,
                       offset: int,
                       frame_column: str = "frame") -> pd.DataFrame:
    """
    Shifts all frame-index columns in a tracking DataFrame forward by offset frames.

    Affected columns: frame, first_frame, last_frame (if present).
    The split column is a boolean flag and is intentionally left unchanged.

    Args:
        df (pd.DataFrame): tracking DataFrame
        offset (int): number of frames to add (must be >= 0)
        frame_column (str): name of the primary frame column. Defaults to "frame"

    Returns:
        pd.DataFrame: copy of df with frame-index columns shifted by offset
    """
    if offset < 0:
        raise ValueError(f"Offset must be >= 0, got {offset}.")

    df = df.copy()
    df[frame_column] = df[frame_column] + offset
    for col in ("first_frame", "last_frame"):
        if col in df.columns:
            df[col] = df[col] + offset
    return df
