import numpy as np
import pandas as pd
from typing import Dict, Tuple
from mutate.fuse import fuse_track_output

VALID_SET_OPERATIONS = ["difference", "union", "intersect"]

def multichannel_set_operations(data: Dict[str, pd.DataFrame],
                                masks: Dict[str, np.ndarray],
                                type: str = "difference") -> Tuple[pd.DataFrame, np.ndarray]:
    """    Perform set operations across multiple channels of tracking data and masks.
    This function applies a specified set operation (difference, union, or intersect)
    across all provided channels, merging the results into a single DataFrame and mask.
    Parameters:
    ----------
    data : Dict[str, pd.DataFrame]
        A dictionary where keys are channel identifiers and values are pandas DataFrames
        containing tracking data with columns 'frame', 'x', 'y', and 'trackID'.
    masks : Dict[str, np.ndarray]
        A dictionary where keys are channel identifiers and values are 3D numpy arrays
        representing binary masks with shape (frames, height, width).
    type : str, optional
        The type of set operation to perform. Supported values are:
        - "difference": Keeps elements in the first channel that are not in subsequent channels.
        - "union": Combines elements from all channels.
        - "intersect": Keeps only elements that are present in all channels.
        Default is "difference".
    Returns:
    -------
    Tuple[pd.DataFrame, np.ndarray]
        A tuple containing:
        - A pandas DataFrame with the merged tracking data.
        - A 3D numpy array representing the resulting mask after the set operation.
    Raises:
    ------
    KeyError: If the specified `type` is not a valid set operation.
    Notes:
    - The function assumes that all DataFrames have the same structure and that masks are compatible in terms of dimensions.
    - The resulting DataFrame will contain unique trackIDs across all channels, adjusted to avoid collisions.
    """
    if type not in VALID_SET_OPERATIONS:
        raise KeyError(f"{type} is not a valid set operation. Supported operations are {VALID_SET_OPERATIONS}")
    
    keys = list(data.keys())
    if not keys:
        return pd.DataFrame(columns=["frame", "x", "y", "trackID"]), np.array([])

    data_out = data[keys[0]]
    mask_out = masks[keys[0]]
    
    for key in keys[1:]:
        data_out, mask_out = set_operation(
            data_out, data[key],
            mask_out, masks[key],
            type=type
        )

    return data_out, mask_out


def set_operation(data_ref: pd.DataFrame,
                  data_diff: pd.DataFrame,
                  masks_ref: np.ndarray,
                  masks_diff: np.ndarray,
                  type: str = "difference") -> Tuple[pd.DataFrame, np.ndarray]:
    """    Perform a set operation (difference, union, or intersect) on two sets of tracking data and masks.
    This function applies the specified set operation on the provided tracking data and masks,
    merging the results into a single DataFrame and mask.
    Parameters:
    ----------
    data_ref : pd.DataFrame
        The reference tracking data DataFrame with columns 'frame', 'x', 'y', and 'trackID'.
    data_diff : pd.DataFrame
        The tracking data DataFrame to compare against the reference.
    masks_ref : np.ndarray
        The reference binary mask with shape (frames, height, width).
    masks_diff : np.ndarray
        The binary mask to compare against the reference, with shape (frames, height, width).
    type : str, optional
        The type of set operation to perform. Supported values are:
        - "difference": Keeps elements in the reference that are not in the diff.
        - "union": Combines elements from both reference and diff.
        - "intersect": Keeps only elements that are present in both reference and diff.
        Default is "difference".
    Returns:
    -------
    Tuple[pd.DataFrame, np.ndarray]
        A tuple containing:
        - A pandas DataFrame with the merged tracking data.
        - A 3D numpy array representing the resulting mask after the set operation.
    Raises:
    ------
    KeyError: If the specified `type` is not a valid set operation.
    Notes:
    - The function assumes that both `data_ref` and `data_diff` have the same structure.
    - The resulting DataFrame will contain unique trackIDs adjusted to avoid collisions.
    """
    if type not in VALID_SET_OPERATIONS:
        raise KeyError(f"{type} is not a valid set operation. Supported operations are {VALID_SET_OPERATIONS}")

    # Remap trackIDs in data_diff to avoid collisions
    masks_diff = masks_diff.copy()
    max_ref_id = data_ref["trackID"].max()
    
        # Merge dataframes (needed to reassemble later if union)
    data_merged = fuse_track_output([data_ref, data_diff])
    
    data_diff = data_diff.copy()
    data_diff["trackID"] += max_ref_id
    masks_diff[masks_diff > 0] += max_ref_id


    frames = max(masks_ref.shape[0], masks_diff.shape[0])
    height, width = masks_ref.shape[1:]

    mask_out = np.zeros((frames, height, width), dtype=masks_ref.dtype)
    kept_ids = set()
    kept_rows = []

    for frame in range(frames):
        ref_mask = masks_ref[frame] if frame < masks_ref.shape[0] else np.zeros((height, width), dtype=masks_ref.dtype)
        diff_mask = masks_diff[frame] if frame < masks_diff.shape[0] else np.zeros((height, width), dtype=masks_diff.dtype)

        df_ref_f = data_ref[data_ref["frame"] == frame]
        df_diff_f = data_diff[data_diff["frame"] == frame]

        ids_to_keep = _frame_set_operation(df_ref_f, df_diff_f, ref_mask, diff_mask, type)

        # Filter merged data only for this frame and keep only valid trackIDs
        merged_f = data_merged[(data_merged["frame"] == frame) & (data_merged["trackID"].isin(ids_to_keep))]
        kept_rows.append(merged_f)

        for y in range(height):
            for x in range(width):
                val_r = ref_mask[y, x]
                val_d = diff_mask[y, x]

                if type == "difference" and val_r in ids_to_keep:
                    mask_out[frame, y, x] = val_r
                elif type == "intersect" and val_r in ids_to_keep:
                    mask_out[frame, y, x] = val_r
                elif type == "union":
                    if val_r in ids_to_keep:
                        mask_out[frame, y, x] = val_r
                    elif val_d in ids_to_keep:
                        mask_out[frame, y, x] = val_d

    # Filter final dataframe
    final_data = pd.concat(kept_rows, ignore_index=True)
    return final_data, mask_out


def _frame_set_operation(df_ref_f: pd.DataFrame,
                         df_diff_f: pd.DataFrame,
                         ref_mask: np.ndarray,
                         diff_mask: np.ndarray,
                         op_type: str) -> set:
    """    Perform a set operation on the tracking data for a single frame.
    This function applies the specified set operation (difference, union, or intersect)
    on the tracking data for a single frame, determining which trackIDs to keep based on
    the provided masks.
    Parameters:
    ----------
    df_ref_f : pd.DataFrame
        The reference tracking data DataFrame for the current frame.
    df_diff_f : pd.DataFrame
        The tracking data DataFrame to compare against the reference for the current frame.
    ref_mask : np.ndarray
        The reference binary mask for the current frame with shape (height, width).
    diff_mask : np.ndarray
        The binary mask to compare against the reference for the current frame with shape (height, width).
    op_type : str
        The type of set operation to perform. Supported values are:
        - "difference": Keeps elements in the reference that are not in the diff.
        - "union": Combines elements from both reference and diff.
        - "intersect": Keeps only elements that are present in both reference and diff.
    Returns:
    -------
    set
        A set of trackIDs that should be kept after applying the specified set operation.
    Raises:
    ------
    KeyError: If the specified `op_type` is not a valid set operation.
    Notes:
    - The function assumes that both `df_ref_f` and `df_diff_f` have the same structure.
    - The resulting set will contain unique trackIDs adjusted to avoid collisions.
    """
    keep_ids = set()

    if op_type == "difference":
        for _, row in df_ref_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < ref_mask.shape[0] and 0 <= y < ref_mask.shape[1]:
                if diff_mask[x, y] == 0:
                    keep_ids.add(row["trackID"])

    elif op_type == "intersect":
        for _, row in df_ref_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < diff_mask.shape[0] and 0 <= y < diff_mask.shape[1]:
                if diff_mask[x, y] != 0:
                    keep_ids.add(row["trackID"])

    elif op_type == "union":
        # Include all trackIDs from ref where the point exists in *either* mask
        for _, row in df_ref_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < ref_mask.shape[0] and 0 <= y < ref_mask.shape[1]:
                if ref_mask[x, y] != 0 or diff_mask[x, y] != 0:
                    keep_ids.add(row["trackID"])

        # Include all trackIDs from diff where point exists *only* in diff
        for _, row in df_diff_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < diff_mask.shape[0] and 0 <= y < diff_mask.shape[1]:
                if ref_mask[x, y] == 0 and diff_mask[x, y] != 0:
                    keep_ids.add(row["trackID"])

    return keep_ids
