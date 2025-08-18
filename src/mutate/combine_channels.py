import numpy as np
import pandas as pd
from typing import Dict, Tuple
from itertools import repeat
from mutate.fuse import fuse_track_output
from skimage.morphology import disk
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

VALID_SET_OPERATIONS = ["difference", "union", "intersect"]

def multichannel_set_operations(data: Dict[str, pd.DataFrame],
                                masks: Dict[str, np.ndarray],
                                type: str = "difference",
                                radius: int = 0) -> Tuple[pd.DataFrame, np.ndarray]:
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
    radius : int
        The radius (px) within which will be checked if there is an overlapp. defaults to 0 = within center pixel. larger number will increase accuracy but punish performance
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
        data_out, mask_out = set_operation_parallelized(
            data_out, data[key],
            mask_out, masks[key],
            type=type,
            radius = radius
        )

    return data_out, mask_out


def set_operation_parallelized(data_ref: pd.DataFrame,
                  data_diff: pd.DataFrame,
                  masks_ref: np.ndarray,
                  masks_diff: np.ndarray,
                  type: str = "difference",
                  radius: int = 0) -> Tuple[pd.DataFrame, np.ndarray]:
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
    radius : int
        The radius (px) within which will be checked if there is an overlapp. defaults to 0 = within center pixel. larger number will increase accuracy but punish performance
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

    # Zero out mask values not present in the DataFrame (ref and diff)
    valid_ref_ids = set(data_ref["trackID"].unique())
    valid_diff_ids = set(data_diff["trackID"].unique())

    # Vectorized cleanup of masks
    masks_ref[~np.isin(masks_ref, list(valid_ref_ids))] = 0
    masks_diff[~np.isin(masks_diff, list(valid_diff_ids))] = 0

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

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(
            _process_frame,
            [data_ref[data_ref["frame"] == f] for f in range(frames)],
            [data_diff[data_diff["frame"] == f] for f in range(frames)],
            [masks_ref[f] if f < masks_ref.shape[0] else np.zeros((height, width), dtype=masks_ref.dtype) for f in range(frames)],
            [masks_diff[f] if f < masks_diff.shape[0] else np.zeros((height, width), dtype=masks_diff.dtype) for f in range(frames)],
            repeat(data_merged),
            [type]*frames,
            [radius]*frames,
            range(frames)
        ))
        
    kept_rows = []
    mask_out = np.zeros((frames, height, width), dtype=masks_ref.dtype)

    for merged_f, mask_frame, frame in results:
        kept_rows.append(merged_f)
        mask_out[frame] = mask_frame

    final_data = pd.concat(kept_rows, ignore_index=True)
    return final_data, mask_out
        
def _process_frame(df_ref_f, df_diff_f, ref_mask, diff_mask, data_merged, op_type, radius, frame):
    height, width = ref_mask.shape
    ids_to_keep = _frame_set_operation_mask(df_ref_f, df_diff_f, ref_mask, diff_mask, op_type, radius)

    merged_f = data_merged[(data_merged["frame"] == frame) & (data_merged["trackID"].isin(ids_to_keep))]

    mask_out_frame = np.zeros((height, width), dtype=ref_mask.dtype)
    for y in range(height):
        for x in range(width):
            val_r = ref_mask[y, x]
            val_d = diff_mask[y, x]

            if op_type == "difference" and val_r in ids_to_keep:
                mask_out_frame[y, x] = val_r
            elif op_type == "intersect" and val_r in ids_to_keep:
                mask_out_frame[y, x] = val_r
            elif op_type == "union":
                if val_r in ids_to_keep:
                    mask_out_frame[y, x] = val_r
                elif val_d in ids_to_keep:
                    mask_out_frame[y, x] = val_d

    return merged_f, mask_out_frame, frame



def _frame_set_operation_mask(df_ref_f: pd.DataFrame,
                         df_diff_f: pd.DataFrame,
                         ref_mask: np.ndarray,
                         diff_mask: np.ndarray,
                         op_type: str,
                         radius: int = 0) -> set:
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
    radius : int
        The radius (px) within which will be checked if there is an overlapp. defaults to 0 = within center pixel. larger number will increase accuracy but punish performance
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
    shape = ref_mask.shape

    def has_nonzero_in_region(mask, x, y):
        if radius == 0:
            return mask[x, y] != 0
        region = get_circular_mask(shape, (x, y), radius)
        return np.any(mask[region])

    if op_type == "difference":
        for _, row in df_ref_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                if not has_nonzero_in_region(diff_mask, x, y):
                    keep_ids.add(row["trackID"])

    elif op_type == "intersect":
        for _, row in df_ref_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                if has_nonzero_in_region(diff_mask, x, y):
                    keep_ids.add(row["trackID"])

    elif op_type == "union":
        for _, row in df_ref_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                if has_nonzero_in_region(ref_mask, x, y) or has_nonzero_in_region(diff_mask, x, y):
                    keep_ids.add(row["trackID"])

        for _, row in df_diff_f.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                if not has_nonzero_in_region(ref_mask, x, y) and has_nonzero_in_region(diff_mask, x, y):
                    keep_ids.add(row["trackID"])

    return keep_ids

def get_circular_mask(shape, center, radius):
    """create circular mask"""
    y0, x0 = center
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - x0)**2 + (Y - y0)**2
    return dist_sq <= radius**2