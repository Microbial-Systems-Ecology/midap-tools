
import os
import glob
import h5py
import re
import numpy as np
import pandas as pd
from PIL import Image

def load_tracking_data(path, group):
    res_path = os.path.join(path,group, "track_output","*.csv")
    res_path = glob.glob(res_path)
    if len(res_path) == 0:
        raise FileNotFoundError(f"Results data  (track_output/*.csv) for {group} at position path {path} does not exist!")
    tracking_file = res_path[0]
    data = pd.read_csv(tracking_file)
    return data

def load_segmentations_h5(path, group, binary = True):
    res_path = os.path.join(path,group, "track_output","segmentations*.h5")
    res_path = glob.glob(res_path)
    if len(res_path) == 0:
        raise FileNotFoundError(f"Segmentation data (track_output/segmentations_*.h5) for {group} at position path {path} does not exist!")
    tracking_file = res_path[0]
    with h5py.File(tracking_file, 'r') as f:
        data = f["segmentations"][:]
    if binary:
        data = data > 0
        
    return data

def load_tracking_h5(path, group):
    res_path = os.path.join(path,group, "track_output","tracking*.h5")
    res_path = glob.glob(res_path)
    if len(res_path) == 0:
        raise FileNotFoundError(f"tracking data (track_output/tracking_*.h5) for {group} at position path {path} does not exist!")
    tracking_file = res_path[0]
    
    with h5py.File(tracking_file, 'r') as f:
        data = f["labels"][:]
    return data

def load_cut_im_stack(path: str, group: str, invert: bool = False) -> np.ndarray:
    """
    Load a sequential stack of cut images from disk into a NumPy array.
    The function expects images named with a frame index pattern
    (e.g. frame0001.png, frame0002.png, ..., frameNNNN.png) located at:

        {path}/{group}/cut_im/

    Images are loaded using PIL, converted to single-channel grayscale,
    sorted numerically by frame index, and stacked into a 3D NumPy array.

    Parameters
    ----------
    path : str
        Root directory containing the group folder.
    group : str
        Group identifier subdirectory.
    invert : bool, optional
        If True, invert image intensities (255 - intensity) such that:
        - black corresponds to high signal
        - white corresponds to low signal


    Returns
    -------
    np.ndarray
        A 3D array of shape (num_frames, height, width) with dtype uint8.

    Raises
    ------
    FileNotFoundError
        If no matching frame images are found.
    ValueError
        If a frame number cannot be extracted from a filename.

    Notes
    -----
    - Frame ordering is determined strictly by the numeric value following
      the substring "frame" in the filename.
    - All images are coerced to 8-bit grayscale ("L" mode).
    - If your data are not single-channel, this function is not suitable
      without modification.
    """

    im_path = os.path.join(path, group, "cut_im", "*frame*.png")
    im_paths = glob.glob(im_path)

    if not im_paths:
        raise FileNotFoundError(f"No images found matching {im_path}")

    frame_re = re.compile(r"frame(\d+)", re.IGNORECASE)

    def frame_index(p: str) -> int:
        m = frame_re.search(os.path.basename(p))
        if not m:
            raise ValueError(f"Cannot extract frame number from {p}")
        return int(m.group(1))

    im_paths.sort(key=frame_index)

    frames = []
    for p in im_paths:
        with Image.open(p) as im:
            im = im.convert("L")  # enforce single-channel grayscale
            arr = np.asarray(im, dtype=np.uint8)

            if invert:
                arr = 255 - arr

            arr = arr / 255  #renoramlize between 0-1
            frames.append(arr)

    return np.stack(frames, axis=0)


def save_tracking_data(data: pd.DataFrame, 
                       path: str, 
                       group: str, 
                       postfix = "custom"):
    """
    Save tracking data DataFrame to CSV in path/group/track_output/track_output_postfix.csv.

    Parameters:
        data (pd.DataFrame): Tracking data to save.
        path (str): Base directory.
        group (str): Subdirectory group name.
        postfix (str): postifx used for the file. defaults to custom
    """
    save_dir = os.path.join(path, group, "track_output")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"track_output_{postfix}.csv")
    data.to_csv(filename, index=False)
    return filename


def save_segmentations_h5(data: np.ndarray, 
                          path: str, 
                          group: str, 
                          binary: bool = True,
                          file_postfix = "custom"):
    """
    Save segmentation data to HDF5 file at path/group/track_output/prefix_postfix.h5

    Parameters:
        data (np.ndarray): The segmentation mask to save.
        path (str): Base directory path.
        group (str): Subdirectory group name.
        binary (bool): If True, data is saved as binary (0/1). Otherwise, values are preserved.
        file_postfix (str): postifx used for the file. defaults to custom
    """
    save_group = "labels"
    file_prefix = "tracking"
    save_dir = os.path.join(path, group, "track_output")
    os.makedirs(save_dir, exist_ok=True)

    if binary:
        data = (data > 0).astype(np.uint8)
        save_group = "segmentations"
        file_prefix = "segmentations"

    filename = os.path.join(save_dir, f"{file_prefix}_{file_postfix}.h5")
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset(save_group, data=data)

    return filename
