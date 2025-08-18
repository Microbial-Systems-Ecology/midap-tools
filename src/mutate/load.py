
import os
import glob
import h5py
import numpy as np
import pandas as pd


def load_tracking_data(path, group):
    res_path = os.path.join(path,group, "track_output","*.csv")
    tracking_file = glob.glob(res_path)[0]
    data = pd.read_csv(tracking_file)
    return data

def load_segmentations_h5(path, group, binary = True):
    res_path = os.path.join(path,group, "track_output","segmentations*.h5")
    tracking_file = glob.glob(res_path)[0]
    with h5py.File(tracking_file, 'r') as f:
        data = f["segmentations"][:]
    if binary:
        data = data > 0
        
    return data

def load_tracking_h5(path, group):
    res_path = os.path.join(path,group, "track_output","tracking*.h5")
    tracking_file = glob.glob(res_path)[0]
    
    with h5py.File(tracking_file, 'r') as f:
        data = f["labels"][:]
    return data


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
