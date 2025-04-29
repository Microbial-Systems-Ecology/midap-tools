
import os
import glob
import h5py
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
