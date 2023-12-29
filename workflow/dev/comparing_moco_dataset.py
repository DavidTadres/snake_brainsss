import pathlib
import matplotlib.pyplot as plt
import h5py
import numpy as np


original_fly_path = pathlib.Path(
    "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308"
)
my_fly_paths = pathlib.Path(
    "/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002"
)
my_savepaths = pathlib.Path(my_fly_paths, "testing")

def compare_two_h5py_vol_arrays(original_brain_path, my_brain_path, savepath):
    with h5py.File(original_brain_path, "r") as hf:
        original_moco_brain = hf["data"][:] # this should load the data