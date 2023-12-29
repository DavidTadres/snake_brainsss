import pathlib
import matplotlib.pyplot as plt
import numpy as np
import h5py

original_fly_path = pathlib.Path(
    "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308"
)
my_fly_paths = pathlib.Path(
    "/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002"
)
my_savepaths = pathlib.Path(my_fly_paths, "testing")

original_motcorr_params = np.load(pathlib.Path(original_fly_path, 'anat_0/moco/motcorr_params.npy'))

my_motcorr_params = np.load(pathlib.Path(my_fly_paths, 'anat_0/moco/motcorr_params.npy'))

diff_motcorr_params = original_motcorr_params - my_motcorr_params
# np.max(np.abs(diff_motcorr_params))
# 0.29351043701171875 # This is not a very small number. But I don't really know what it means...

with h5py.File(pathlib.Path(original_fly_path, 'anat_0/moco/channel_1_moco.h5'), "r") as hf:
    original_brain = hf["data"]  # this doesn't actually LOAD the data - it is just a proxy

with h5py.File(pathlib.Path(my_fly_paths, 'anat_0/moco/channel_1_moco.h5'), 'r') as hf:
    my_brain = hf["data"]

fig = plt.figure()
# First, just plot a sample slice to see gross changes
ax1 = fig.add_subplot(221)