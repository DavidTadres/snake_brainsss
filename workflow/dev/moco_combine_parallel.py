import pathlib
import numpy as np
import natsort
import h5py
import matplotlib.pyplot as plt

parallel_path = pathlib.Path('/Users/dtadres/Documents/test_moco')
original_anatomy_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/moco/channel_1_moco.h5')

# preallocate array
stitched_anatomy_brain = np.zeros((256, 128,49, 609), dtype=np.float32)

relevant_z = 25
relevant_t = 60

all_npy_files = []
for current_file in natsort.natsorted(parallel_path.iterdir()):
    if 'npy' in current_file.name and 'channel_1.nii' in current_file.name:
        print(current_file.name)
        index_start = int(current_file.name.split('chunks_')[-1].split('-')[0])
        index_end = int(current_file.name.split('.npy')[0].split('-')[-1])
        print(index_start)
        print(index_end)
        stitched_anatomy_brain[:,:,:,index_start:index_end] = np.load(current_file)

# Read original file
with h5py.File(original_anatomy_path, 'r') as hf:
    original_proxy = hf['data']
    original_slice = original_proxy[:,:,relevant_z,relevant_t]


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.imshow(original_slice.T)

ax2 = fig.add_subplot(222)
ax2.imshow(stitched_anatomy_brain[:,:,relevant_z,relevant_t].T)

# Next plot delta of the two brain
ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
subtracted_brain = (
        stitched_anatomy_brain[:,:,relevant_z,relevant_t].T
        - original_slice.T
)
ax3.imshow(subtracted_brain)
ax3.set_title(
    "subtracted brain slice #" + repr(int(round(stitched_anatomy_brain.shape[2] / 2)))
)
""" Could be used on sherlock but don't want to download whole dataset here.
counts_original, edges_original = np.histogram(original_mean_brain, bins=1000)
counts_my, edges_my = np.histogram(my_mean_brain, bins=1000)
ax4 = fig.add_subplot(224)
ax4.stairs(counts_original, edges_original, fill=True, alpha=1, color="k")
ax4.stairs(counts_my, edges_my, fill=True, alpha=0.5, color="r")
ax4.set_yscale("log")
delta = (
        original_mean_brain - my_mean_brain
)  # what's the difference in value between the two arrays?
ax4.set_title(
    "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
)"""
