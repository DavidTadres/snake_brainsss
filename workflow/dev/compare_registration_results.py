"""
Problem: registration results are non-deterministic:
https://github.com/ANTsX/ANTsPy/issues/378

I want to get an idea whether two datasets have a systematic difference
Idea: load both datasets, calculate the difference(frame) and take the mean per diff(frame) and plot.
If for example stitching is off, we should see a peak every n frames.
If time is not aligned, diff should increase over time
"""

import numpy as np
import pathlib
import nibabel as nib
import h5py
import matplotlib.pyplot as plt


def compare_moco_results():
    path_original = pathlib.Path(
        '/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/moco/channel_1_moco.h5')
    path_new = pathlib.Path(
        '/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/temp_moco/stitched_ch1.nii')
    savepath = pathlib.Path(
        '/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/testing/time_series_moco_mean.png')

    with h5py.File(path_original, 'r') as hf:
        original_proxy = hf['data']
        # print(loop_proxy.shape)
        # loop_one_slice = loop_proxy[:, :, 3, 50]
        original_data = original_proxy[:]
        print('first loaded')

        new_proxy = nib.load(path_new)
        new_data = np.asarray(new_proxy.dataobj, dtype=np.float32)
        print('second loaded ')

        diff = new_data - original_data #

        mean_over_time = np.nanmean(diff, axis=(0,1,2))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mean_over_time)
        fig.savefig(savepath)


