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


def compare_moco_results(path_original, path_new, savepath):

    with h5py.File(path_original, 'r') as hf:
        original_proxy = hf['data']
        original_data = original_proxy[:]
        print('first loaded')

        #new_proxy = nib.load(path_new)
        #new_data = np.asarray(new_proxy.dataobj, dtype=np.float32)
        with h5py.File(path_new, 'r') as hf:
            new_data = hf['data']
            new_data = original_proxy[:]
            print('second loaded ')

            diff = []
            for i in range(new_data.shape[-1]):
                print(i)
                diff.append(new_data[:,:,:,i] - original_data[:,:,:,i]) #
            diff = np.asarray(diff)

            mean_over_time = np.nanmean(diff, axis=(1,2,3))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(mean_over_time)
            ax.set_xlabel('frames')
            ax.set_ylabel('difference\nmean pixel intensity')
            fig.savefig(savepath)


