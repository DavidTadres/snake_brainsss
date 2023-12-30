"""
Files are not identical size but this might be because one is saved as compress and the other
is not: https://stackoverflow.com/questions/61028349/why-are-two-h5py-files-different-in-size-when-content-is-the-same
"""

import h5py
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use("agg") # As this should be run on sherlock, use non-interactive backend!

def run_comparison():
    path_loop = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/channel_2_moco_zscore.h5loop.h5')
    path_vec = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/channel_2_moco_zscore.h5')
    #path_vec_original = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/channel_2_moco_zscore_VECT.h5')

    with h5py.File(path_loop, 'r') as hf:
        loop_proxy = hf['data']
        #print(loop_proxy.shape)
        #loop_one_slice = loop_proxy[:, :, 3, 50]
        loop_data = loop_proxy[:]
    print('first loaded')

    with h5py.File(path_vec, 'r') as hf:
        vec_proxy = hf['data']
        #vec_one_slice = vec_proxy[:, :, 3, 50]
        vec_data = vec_proxy
    print('second loaded')

    """with h5py.File(path_vec_original, 'r') as hf:
        vec_orig_proxy = hf['data']
        print(vec_orig_proxy.shape)
        vec_orig_one_slice = vec_orig_proxy[:, :, 3, 50]
    print('third loaded')"""

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(loop_data[:,:,int(loop_data.shape[2]/2), int(loop_data.shape[3]/2)].T)
    ax1.set_title(path_loop.name + ', z=' + repr(int(loop_data.shape[2]/2)) + ', t=' + repr(int(loop_data.shape[3]/2)))

    ax2 = fig.add_subplot(222)
    ax2.imshow(vec_proxy[:,:,int(vec_proxy.shape[2]/2), int(vec_proxy.shape[3]/2)])
    ax2.set_title(path_loop.name + ', z=' + repr(int(vec_proxy.shape[2]/2)) + ', t=' + repr(int(vec_proxy.shape[3]/2)).T)

    delta = loop_data[:,:,int(loop_data.shape[2]/2), int(loop_data.shape[3]/2)] - vec_proxy[:,:,int(vec_proxy.shape[2]/2), int(vec_proxy.shape[3]/2)]
    ax3 = fig.add_subplot(223)
    ax3.imshow(delta.T)
    ax3.set_title('Max delta in this slice' + repr(np.max(delta)))

    # Next, plot histogram of both brains using ALL data (not just a single slice
    counts_original, edges_original = np.histogram(loop_data, bins=1000)
    counts_my, edges_my = np.histogram(vec_proxy, bins=1000)
    ax4 = fig.add_subplot(224)
    ax4.stairs(counts_original, edges_original, fill=True, alpha=1, color="k")
    ax4.stairs(counts_my, edges_my, fill=True, alpha=0.5, color="r")
    ax4.set_yscale("log")
    delta = (
            loop_data - vec_proxy
    )  # what's the difference in value between the two arrays?
    ax4.set_title(
        "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
    )

    fig.savefig(pathlib.Path(path_vec.parent, path_vec.name + '_delta.png'))