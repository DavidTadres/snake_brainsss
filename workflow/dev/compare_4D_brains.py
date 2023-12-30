"""
Only run this on sherlock

Here we compare the full brain (x,y,z,t) dataset between the original brainsss and the
snake-brainsss pipeline.
"""

import pathlib
import numpy as np
import nibabel as nib
import h5py
import matplotlib.pyplot as plt


original_fly_path = pathlib.Path(
    "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308"
)
my_fly_paths = pathlib.Path(
    "/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002"
)
my_savepaths = pathlib.Path(my_fly_paths, "testing")


def compared_two_4D_arrays(original_brain_path, my_brain_path, savepath):
    #original_brain_proxy = nib.load(original_brain_path)
    #original_brain = np.asarray(original_brain_proxy.dataobj, np.float32)
    with h5py.File(original_brain_path, "r") as hf:
        original_brain = hf["data"][:] # load whole array in to memory

    #my_brain_proxy = nib.load(my_brain_path)
    #my_brain = np.asarray(my_brain_proxy.dataobj, np.float32)

    with h5py.File(my_brain_path, "r") as hf:
        my_brain = hf["data"][:] # load whole array in to memory

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(original_brain[:,:,int(round(original_brain.shape[2]/2)), 0].T)
    ax1.set_title(original_fly_path.name + ', brain, z=' + repr(int(round(original_brain.shape[2]/2))) + ', t=0')

    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax2.imshow(my_brain[:, :, int(round(my_brain.shape[2] / 2)), 0].T)
    ax2.set_title(my_brain_path.name + ', brain, z=' + repr(int(round(my_brain.shape[2] / 2))) + ', t=0')

    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    subtracted_brain = (my_brain[:,:,int(round(my_brain.shape[2]/2)), 0].T
                        - original_brain[:,:,int(round(original_brain.shape[2]/2)), 0].T)
    ax3.imshow(subtracted_brain)
    ax3.set_title(
        "subtracted brain slice #" + repr(int(round(my_brain.shape[2] / 2)))
    )

    counts_original, edges_original = np.histogram(original_brain, bins=1000)
    counts_my, edges_my = np.histogram(my_brain, bins=1000)
    ax4 = fig.add_subplot(224)
    ax4.stairs(counts_original, edges_original, fill=True, alpha=1, color="k")
    ax4.stairs(counts_my, edges_my, fill=True, alpha=0.5, color="r")
    ax4.set_yscale("log")
    delta = (
        original_brain - my_brain
    )  # what's the difference in value between the two arrays?
    ax4.set_title(
        "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
    )

    # Set title from savename for identification
    fig.suptitle(savepath.name)
    fig.tight_layout()
    savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(savepath)

compared_two_4D_arrays(original_fly_path=pathlib.Path(
    original_fly_path, 'func_0/functional_channel_2_moco_zscore.h5'),
    my_brain_path=pathlib.Path(my_fly_paths, 'func_0/channel_2_moco_zscore.h5'),
    savepath=pathlib.Path(my_savepaths, 'functional_channel_2_moco_zscore.png'))