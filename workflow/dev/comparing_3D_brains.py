"""
I am using a test dataset from Yandan where I use her 'import' data to run a whole dataset through
snake-brainsss.
Code here is to prepare plots and compare arrays for the comparison
"""
import nibabel as nib
import pathlib
import matplotlib.pyplot as plt

import numpy as np

original_fly_path = pathlib.Path(
    "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308"
)
my_fly_paths = pathlib.Path(
    "/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002"
)
my_savepaths = pathlib.Path(my_fly_paths, "testing")


def compare_two_3D_arrays(original_brain_path, my_brain_path, savepath):
    # Compare mean brain results
    original_mean_brain_proxy = nib.load(original_brain_path)
    original_mean_brain = np.asarray(original_mean_brain_proxy.dataobj, np.float32)

    my_mean_brain_proxy = nib.load(my_brain_path)
    my_mean_brain = np.asarray(my_mean_brain_proxy.dataobj, np.float32)

    fig = plt.figure()
    # First, just plot a sample slice to see gross changes
    ax1 = fig.add_subplot(221)
    ax1.imshow(
        original_mean_brain[:, :, int(round(original_mean_brain.shape[2] / 2))].T
    )
    ax1.set_title(
        original_fly_path.name
        + ", mean brain, z="
        + repr(int(round(original_mean_brain.shape[2] / 2)))
    )
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax2.imshow(my_mean_brain[:, :, int(round(my_mean_brain.shape[2] / 2))].T)
    ax2.set_title(
        my_fly_paths.name
        + ", mean brain, z="
        + repr(int(round(my_mean_brain.shape[2] / 2)))
    )

    # Next plot delta of the two brain
    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    subtracted_brain = (
        my_mean_brain[:, :, int(round(my_mean_brain.shape[2] / 2))].T
        - original_mean_brain[:, :, int(round(original_mean_brain.shape[2] / 2))].T
    )
    ax3.imshow(subtracted_brain)
    ax3.set_title(
        "subtracted brain slice #" + repr(int(round(my_mean_brain.shape[2] / 2)))
    )

    # Next, plot histogram of both brains using ALL data (not just a single slice
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
    )

    # Set title from savename for identification
    fig.suptitle(savepath.name)
    fig.tight_layout()
    savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(savepath)

'''
compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/imaging/functional_channel_1_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/imaging/channel_1_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_func_0_imaging_ch1.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/imaging/functional_channel_2_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/imaging/channel_2_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_func_0_imaging_ch2.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "anat_0/imaging/anatomy_channel_1_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "anat_0/imaging/channel_1_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_anat_0_imaging_ch1.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "anat_0/imaging/anatomy_channel_2_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "anat_0/imaging/channel_2_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_anat_0_imaging_ch2.png"),
)
'''
#######
# MOCO MEAN BRAINS
#######
'''
compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "anat_0/moco/anatomy_channel_1_moc_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "anat_0/moco/channel_1_moco_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_anat_0_moco_ch1.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "anat_0/moco/anatomy_channel_2_moc_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "anat_0/moco/channel_2_moco_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_anat_0_moco_ch2.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/moco/functional_channel_1_moc_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/moco/channel_1_moco_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_func_0_moco_ch1.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/moco/functional_channel_2_moc_mean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/moco/channel_2_moco_mean.nii"),
    savepath=pathlib.Path(my_savepaths, "meanbrain_func_0_moco_ch2.png"),
)

'''
'''
###
# BEHAVIOR-Z SCORE CORRELATED BRAINS
###
compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/corr/20220420_corr_dRotLabY.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/corr/channel_2_corr_dRotLabY.nii"),
    savepath=pathlib.Path(my_savepaths, "corr_dRotLabY_func_0_ch2.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/corr/20220420_corr_dRotLabZneg.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/corr/channel_2_corr_dRotLabZneg.nii"),
    savepath=pathlib.Path(my_savepaths, "corr_dRotLabZ_func_0_ch2.png"),
)

compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "func_0/corr/20220420_corr_dRotLabZpos.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "func_0/corr/channel_2_corr_dRotLabZpos.nii"),
    savepath=pathlib.Path(my_savepaths, "corr_dRotLabZpos_func_0_ch2.png"),
)
'''
####
# Clean anatomy
#####
compare_two_3D_arrays(
    original_brain_path=pathlib.Path(
        original_fly_path, "anat_0/moco/anatomy_channel_1_moc_mean_clean.nii"
    ),
    my_brain_path=pathlib.Path(my_fly_paths, "anat_0/moco/channel_1_moco_mean_clean.nii"),
    savepath=pathlib.Path(my_savepaths, "channel_1_moco_mean_clean.png"),
)