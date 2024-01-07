
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
mpl.use("agg") # As this should be run on sherlock, use non-interactive backend!

import helper_functions_visualize_brain as hfvb

#import helper_functions_visualize_brain as hfvb
#############
# Path and stuff, for both!
warped_brainsss = True # If false, take warped-brain from snake-brainsss!

# Load only once
fixed = hfvb.load_fda_meanbrain()
atlas = hfvb.load_roi_atlas()
explosion_rois = hfvb.load_explosion_groups()
all_rois = hfvb.unnest_roi_groups(explosion_rois)
roi_masks = hfvb.make_single_roi_masks(all_rois, atlas)
roi_contours = hfvb.make_single_roi_contours(roi_masks, atlas)
input_canvas = np.zeros((500,500,3)) #+.5 #.5 for diverging
print('prep complete')

#################
# general variables
savepath = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_004/testing/negative_control_(fly309)_clustered_warped_brainsss_' + repr(warped_brainsss) + '.png')

WARP_DIRECTORY = "warp"  # YANDAN
WARP_SUB_DIR_FUNC_TO_ANAT = "func-to-anat_fwdtransforms_2umiso"  # YANDAN
WARP_SUB_DIR_ANAT_TO_ATLAS = "anat-to-meanbrain_fwdtransforms_2umiso"  # YANDAN
STA_WARP_DATASET_PATH = "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"
FLY = 'fly_309'
path_orig = pathlib.Path('/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_309/')  # YANDAN
######
# brainsss data loading

label_path_orig = pathlib.Path(path_orig, 'func_0/clustering/cluster_labels.npy')
signal_path_orig = pathlib.Path(path_orig, 'func_0/clustering/cluster_signals.npy')
fictrac_path_orig = pathlib.Path(path_orig, 'func_0/fictrac/fictrac-20230525_164921.dat')
timestamps_path_orig = pathlib.Path(path_orig, 'func_0/imaging/functional.xml')

explosion_plot_original = hfvb.prepare_brain(label_path_orig, signal_path_orig,
                                             fictrac_path_orig, timestamps_path_orig,
                                             WARP_DIRECTORY,
                                             WARP_SUB_DIR_FUNC_TO_ANAT,
                                             WARP_SUB_DIR_ANAT_TO_ATLAS,
                                             STA_WARP_DATASET_PATH,
                                             FLY,
                                             fixed,
                                             explosion_rois,
                                             roi_masks,
                                             roi_contours,
                                             input_canvas
                                             )
print('first explosion plot finished')
###################################################
####
# SNAKE-BRAINS
##
# The next part is 'my'
if warped_brainsss:
    WARP_DIRECTORY_MY = WARP_DIRECTORY
    WARP_SUB_DIR_FUNC_TO_ANAT_MY = WARP_SUB_DIR_FUNC_TO_ANAT
    WARP_SUB_DIR_ANAT_TO_ATLAS_MY =  WARP_SUB_DIR_ANAT_TO_ATLAS
    STA_WARP_DATASET_PATH_MY = STA_WARP_DATASET_PATH
    FLY_MY = FLY
else:
    WARP_DIRECTORY_MY = ""
    WARP_SUB_DIR_FUNC_TO_ANAT_MY =  "func_0/warp/channel_1_func-to-channel_1_anat_fwdtransforms_2umiso"
    WARP_SUB_DIR_ANAT_TO_ATLAS_MY = "anat_0/warp/channel_1_anat-to-channel_jfrc_meanbrain_fwdtransforms_2umiso"
    STA_WARP_DATASET_PATH_MY = '/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato'
    FLY_MY = 'fly_004'

path_my = pathlib.Path('/Volumes'
                    '/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_004')
label_path_my = pathlib.Path(path_my, 'func_0/clustering/channel_2_cluster_labels.npy')
signal_path_my = pathlib.Path(path_my, 'func_0/clustering/channel_2_cluster_signals.npy')
fictrac_path_my = pathlib.Path(path_my, 'func_0/fictrac/fictrac_behavior_data.dat')
timestamps_path_my = pathlib.Path(path_my, 'func_0/imaging/recording_metadata.xml')


explosion_plot_my = hfvb.prepare_brain(label_path_my, signal_path_my,
                                       fictrac_path_my, timestamps_path_my,
                                       WARP_DIRECTORY_MY,
                                       WARP_SUB_DIR_FUNC_TO_ANAT_MY,
                                       WARP_SUB_DIR_ANAT_TO_ATLAS_MY,
                                       STA_WARP_DATASET_PATH_MY,
                                       FLY_MY,
                                       fixed,
                                       explosion_rois,
                                       roi_masks,
                                       roi_contours,
                                       input_canvas
                                       )
###################################################
###################################################
###################################################

# Run visualize_brain_my in console
delta = explosion_plot_original - explosion_plot_my

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.imshow(explosion_plot_original)
ax1.set_title('brainsss')

ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax2.imshow(explosion_plot_my)
ax2.set_title('snake-brainsss')

delta = explosion_plot_original-explosion_plot_my
ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
ax3.imshow(delta)

counts_original, edges_original = np.histogram(explosion_plot_original, bins=10)
counts_my, edges_my = np.histogram(explosion_plot_my, bins=10)
ax4 = fig.add_subplot(224)
ax4.stairs(counts_original, edges_original, fill=True, alpha=1, color="k")
ax4.stairs(counts_my, edges_my, fill=True, alpha=0.5, color="r")
ax4.set_yscale("log")
current_ylim = ax4.get_ylim()
ax4.set_ylim(10 ** 1, current_ylim[-1])

delta = (
        explosion_plot_original - explosion_plot_my
)  # what's the difference in value between the two arrays?
ax4.set_title(
    "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
)

fig.tight_layout()
fig.savefig(savepath)

