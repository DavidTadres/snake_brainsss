
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
mpl.use("agg") # As this should be run on sherlock, use non-interactive backend!

import helper_functions_visualize_brain as hfvb

def compare_clustering():
    warped_brainsss = False # If false, take warped-brain from snake-brainsss!
    savepath = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/testing/clustered_warped_brainsss_' + repr(warped_brainsss) + '.png')

    if warped_brainsss:
        WARP_DIRECTORY = "warp" # YANDAN
        WARP_SUB_DIR_FUNC_TO_ANAT = "func-to-anat_fwdtransforms_2umiso"  # YANDAN
        WARP_SUB_DIR_ANAT_TO_ATLAS =  "anat-to-meanbrain_fwdtransforms_2umiso" # YANDAN
    else:
        #WARP_DIRECTORY = "anat_0/warp" # SNAKE-BRAINS
        #WARP_SUB_DIR = "channel_1_anat-to-channel_jfrc_meanbrain_fwdtransforms_2umiso" # SNAKE-BRAINS
        WARP_DIRECTORY = ""
        WARP_SUB_DIR_FUNC_TO_ANAT =  "func_0/warp/channel_1_func-to-channel_1_anat_fwdtransforms_2umiso"
        WARP_SUB_DIR_ANAT_TO_ATLAS = "anat_0/warp/channel_1_anat-to-channel_jfrc_meanbrain_fwdtransforms_2umiso"

    fps = 100
    resolution = 10 #desired resolution in ms
    behaviors = ['dRotLabY']
    ###
    # BRAINSSSS
    path = pathlib.Path('/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/') # YANDAN

    labels = np.load(pathlib.Path(path, 'func_0/clustering/cluster_labels.npy'))
    signal = np.load(pathlib.Path(path, 'func_0/clustering/cluster_signals.npy'))

    # Organize fictrac data
    fictrac_raw = hfvb.load_fictrac(pathlib.Path(path, 'func_0/fictrac/fictrac-20230525_164921.dat'))
    timestamps = hfvb.load_timestamps(pathlib.Path(path, 'func_0/imaging/functional.xml'))

    expt_len = fictrac_raw.shape[0]/fps*1000
    corrs = []
    for current_behavior in behaviors:
        for z in range(49):
            fictrac_trace = hfvb.smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, current_behavior,
                                                               timestamps[:, z])
            fictrac_trace_L = np.clip(fictrac_trace.flatten(), None, 0) * -1 # Check what this does
            for voxel in range(2000):
                corrs.append(scipy.stats.pearsonr(signal[z, voxel, :], fictrac_trace.flatten())[0])
    fixed = hfvb.load_fda_meanbrain()
    n_clusters = signal.shape[1] # should be 2000
    # Corr brain is the correlation between signals from clustering and behavior!
    whole_corr = np.reshape(np.asarray(corrs),(49,2000))
    reformed_brain = []
    for z in range(49):
        colored_by_betas = np.zeros((256*128))
        # for each cluster number (0, 1...1999)
        for cluster_num in range(n_clusters):
            # Where in the labels is a given
            cluster_indeces = np.where(labels[z,:] == cluster_num)[0]
            colored_by_betas[cluster_indeces] = whole_corr[z, cluster_num]
        colored_by_betas = colored_by_betas.reshape(256,128)
        reformed_brain.append(colored_by_betas)
    STA_brain = np.swapaxes(np.asarray(reformed_brain)[np.newaxis, :, :, :], 0, 1)
    warps_ZPOS = hfvb.warp_STA_brain(STA_brain=STA_brain, fly='fly_308',fixed=fixed,
                                     anat_to_mean_type='myr',
                                     WARP_DIRECTORY=WARP_DIRECTORY,
                                     WARP_SUB_DIR_FUNC_TO_ANAT=WARP_SUB_DIR_FUNC_TO_ANAT,
                                     WARP_SUB_DIR_ANAT_TO_ATLAS=WARP_SUB_DIR_ANAT_TO_ATLAS
                                     )
    atlas = hfvb.load_roi_atlas()
    explosion_rois = hfvb.load_explosion_groups()
    all_rois = hfvb.unnest_roi_groups(explosion_rois)
    roi_masks = hfvb.make_single_roi_masks(all_rois, atlas)
    roi_contours = hfvb.make_single_roi_contours(roi_masks, atlas)
    input_canvas = np.zeros((500,500,3)) #+.5 #.5 for diverging
    data_to_plot = warps_ZPOS[0][:,:,::-1]
    vmax = .2
    explosion_map = hfvb.place_roi_groups_on_canvas(explosion_rois,
                                                                roi_masks,
                                                                roi_contours,
                                                                data_to_plot,
                                                                input_canvas,
                                                                vmax=vmax,
                                                                cmap='hot',
                                                                diverging=False)#'hot')
    plt.imshow(explosion_map[150:,:])

    explosion_map_original = explosion_map.copy()

    ###################################################
    ###################################################
    ###################################################
    ####
    # SNAKE-BRAINS

    path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002')

    labels = np.load(pathlib.Path(path, 'func_0/clustering/channel_2_cluster_labels.npy'))
    signal = np.load(pathlib.Path(path, 'func_0/clustering/channel_2_cluster_signals.npy'))

    # Organize fictrac data
    fictrac_raw = hfvb.load_fictrac(pathlib.Path(path, 'func_0/fictrac/fictrac_behavior_data.dat'))
    timestamps = hfvb.load_timestamps(pathlib.Path(path, 'func_0/imaging/recording_metadata.xml'))
    expt_len = fictrac_raw.shape[0]/fps*1000
    corrs = []
    for current_behavior in behaviors:
        for z in range(49):
            fictrac_trace = hfvb.smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, current_behavior,
                                                               timestamps[:, z])
            fictrac_trace_L = np.clip(fictrac_trace.flatten(), None, 0) * -1 # Check what this does
            for voxel in range(2000):
                corrs.append(scipy.stats.pearsonr(signal[z, voxel, :], fictrac_trace.flatten())[0])
    fixed = hfvb.load_fda_meanbrain()
    n_clusters = signal.shape[1] # should be 2000
    # Corr brain is the correlation between signals from clustering and behavior!
    whole_corr = np.reshape(np.asarray(corrs),(49,2000))
    reformed_brain = []
    for z in range(49):
        colored_by_betas = np.zeros((256*128))
        # for each cluster number (0, 1...1999)
        for cluster_num in range(n_clusters):
            # Where in the labels is a given
            cluster_indeces = np.where(labels[z,:] == cluster_num)[0]
            colored_by_betas[cluster_indeces] = whole_corr[z, cluster_num]
        colored_by_betas = colored_by_betas.reshape(256,128)
        reformed_brain.append(colored_by_betas)
    STA_brain = np.swapaxes(np.asarray(reformed_brain)[np.newaxis, :, :, :], 0, 1)
    warps_ZPOS = hfvb.warp_STA_brain(STA_brain=STA_brain, fly='fly_308',fixed=fixed,
                                anat_to_mean_type='myr',
                                WARP_DIRECTORY=WARP_DIRECTORY,
                                WARP_SUB_DIR=WARP_SUB_DIR)
    atlas = hfvb.load_roi_atlas()
    explosion_rois = hfvb.load_explosion_groups()
    all_rois = hfvb.unnest_roi_groups(explosion_rois)
    roi_masks = hfvb.make_single_roi_masks(all_rois, atlas)
    roi_contours = hfvb.make_single_roi_contours(roi_masks, atlas)
    input_canvas = np.zeros((500,500,3)) #+.5 #.5 for diverging
    data_to_plot = warps_ZPOS[0][:,:,::-1]
    vmax = .2
    explosion_map = hfvb.place_roi_groups_on_canvas(explosion_rois,
                                                                roi_masks,
                                                                roi_contours,
                                                                data_to_plot,
                                                                input_canvas,
                                                                vmax=vmax,
                                                                cmap='hot',
                                                                diverging=False)#'hot')
    plt.imshow(explosion_map[150:,:])

    explosion_map_original = explosion_map.copy()
    corrs = []
    for current_behavior in behaviors:
        for z in range(49):
            fictrac_trace = hfvb.smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, current_behavior,
                                                               timestamps[:, z])
            fictrac_trace_L = np.clip(fictrac_trace.flatten(), None, 0) * -1 # Check what this does
            for voxel in range(2000):
                corrs.append(scipy.stats.pearsonr(signal[z, voxel, :], fictrac_trace.flatten())[0])
    fixed = hfvb.load_fda_meanbrain()
    n_clusters = signal.shape[1] # should be 2000
    # Corr brain is the correlation between signals from clustering and behavior!
    whole_corr = np.reshape(np.asarray(corrs),(49,2000))
    reformed_brain = []
    for z in range(49):
        colored_by_betas = np.zeros((256*128))
        # for each cluster number (0, 1...1999)
        for cluster_num in range(n_clusters):
            # Where in the labels is a given
            cluster_indeces = np.where(labels[z,:] == cluster_num)[0]
            colored_by_betas[cluster_indeces] = whole_corr[z, cluster_num]
        colored_by_betas = colored_by_betas.reshape(256,128)
        reformed_brain.append(colored_by_betas)
    STA_brain = np.swapaxes(np.asarray(reformed_brain)[np.newaxis, :, :, :], 0, 1)
    warps_ZPOS = hfvb.warp_STA_brain(STA_brain=STA_brain, fly='fly_308',fixed=fixed,
                                anat_to_mean_type='myr',
                                WARP_DIRECTORY=WARP_DIRECTORY,
                                WARP_SUB_DIR=WARP_SUB_DIR)
    atlas = hfvb.load_roi_atlas()
    explosion_rois = hfvb.load_explosion_groups()
    all_rois = hfvb.unnest_roi_groups(explosion_rois)
    roi_masks = hfvb.make_single_roi_masks(all_rois, atlas)
    roi_contours = hfvb.make_single_roi_contours(roi_masks, atlas)
    input_canvas = np.zeros((500,500,3)) #+.5 #.5 for diverging
    data_to_plot = warps_ZPOS[0][:,:,::-1]
    vmax = .2
    explosion_map = hfvb.place_roi_groups_on_canvas(explosion_rois,
                                                                roi_masks,
                                                                roi_contours,
                                                                data_to_plot,
                                                                input_canvas,
                                                                vmax=vmax,
                                                                cmap='hot',
                                                                diverging=False)#'hot')
    plt.imshow(explosion_map[150:,:])

    explosion_map_original = explosion_map.copy()

    ###################################################
    ###################################################
    ###################################################

    # Run visualize_brain_my in console
    delta = explosion_map_original - explosion_map

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(explosion_map_original)
    ax1.set_title('brainsss')

    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax2.imshow(explosion_map)
    ax2.set_title('snake-brainsss')

    delta = explosion_map_original-explosion_map
    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    ax3.imshow(delta)

    counts_original, edges_original = np.histogram(explosion_map_original, bins=10)
    counts_my, edges_my = np.histogram(explosion_map, bins=10)
    ax4 = fig.add_subplot(224)
    ax4.stairs(counts_original, edges_original, fill=True, alpha=1, color="k")
    ax4.stairs(counts_my, edges_my, fill=True, alpha=0.5, color="r")
    ax4.set_yscale("log")
    current_ylim = ax4.get_ylim()
    ax4.set_ylim(10 ** 1, current_ylim[-1])

    delta = (
            explosion_map_original - explosion_map
    )  # what's the difference in value between the two arrays?
    ax4.set_title(
        "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
    )

    fig.tight_layout()
    fig.savefig(savepath)