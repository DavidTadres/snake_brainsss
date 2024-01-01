# This really should be in another file. Code start line 500
import pathlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from xml.etree import ElementTree as ET
import ants
import matplotlib
import pickle
import cv2


def load_fictrac(fictrac_file_path):
    # def load_fictrac(directory, file='fictrac.dat'):
    """
    NEW:


    ORIGINAL:
    Loads fictrac data from .dat file that fictrac outputs.

    To-do: change units based on diameter of ball etc.
    For speed sanity check, instead remove bad frames so we don't have to throw out whole trial.

    Parameters
    ----------
    directory: string of full path to file
    file: string of file name

    Returns
    -------
    fictrac_data: pandas dataframe of all parameters saved by fictrac"""

    # for item in os.listdir(directory):
    #  if '.dat' in item:
    #    file = item

    # with open(os.path.join(directory, file),'r') as f:
    with open(fictrac_file_path, "r") as f:
        df = pd.DataFrame(l.rstrip().split() for l in f)

        # Name columns
        df = df.rename(
            index=str,
            columns={
                0: "frameCounter",
                1: "dRotCamX",
                2: "dRotCamY",
                3: "dRotCamZ",
                4: "dRotScore",
                5: "dRotLabX",
                6: "dRotLabY",
                7: "dRotLabZ",
                8: "AbsRotCamX",
                9: "AbsRotCamY",
                10: "AbsRotCamZ",
                11: "AbsRotLabX",
                12: "AbsRotLabY",
                13: "AbsRotLabZ",
                14: "positionX",
                15: "positionY",
                16: "heading",
                17: "runningDir",
                18: "speed",
                19: "integratedX",
                20: "integratedY",
                21: "timeStamp",
                22: "sequence",
            },
        )

        # Remove commas
        for column in df.columns.values[:-1]:
            df[column] = [float(x[:-1]) for x in df[column]]

        fictrac_data = df

    # sanity check for extremely high speed (fictrac failure)
    speed = np.asarray(fictrac_data["speed"])
    max_speed = np.max(speed)
    if max_speed > 10:
        raise Exception(
            "Fictrac ball tracking failed (reporting impossibly high speed)."
        )
    return fictrac_data
def load_timestamps(path_to_metadata):
    """
    NEW:
    Parses a Bruker xml file to get the times of each frame
    :param path_to_metadata: a full filepath to a recording_metadata.xml file
    return: timestamps: [t,z] numpy array of times (in ms) of Bruker imaging frames.

    ORIGINAL:

    Parses a Bruker xml file to get the times of each frame, or loads h5py file if it exists.

    First tries to load from 'timestamps.h5' (h5py file). If this file doesn't exist
    it will load and parse the Bruker xml file, and save the h5py file for quick loading in the future.

    Parameters
    ----------
    directory: full directory that contains xml file (str).
    file: Defaults to 'functional.xml'

    Returns
    -------
    timestamps: [t,z] numpy array of times (in ms) of Bruker imaging frames.

    """
    # Not sure if h5py really needs to be made...We only need the timestamps once per call
    # and it's different for each recording...
    # if '.h5' in path_to_metadata:
    # try:
    #    print('Trying to load timestamp data from hdf5 file.')
    #    #with h5py.File(os.path.join(directory, 'timestamps.h5'), 'r') as hf:
    #    with h5py.File(path_to_metadata, 'r') as hf:
    #        timestamps = hf['timestamps'][:]

    # except:
    # else:
    # print('Failed. Extracting frame timestamps from bruker xml file.')
    # xml_file = os.path.join(directory, file)
    # xml_file = pathlib.Path(directory, file)

    tree = ET.parse(path_to_metadata)
    root = tree.getroot()
    timestamps = []

    sequences = root.findall("Sequence")
    for sequence in sequences:
        frames = sequence.findall("Frame")
        for frame in frames:
            # filename = frame.findall('File')[0].get('filename')
            time = float(frame.get("relativeTime"))
            timestamps.append(time)
    timestamps = np.multiply(timestamps, 1000)

    if len(sequences) > 1:
        timestamps = np.reshape(timestamps, (len(sequences), len(frames)))
    else:
        timestamps = np.reshape(timestamps, (len(frames), len(sequences)))

    ### Save h5py file ###
    # with h5py.File(os.path.join(directory, 'timestamps.h5'), 'w') as hf:
    #    hf.create_dataset("timestamps", data=timestamps)

    print("Success.")
    return timestamps

def smooth_and_interp_fictrac(
    fictrac, fps, resolution, expt_len, behavior, timestamps=None, z=None
):  # , smoothing=25, z=None):
    if behavior == "dRotLabZpos":
        behavior = "dRotLabZ"
        clip = "pos"
    elif behavior == "dRotLabZneg":
        behavior = "dRotLabZ"
        clip = "neg"
    else:
        clip = None

    ### get orginal timestamps ###
    camera_rate = 1 / fps * 1000  # camera frame rate in ms
    x_original = np.arange(
        0, expt_len, camera_rate
    )  # same shape as fictrac (e.g. 20980)

    ### smooth ###
    # >>> DANGEROUS - the filter length of the following function is not normalized by the fps
    # e.g. Bella recorded at 100 fps and if we do fictrac_smooth with window length 25 we get
    # filtered data over 10ms * 25 = 250ms.
    # If I record at 50fps each frame is only 20ms. We still filter over 25 points so now we
    # filter over 25*20 = 500ms
    # <<<<
    # I remove the smoothing input from this function and make it dependent on the fps
    smoothing = int(
        np.ceil(0.25 / (1 / fps))
    )  # This will always yield 250 ms (or the next closest
    # possible number, e.g. if we have 50fps we would get a smotthing window of 12.5 which we can't
    # index of course. We always round up so with 50 fps we'd get 13 = 260 ms
    fictrac_smoothed = scipy.signal.savgol_filter(
        np.asarray(fictrac[behavior]), smoothing, 3
    )
    # Identical shape in output as input, e.g. 20980

    ### clip if desired ###
    if clip == "pos":
        fictrac_smoothed = np.clip(
            fictrac_smoothed, a_min=0, a_max=None
        )  # Unsure what this does
    elif clip == "neg":
        fictrac_smoothed = (
            np.clip(fictrac_smoothed, a_min=None, a_max=0) * -1
        )  # Unsure what this does

    ### interpolate ###
    # This function probably just returns everything from an input array
    fictrac_interp_temp = scipy.interpolate.interp1d(
        x_original, fictrac_smoothed, bounds_error=False
    )  # yields a function
    xnew = np.arange(0, expt_len, resolution)  # 0 to last time at subsample res
    # ## different number, e.g. 41960, or just 2x shape before.
    # This is probably because resolution is set to 10. If framerate is 50 we have a frame every 20 ms.
    if timestamps is None:
        fictrac_interp = fictrac_interp_temp(xnew)
    elif z is not None:  # For testing only!
        fictrac_interp = fictrac_interp_temp(timestamps[:, z])
    else:
        # So we only select which timestamps here.
        # fictrac_interp = fictrac_interp_temp(timestamps[:,z]) # This would return ALL timestamps per z slice
        fictrac_interp = fictrac_interp_temp(timestamps)

    ### convert units for common cases ###
    sphere_radius = 4.5e-3  # in m
    if behavior in ["dRotLabY"]:
        """starts with units of rad/frame
        * sphere_radius(m); now in m/frame
        * fps; now in m/sec
        * 1000; now in mm/sec"""

        fictrac_interp = fictrac_interp * sphere_radius * fps * 1000  # now in mm/sec

    if behavior in ["dRotLabZ"]:
        """starts with units of rad/frame
        * 180 / np.pi; now in deg/frame
        * fps; now in deg/sec"""

        fictrac_interp = fictrac_interp * 180 / np.pi * fps

    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False)

    return fictrac_interp
def load_fda_meanbrain():
    fixed_path = "/Volumes/groups/trc/data/Brezovec/2P_Imaging/anat_templates/20220301_luke_2_jfrc_affine_zflip_2umiso.nii"  # luke.nii"
    fixed_resolution = (2, 2, 2)
    fixed = np.asarray(nib.load(fixed_path).get_fdata().squeeze(), dtype="float32")
    fixed = ants.from_numpy(fixed)
    fixed.set_spacing(fixed_resolution)
    return fixed


def warp_STA_brain(STA_brain, fly, fixed, anat_to_mean_type,
                   WARP_DIRECTORY,
                   WARP_SUB_DIR_FUNC_TO_ANAT,
                   WARP_SUB_DIR_ANAT_TO_ATLAS,
                   DATASET_PATH):
    import os
    n_tp = STA_brain.shape[1]
    dataset_path = DATASET_PATH
    moving_resolution = (2.611, 2.611, 5)
    ###########################
    ### Organize Transforms ###
    ###########################
    #warp_directory = os.path.join(dataset_path, fly, "warp")
    warp_directory = os.path.join(dataset_path, fly, WARP_DIRECTORY)
    #warp_sub_dir = "func-to-anat_fwdtransforms_2umiso"
    warp_sub_dir = WARP_SUB_DIR_FUNC_TO_ANAT
    affine_file = os.listdir(os.path.join(warp_directory, warp_sub_dir))[0]
    affine_path = os.path.join(warp_directory, warp_sub_dir, affine_file)
    #if anat_to_mean_type == "myr":
    #    warp_sub_dir = "anat-to-meanbrain_fwdtransforms_2umiso"
    #elif anat_to_mean_type == "non_myr":
    #    warp_sub_dir = "anat-to-non_myr_mean_fwdtransforms_2umiso"
    #else:
    #    print("invalid anat_to_mean_type")
    #    return
    warp_sub_dir = WARP_SUB_DIR_ANAT_TO_ATLAS
    syn_files = os.listdir(os.path.join(warp_directory, warp_sub_dir))
    syn_linear_path = os.path.join(
        warp_directory, warp_sub_dir, [x for x in syn_files if ".mat" in x][0]
    )
    syn_nonlinear_path = os.path.join(
        warp_directory, warp_sub_dir, [x for x in syn_files if ".nii.gz" in x][0]
    )
    ####transforms = [affine_path, syn_linear_path, syn_nonlinear_path]
    transforms = [
        syn_nonlinear_path,
        syn_linear_path,
        affine_path,
    ]  ### INVERTED ORDER ON 20220503!!!!
    # ANTS DOCS ARE SHIT. THIS IS PROBABLY CORRECT, AT LEAST IT NOW WORKS FOR THE FLY(134) THAT WAS FAILING

    ### Warp timeponts
    warps = []
    for tp in range(n_tp):
        to_warp = np.rollaxis(STA_brain[:, tp, :, :], 0, 3)
        moving = ants.from_numpy(to_warp)
        moving.set_spacing(moving_resolution)
        ########################
        ### Apply Transforms ###
        ########################
        moco = ants.apply_transforms(fixed, moving, transforms)
        warped = moco.numpy()
        warps.append(warped)

    return warps
def load_roi_atlas():
    atlas_path = "/Volumes/groups/trc/data/Brezovec/2P_Imaging/anat_templates/jfrc_2018_rois_improve_reorient_transformed.nii"
    atlas = np.asarray(nib.load(atlas_path).get_fdata().squeeze(), dtype="float32")
    atlas = ants.from_numpy(atlas)
    atlas.set_spacing((0.76, 0.76, 0.76))
    atlas = ants.resample_image(atlas, (2, 2, 2), use_voxels=False)
    atlas = atlas.numpy()
    atlas_int = np.rint(atlas)
    atlas_clean = np.copy(atlas_int)
    diff_atlas = atlas_int - atlas
    thresh = 0.001
    atlas_clean[np.where(np.abs(diff_atlas) > thresh)] = 0
    return atlas_clean

def load_explosion_groups():

    explosion_rois_file = "/Volumes/groups/trc/data/Brezovec/2P_Imaging/anat_templates/20220425_explosion_plot_rois.pickle"
    explosion_rois = pickle.load(open(explosion_rois_file, "rb"))
    return explosion_rois
def unnest_roi_groups(explosion_rois):
    all_rois = []
    for roi_group in explosion_rois:
        all_rois.extend(explosion_rois[roi_group]["rois"].keys())
    return all_rois
def make_single_roi_masks(all_rois, atlas):
    """
    <input>:39: DeprecationWarning:
    Please use `binary_erosion` from the `scipy.ndimage` namespace, the `scipy.ndimage.morphology` namespace is deprecated.
    <input>:40: DeprecationWarning:
    Please use `binary_dilation` from the `scipy.ndimage` namespace, the `scipy.ndimage.morphology` namespace is deprecated.

    :param all_rois:
    :param atlas:
    :return:
    """
    masks = {}
    for roi in all_rois:
        mask = np.zeros(atlas.shape)
        mask[np.where(atlas == roi)] = 1
        mask_eroded = scipy.ndimage.morphology.binary_erosion(mask, structure=np.ones((2, 2, 2)))
        mask_dilated = scipy.ndimage.morphology.binary_dilation(mask_eroded, iterations=2)
        masks[roi] = mask_dilated
    return masks
def make_single_roi_contours(roi_masks, atlas):
    roi_contours = {}
    for roi in roi_masks:
        mask = roi_masks[roi]
        _, mask_binary = cv2.threshold(
            np.max(mask, axis=-1).astype("uint8"), 0, 1, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # cv2.RETR_TREE

        canvas = np.ones(atlas[:, :, 0].shape)
        out = cv2.drawContours(canvas, contours, -1, (0, 255, 0), 1)
        out = np.abs(out - 1)  # flip 0/1
        roi_contour = np.repeat(
            out[:, :, np.newaxis], repeats=4, axis=-1
        )  ### copy into rgba channels to make white

        # get edge location
        left_edge = np.where(np.sum(np.nan_to_num(roi_contour), axis=0) > 0)[0][0]
        right_edge = np.where(np.sum(np.nan_to_num(roi_contour), axis=0) > 0)[0][-1]
        top_edge = np.where(np.sum(np.nan_to_num(roi_contour), axis=1) > 0)[0][0]
        bottom_edge = np.where(np.sum(np.nan_to_num(roi_contour), axis=1) > 0)[0][-1]

        roi_contours[roi] = {}
        roi_contours[roi]["contour"] = roi_contour
        roi_contours[roi]["left_edge"] = left_edge
        roi_contours[roi]["right_edge"] = right_edge
        roi_contours[roi]["top_edge"] = top_edge
        roi_contours[roi]["bottom_edge"] = bottom_edge
    return roi_contours

def get_dim_info(item, full_x_mid, full_y_mid):
    y_mid = int(item.shape[0] / 2)
    x_mid = int(item.shape[1] / 2)

    height = item.shape[0]
    width = item.shape[1]

    left = full_x_mid - x_mid
    right = left + width

    top = full_y_mid - y_mid
    bottom = top + height
    return {"left": left, "right": right, "top": top, "bottom": bottom}
def place_roi_groups_on_canvas(
    explosion_rois,
    roi_masks,
    roi_contours,
    data_to_plot,
    input_canvas,
    vmax,
    cmap,
    diverging=False,
):
    full_y_mid = int(input_canvas.shape[0] / 2)
    full_x_mid = int(input_canvas.shape[1] / 2)

    for roi_group in explosion_rois:
        x_shift = explosion_rois[roi_group]["x_shift"]
        y_shift = explosion_rois[roi_group]["y_shift"]

        roi_data = []
        left_edges = []
        right_edges = []
        bottom_edges = []
        top_edges = []

        for roi in explosion_rois[roi_group]["rois"]:
            mask = roi_masks[roi]
            # masked_roi = mask[...,np.newaxis]*data_to_plot # for 3 channel
            masked_roi = mask * data_to_plot

            ### maximum projection along z-axis
            # works for negative values
            maxs = np.max(masked_roi, axis=2)
            mins = np.min(masked_roi, axis=2)
            maxs[np.where(np.abs(mins) > maxs)] = mins[np.where(np.abs(mins) > maxs)]
            roi_data.append(maxs)
            # masked_roi_flat = maxs

            ### maximum projection along z-axis
            # masked_roi_flat = np.max(masked_roi,axis=2)
            # roi_data.append(masked_roi_flat)

            left_edges.append(roi_contours[roi]["left_edge"])
            right_edges.append(roi_contours[roi]["right_edge"])
            top_edges.append(roi_contours[roi]["top_edge"])
            bottom_edges.append(roi_contours[roi]["bottom_edge"])

        # get extreme edges from all rois used
        left_edge = np.min(left_edges) - 1
        right_edge = np.max(right_edges) + 1
        top_edge = np.min(top_edges) - 1
        bottom_edge = np.max(bottom_edges) + 1

        ### this projects across all the roi_data from each roi
        # roi_datas = np.max(np.asarray(roi_data),axis=0) # this one line is sufficient for not diverging
        maxs = np.max(np.asarray(roi_data), axis=0)
        mins = np.min(np.asarray(roi_data), axis=0)
        maxs[np.where(np.abs(mins) > maxs)] = mins[np.where(np.abs(mins) > maxs)]
        roi_datas = maxs

        ###ADD MAX MIN HERE LIKE ABOVE

        ### cutout this grouping
        # data_map = np.swapaxes(roi_datas[top_edge:bottom_edge,left_edge:right_edge,:],0,1) # for 3 channel
        data_map = np.swapaxes(
            roi_datas[top_edge:bottom_edge, left_edge:right_edge], 0, 1
        )
        ### apply gain
        # data_map = data_map * gain
        """
        <input>:160: MatplotlibDeprecationWarning:
        The get_cmap function was deprecated in Matplotlib 3.7 and will be removed two minor releases later. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap(obj)`` instead.
        """
        mycmap = matplotlib.cm.get_cmap(cmap)
        # mycmap.set_bad('k',1) # make nans black

        if diverging:
            # this will normalize all value to [0,1], with 0.5 being the new "0" basically
            # current issue - a zero that should be background now looks like negative.
            # solution: could use nans instead and set bad color
            # with diverging we should make background white!
            # so actually just set the input_canvas as 0.5!!!
            # then make contours nan and set back as black
            norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
            data_map = norm(data_map)
        else:
            data_map = data_map / vmax

        data_map = mycmap(data_map)[..., :3]  # lose alpha channel

        dims = get_dim_info(data_map, full_x_mid, full_y_mid)

        ### ADD TO CANVAS
        input_canvas[
            dims["top"] + y_shift : dims["bottom"] + y_shift,
            dims["left"] + x_shift : dims["right"] + x_shift,
            :3,
        ] = data_map

        ### ADD CONTOUR TO CANVAS
        for roi in explosion_rois[roi_group]["rois"]:
            contour = roi_contours[roi]["contour"]
            contour = np.swapaxes(
                contour[top_edge:bottom_edge, left_edge:right_edge], 0, 1
            )
            ys = np.where(contour[:, :, 0] > 0)[0] + dims["top"] + y_shift
            xs = np.where(contour[:, :, 0] > 0)[1] + dims["left"] + x_shift
            input_canvas[ys, xs] = 0  # 1
    return input_canvas
##

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
mpl.use("agg") # As this should be run on sherlock, use non-interactive backend!



#import helper_functions_visualize_brain as hfvb
#############
# Path and stuff, for both!
warped_brainsss = True # If false, take warped-brain from snake-brainsss!
savepath = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/testing/clustered_warped_brainsss_' + repr(warped_brainsss) + '.png')

WARP_DIRECTORY = "warp"  # YANDAN
WARP_SUB_DIR_FUNC_TO_ANAT = "func-to-anat_fwdtransforms_2umiso"  # YANDAN
WARP_SUB_DIR_ANAT_TO_ATLAS = "anat-to-meanbrain_fwdtransforms_2umiso"  # YANDAN
STA_WARP_DATASET_PATH = "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"

# The next part is 'my
if warped_brainsss:
    WARP_DIRECTORY_MY = "warp" # YANDAN
    WARP_SUB_DIR_FUNC_TO_ANAT_MY = "func-to-anat_fwdtransforms_2umiso"  # YANDAN
    WARP_SUB_DIR_ANAT_TO_ATLAS_MY =  "anat-to-meanbrain_fwdtransforms_2umiso" # YANDAN
else:
    #WARP_DIRECTORY = "anat_0/warp" # SNAKE-BRAINS
    #WARP_SUB_DIR = "channel_1_anat-to-channel_jfrc_meanbrain_fwdtransforms_2umiso" # SNAKE-BRAINS
    WARP_DIRECTORY_MY = ""
    WARP_SUB_DIR_FUNC_TO_ANAT_MY =  "func_0/warp/channel_1_func-to-channel_1_anat_fwdtransforms_2umiso"
    WARP_SUB_DIR_ANAT_TO_ATLAS_MY = "anat_0/warp/channel_1_anat-to-channel_jfrc_meanbrain_fwdtransforms_2umiso"

path_my = pathlib.Path('/Volumes'
                    '/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002')
STA_WARP_DATASET_PATH_MY = '/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato'

#################

fps = 100
resolution = 10 #desired resolution in ms
behaviors = ['dRotLabY']
###
# BRAINSSSS
path = pathlib.Path('/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/') # YANDAN

labels = np.load(pathlib.Path(path, 'func_0/clustering/cluster_labels.npy'))
signal = np.load(pathlib.Path(path, 'func_0/clustering/cluster_signals.npy'))

# Organize fictrac data
fictrac_raw = load_fictrac(pathlib.Path(path, 'func_0/fictrac/fictrac-20230525_164921.dat'))
timestamps = load_timestamps(pathlib.Path(path, 'func_0/imaging/functional.xml'))

expt_len = fictrac_raw.shape[0]/fps*1000
corrs = []
for current_behavior in behaviors:
    for z in range(49):
        fictrac_trace = smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, current_behavior,
                                                           timestamps[:, z])
        fictrac_trace_L = np.clip(fictrac_trace.flatten(), None, 0) * -1 # Check what this does
        for voxel in range(2000):
            corrs.append(scipy.stats.pearsonr(signal[z, voxel, :], fictrac_trace.flatten())[0])
fixed = load_fda_meanbrain()
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
warps_ZPOS = warp_STA_brain(STA_brain=STA_brain, fly='fly_308',fixed=fixed,
                                 anat_to_mean_type='myr',
                                 WARP_DIRECTORY=WARP_DIRECTORY,
                                 WARP_SUB_DIR_FUNC_TO_ANAT=WARP_SUB_DIR_FUNC_TO_ANAT,
                                 WARP_SUB_DIR_ANAT_TO_ATLAS=WARP_SUB_DIR_ANAT_TO_ATLAS,
                                 DATASET_PATH=STA_WARP_DATASET_PATH
                                 )
atlas = load_roi_atlas()
explosion_rois = load_explosion_groups()
all_rois = unnest_roi_groups(explosion_rois)
roi_masks = make_single_roi_masks(all_rois, atlas)
roi_contours = make_single_roi_contours(roi_masks, atlas)
input_canvas = np.zeros((500,500,3)) #+.5 #.5 for diverging
data_to_plot = warps_ZPOS[0][:,:,::-1]
vmax = .2
explosion_map = place_roi_groups_on_canvas(explosion_rois,
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



labels = np.load(pathlib.Path(path_my, 'func_0/clustering/channel_2_cluster_labels.npy'))
signal = np.load(pathlib.Path(path_my, 'func_0/clustering/channel_2_cluster_signals.npy'))

# Organize fictrac data
fictrac_raw = load_fictrac(pathlib.Path(path_my, 'func_0/fictrac/fictrac_behavior_data.dat'))
timestamps = load_timestamps(pathlib.Path(path_my, 'func_0/imaging/recording_metadata.xml'))
expt_len = fictrac_raw.shape[0]/fps*1000

corrs = []
for current_behavior in behaviors:
    for z in range(49):
        fictrac_trace = smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, current_behavior,
                                                           timestamps[:, z])
        fictrac_trace_L = np.clip(fictrac_trace.flatten(), None, 0) * -1 # Check what this does
        for voxel in range(2000):
            corrs.append(scipy.stats.pearsonr(signal[z, voxel, :], fictrac_trace.flatten())[0])
fixed = load_fda_meanbrain()
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
warps_ZPOS = warp_STA_brain(STA_brain=STA_brain, fly='fly_002',fixed=fixed,
                            anat_to_mean_type='myr',
                            WARP_DIRECTORY=WARP_DIRECTORY_MY,
                            WARP_SUB_DIR_FUNC_TO_ANAT=WARP_SUB_DIR_FUNC_TO_ANAT_MY,
                            WARP_SUB_DIR_ANAT_TO_ATLAS=WARP_SUB_DIR_ANAT_TO_ATLAS_MY,
                            STA_WARP_DATASET_PATH=STA_WARP_DATASET_PATH_MY)
atlas = load_roi_atlas()
explosion_rois = load_explosion_groups()
all_rois = unnest_roi_groups(explosion_rois)
roi_masks = make_single_roi_masks(all_rois, atlas)
roi_contours = make_single_roi_contours(roi_masks, atlas)
input_canvas = np.zeros((500,500,3)) #+.5 #.5 for diverging
data_to_plot = warps_ZPOS[0][:,:,::-1]
vmax = .2
explosion_map = place_roi_groups_on_canvas(explosion_rois,
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

