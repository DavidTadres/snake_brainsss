import numpy as np
import pathlib
import sys
import time
import traceback
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("agg")  # Agg, is a non-interactive backend that can only write to files.
# Without this I had the following error: Starting a Matplotlib GUI outside of the main thread will likely fail.
import nibabel as nib
import shutil
import ants
import h5py
from scipy import ndimage
import sklearn
import skimage.filters
import sklearn.feature_extraction
import sklearn.cluster

# To import files (or 'modules') from the brainsss folder, define path to scripts!
# path of workflow i.e. /Users/dtadres/snake_brainsss/workflow
#scripts_path = pathlib.Path(
#    __file__
#).parent.resolve()
#sys.path.insert(0, pathlib.Path(scripts_path, "workflow"))
parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
# This just imports '*.py' files from the folder 'brainsss'.
from brainsss import utils
from brainsss import fictrac_utils
from brainsss import corr_utils
from brainsss import align_utils


####################
# GLOBAL VARIABLES #
####################
WIDTH = 120  # This is used in all logging files
# Bruker gives us data going from 0-8191 so we load it as uint16.
# However, as soon as we do the ants.registration step (e.g in
# function motion_correction() we get floats back.
# The original brainsss usually saved everything in float32 but some
# calculations were done with float64 (e.g. np.mean w/o defining dtype).
# To be more explicit, I define here two global variables.
# Dtype is what was mostly called when saving and loading data
DTYPE = np.float32
# Dtype_calculation is what I explicity call e.g. during np.mean
DTYPE_CACLULATIONS = np.float32


def make_supervoxels(
    fly_directory,
    path_to_read,
    save_path_cluster_labels,
    save_path_cluster_signals,
    n_clusters,
):
    """
    :param fly_directory:
    :param path_to_read:
    :param save_path:
    :return:
    """
    CLUSTER_3D = False # No need to try: I learned why clustering was done per plane:
    # Goal was to have highest possible time resolution and we ahve ~20ms between slices
    # and up to half a second for a volume. If we clustered by volume we'd smear the time signal
    # by at least half a second!
    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="make_supervoxels")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "make_supervoxels")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    path_to_read = utils.convert_list_of_string_to_posix_path(path_to_read)
    save_path_cluster_labels = utils.convert_list_of_string_to_posix_path(
        save_path_cluster_labels
    )
    save_path_cluster_signals = utils.convert_list_of_string_to_posix_path(
        save_path_cluster_signals
    )

    # Can have more than one functional channel, hence loop!
    for (
        current_path_to_read,
        current_path_to_save_labels,
        current_path_to_save_signals,
    ) in zip(path_to_read, save_path_cluster_labels, save_path_cluster_signals):
        printlog("Clustering " + repr(current_path_to_read))
        ### LOAD BRAIN ###

        # brain_path = os.path.join(func_path, 'functional_channel_2_moco_zscore_highpass.h5')
        t0 = time.time()
        # with h5py.File(brain_path, 'r+') as h5_file:
        #with h5py.File(current_path_to_read, "r+") as file:
        #    # Load everything into memory, cast as float 32
        #    brain = file["data"][:].astype(np.float32)
        #    # Convert nan to num, ideally in place to avoid duplication of data
        #    brain = np.nan_to_num(brain, copy=False)
        #    # brain = np.nan_to_num(h5_file.get("data")[:].astype('float32'))

        # Everything is only nifty in this pipeline! Define proxy
        brain_data_proxy = nib.load(current_path_to_read)
        # Load everything into memory, cast DTYPE
        brain = np.asarray(brain_data_proxy.dataobj, dtype=DTYPE)
        # Convert nan to num, ideally in place to avoid duplication of data
        brain = np.nan_to_num(brain, copy=False)

        printlog("brain shape: {}".format(brain.shape))
        printlog("load duration: {} sec".format(time.time() - t0))

        ### MAKE CLUSTER DIRECTORY ###

        current_path_to_save_labels.parent.mkdir(exist_ok=True, parents=True)

        ### FIT CLUSTERS ###

        printlog("fitting clusters")
        t0 = time.time()
        # connectivity = sklearn.feature_extraction.image.grid_to_graph(256,128)
        connectivity = sklearn.feature_extraction.image.grid_to_graph(
            brain.shape[0], brain.shape[1]
        )
        cluster_labels = []
        # for z in range(49):
        for z in range(brain.shape[2]):
            # neural_activity = brain[:,:,z,:].reshape(-1, 3384) # I *THINK* 3384 is frames at 30 minutes
            neural_activity = brain[:, :, z, :].reshape(-1, brain.shape[3])
            cluster_model = sklearn.cluster.AgglomerativeClustering(
                n_clusters=n_clusters,
                memory=str(current_path_to_save_labels.parent),
                linkage="ward",
                connectivity=connectivity,
            )
            cluster_model.fit(neural_activity)
            cluster_labels.append(cluster_model.labels_)

        print(
            "str(current_path_to_save_labels.parent) "
            + repr(str(current_path_to_save_labels.parent))
        )
        cluster_labels = np.asarray(cluster_labels)
        # save_file = os.path.join(cluster_dir, 'cluster_labels.npy')
        np.save(current_path_to_save_labels, cluster_labels)
        printlog("cluster fit duration: {} sec".format(time.time() - t0))

        '''if CLUSTER_3D:
            cluster_model_3D = sklearn.cluster.AgglomerativeClustering(
                n_clusters=n_clusters,  # ????
                memory=str(current_path_to_save_labels.parent),
                linkage="ward",
                connectivity=connectivity
            )
            cluster_model_3D.fit(brain.reshape(-1, brain.shape[3]))
            cluster_labels_3D = cluster_model_3D.labels_

            np.save(pathlib.Path(current_path_to_save_labels.parent, '3Dlabels.npy'), cluster_labels_3D)'''

        ### GET CLUSTER AVERAGE SIGNAL ###

        printlog("getting cluster averages")
        t0 = time.time()
        all_signals = []
        # for z in range(49): # < CHANGE, this is just dim=3
        for z in range(brain.shape[2]):
            # neural_activity = brain[:,:,z,:].reshape(-1, 3384) # I *THINK* 3384 is frames at 30 minutes
            neural_activity = brain[:, :, z, :].reshape(-1, brain.shape[3])
            signals = []
            for cluster_num in range(n_clusters):
                cluster_indicies = np.where(cluster_labels[z, :] == cluster_num)[0]
                mean_signal = np.mean(neural_activity[cluster_indicies, :], axis=0)
                signals.append(mean_signal)
            signals = np.asarray(signals)
            all_signals.append(signals)
        all_signals = np.asarray(all_signals)
        # save_file = os.path.join(cluster_dir, 'cluster_signals.npy')
        np.save(current_path_to_save_signals, all_signals)
        printlog("cluster creation took: {} sec".format(time.time() - t0))

        '''if CLUSTER_3D:
            neural_activity_3D = brain.reshape(-1, brain.shape[3])
            signals_3D = []
            for cluster_num in range(n_clusters):
                cluster_indeces = np.where(cluster_labels_3D == cluster_num)[0]
                mean_signal = np.mean(neural_activity[cluster_indeces, :], axis=0)
                signals_3D.append(mean_signal)
            signals_3D = np.asarray(signals_3D)
            np.save(pathlib.Path(current_path_to_save_signals.parent, '3D_signals.npy'), signals_3D)'''

def apply_transforms(
    fly_directory,
    path_to_read_fixed,
    path_to_read_moving,
    path_to_save,
    resolution_of_fixed,
    resolution_of_moving,
    final_2um_iso,
):
    # TODO!!!!!<<<<<<<<<<<<< NOT FINISHED YET. -> Check with Bella/Andrew - this might be Bifrost territory
    """

    :param fly_directory:
    :return:
    """

    ###
    # Logging
    ###
    logfile = utils.create_logfile(fly_directory, function_name="func2anat")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "func2anat")

    # logfile = args['logfile']
    # save_directory = args['save_directory']

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    path_to_read_fixed = utils.convert_list_of_string_to_posix_path(path_to_read_fixed)
    path_to_read_moving = utils.convert_list_of_string_to_posix_path(
        path_to_read_moving
    )
    path_to_save = utils.convert_list_of_string_to_posix_path(path_to_save)

    # There can only be one fixed path!
    path_to_read_fixed = path_to_read_fixed[0]
    # fixed_path = #args['fixed_path']
    fixed_fly = path_to_read_fixed.name  # args['fixed_fly']
    # fixed_resolution = #args['fixed_resolution']

    # final_2um_iso = args['final_2um_iso']
    # nonmyr_to_myr_transform = args['nonmyr_to_myr_transform']

    ###################
    ### Load Brains ###
    ###################

    # fixed = np.asarray(nib.load(fixed_path).get_data().squeeze(), dtype='float32')
    fixed_brain_proxy = nib.load(path_to_read_fixed)
    fixed_brain = np.asarray(
        fixed_brain_proxy.dataobj, dtype=DTYPE
    )  # I'm not using squeeze here! Might introduce
    # a bug so important to keep if statement below!
    utils.check_for_nan_and_inf_func(fixed_brain)
    #
    fixed_brain = ants.from_numpy(fixed_brain)
    fixed_brain.set_spacing(resolution_of_fixed)
    if final_2um_iso:
        fixed = ants.resample_image(fixed_brain, (2, 2, 2), use_voxels=False)

    moving_resolution = resolution_of_moving  # args['moving_resolution']
    for current_path_to_read_moving, current_path_to_save in zip(
        path_to_read_moving, path_to_save
    ):
        # moving_path = args['moving_path']
        moving_fly = current_path_to_read_moving.name  # args['moving_fly']

        # moving = np.asarray(nib.load(moving_path).get_data().squeeze(), dtype='float32')
        moving_brain_proxy = nib.load(current_path_to_read_moving)
        moving_brain = np.asarray(moving_brain_proxy.dataobj, dtype=DTYPE)

        moving_brain = ants.from_numpy(moving_brain)
        moving_brain.set_spacing(moving_brain)

        ###########################
        ### Organize Transforms ###
        ###########################
        affine_file = os.listdir(
            os.path.join(save_directory, "func-to-anat_fwdtransforms_2umiso")
        )[0]
        affine_path = os.path.join(
            save_directory, "func-to-anat_fwdtransforms_2umiso", affine_file
        )

        warp_dir = "anat-to-non_myr_mean_fwdtransforms_2umiso"
        # warp_dir = 'anat-to-meanbrain_fwdtransforms_2umiso'
        syn_files = os.listdir(os.path.join(save_directory, warp_dir))
        syn_linear_path = os.path.join(
            save_directory, warp_dir, [x for x in syn_files if ".mat" in x][0]
        )
        syn_nonlinear_path = os.path.join(
            save_directory, warp_dir, [x for x in syn_files if ".nii.gz" in x][0]
        )

        transforms = [affine_path, syn_linear_path, syn_nonlinear_path]

        ########################
        ### Apply Transforms ###
        ########################
        moco = ants.apply_transforms(fixed, moving, transforms)

        ############
        ### Save ###
        ############
        save_file = os.path.join(
            save_directory, moving_fly + "-applied-" + fixed_fly + ".nii"
        )
        nib.Nifti1Image(moco.numpy(), np.eye(4)).to_filename(save_file)


def align_anat(
    fly_directory,
    rule_name,
    fixed_fly,
    moving_fly,
    path_to_read_fixed,
    path_to_read_moving,
    path_to_save,
    resolution_of_fixed,
    resolution_of_moving,
    iso_2um_resample,
    type_of_transform,
    grad_step,
    flow_sigma,
    total_sigma,
    syn_sampling
):
    """
    This function aligns the moving brain to the fixed brain.
    In the preprocessing pipeline this function is called twice:
        1) The low resolution anatomical scan (e.g. func0/channel_1.nii) is registered with the
           high resolution anatomical scan (e.g. anat0/channel_1.nii).
        2) In the other call, the high resolution anatomical scan (e.g. anat0/channel_1.nii) is
           registered with an anatomy template, which can be found in the '/brain_atlases' folder.


    Diff to Bella's function - instead of calling output something like 'func-to-anat.nii' it defines
    the channel and yields something like 'channel_1_func-to-channel_1_anat.nii'


    :param fly_directory:a pathlib.Path object to a 'fly' (or 'larva') folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param rule_name: string, name of the log file
    :param fixed_fly: string, used for logging
    :param moving_fly: string, used for logging
    :param path_to_read_fixed: Full path as a list of pathlib.Path objects to the nii to be read as the fixed image
    :param path_to_read_moving: Full path as a list of pathlib.Path objects to the nii to be read as the moving image
    :param path_to_save: Full path as a list of pathlib.Path objects to the nii to be saved
    :param resolution_of_fixed: Resolution as a tuple of the fixed image. If None, will extract from metadata (best!)
    :param resolution_of_moving: Resolution of the moving image. If None, will extract from metadata (best!)
    :param iso_2um_resample: either 'fixed' or 'moving'. indicates which of the brains needs to be resamples at 2um.
    #:param iso_2um_fixed:
    #:param iso_2um_moving:
    :param type_of_transform: See https://antspy.readthedocs.io/en/latest/registration.html
    :param grad_step: See https://antspy.readthedocs.io/en/latest/registration.html
    :param flow_sigma: See https://antspy.readthedocs.io/en/latest/registration.html
    :param total_sigma: See https://antspy.readthedocs.io/en/latest/registration.html
    :param syn_sampling: See https://antspy.readthedocs.io/en/latest/registration.html
    :return:
    """

    ####
    # Logging
    ####
    logfile = utils.create_logfile(fly_directory, function_name=rule_name)
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, rule_name)

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    path_to_read_fixed = utils.convert_list_of_string_to_posix_path(path_to_read_fixed)
    path_to_read_moving = utils.convert_list_of_string_to_posix_path(path_to_read_moving)
    path_to_save = utils.convert_list_of_string_to_posix_path(path_to_save)

    # There can only be one fixed brain, of course
    path_to_read_fixed = path_to_read_fixed[0]

    ####
    # DEFINE RESOLUTION
    ####
    # Should not be defined in snakefile but read automatically by xml file!
    if resolution_of_fixed is None:
        metadata_path = pathlib.Path(path_to_read_fixed.parents[1], 'imaging/recording_metadata.xml')
        resolution_of_fixed = align_utils.extract_resolution(metadata_path)

    flip_X = False # Todo - what does this do? Was set to false in brainsss
    flip_Z = False # Todo - what does this do? Was set to false in brainsss
    save_warp_params = True  # copy-paste from brainsss. args['save_warp_params']

    fixed_fly = ('channel_'
                 + path_to_read_fixed.name.split('channel_')[-1].split('_')[0]
                 + '_' + fixed_fly)

    ###################
    ### Load Brains ###
    ###################

    ### Fixed
    # Doesn't load to memory
    fixed_brain_proxy = nib.load(path_to_read_fixed)
    # Load to memory
    fixed_brain = np.squeeze(np.asarray(fixed_brain_proxy.dataobj, dtype=DTYPE))
    utils.check_for_nan_and_inf_func(fixed_brain)

    fixed_brain = ants.from_numpy(fixed_brain)
    fixed_brain.set_spacing(resolution_of_fixed)
    # This is only called during func2anat call
    #if iso_2um_fixed: # Only resample IF iso_2um_fixed is set (
    if 'fixed' in iso_2um_resample:
        fixed_brain = ants.resample_image(fixed_brain, (2, 2, 2), use_voxels=False)

    # It's possible to have to channels for the 'moving' brain. Do this in a loop
    for current_path_to_read_moving, current_path_to_save in zip(
        path_to_read_moving, path_to_save
    ):

        ### Moving
        # Find resolution if None
        if resolution_of_moving is None:
            metadata_path = pathlib.Path(current_path_to_read_moving.parents[1], 'imaging/recording_metadata.xml')
            resolution_of_moving = align_utils.extract_resolution(metadata_path)
        # Load brain data
        moving_brain_proxy = nib.load(current_path_to_read_moving)
        moving_brain = np.squeeze(np.asarray(moving_brain_proxy.dataobj, dtype=DTYPE))
        # Not sure if squeeze is necessary - Bella used the nib.squeeze that should have same (?)
        # functionality as np.squeeze (remove last dimension if length=1)
        utils.check_for_nan_and_inf_func(fixed_brain)
        if len(moving_brain.shape) > 3:
            printlog(
                "WARNING: Here we should only have 3 dimensions not "
                + repr(fixed_brain.shape)
            )
        if flip_X:
            moving_brain = moving_brain[::-1, :, :]
        if flip_Z:
            moving_brain = moving_brain[:, :, ::-1]
        moving_brain = ants.from_numpy(moving_brain)
        moving_brain.set_spacing(resolution_of_moving)
        # This is only applied during anat2atlas rule
        #if iso_2um_moving: # there are also low_res and very_low_res option in brainsss!
        if iso_2um_resample == 'moving':
            moving_brain = ants.resample_image(moving_brain, (2, 2, 2), use_voxels=False)

        # Give a bit more info about the fixed and moving fly by adding channel information!
        moving_fly = 'channel_' + current_path_to_read_moving.name.split('channel_')[-1].split('_')[0] + '_' + moving_fly

        printlog("Starting registration of {} to {}".format(moving_fly, fixed_fly))

        #############
        ### Align ###
        #############
        # Since we only align meanbrains, it's quite fast!
        t0 = time.time()
        moco = ants.registration(
            fixed_brain,
            moving_brain,
            type_of_transform=type_of_transform,
            grad_step=grad_step,
            flow_sigma=flow_sigma,
            total_sigma=total_sigma,
            syn_sampling=syn_sampling,
        )

        printlog(
            "Fixed: {}, {} | Moving: {}, {} | {} | {}".format(
                fixed_fly,
                path_to_read_fixed.name.split("/")[-1],
                moving_fly,
                current_path_to_read_moving.name.split("/")[-1],
                type_of_transform,
                utils.sec_to_hms(time.time() - t0),
            )
        )

        ################################
        ### Save warp params if True ###
        ################################

        if save_warp_params:
            fwdtransformlist = moco["fwdtransforms"]
            print("fwdtransformlist" + repr(fwdtransformlist))
            fwdtransforms_save_folder = pathlib.Path(
                current_path_to_save.parent,
                "{}-to-{}_fwdtransforms".format(moving_fly, fixed_fly),
            )

            #if True in [iso_2um_moving, iso_2um_fixed]:
            # Currently always the case but I changed the variable to iso_2um_resample
            fwdtransforms_save_folder = pathlib.Path(
                fwdtransforms_save_folder.parent, fwdtransforms_save_folder.name + "_2umiso"
            )
            fwdtransforms_save_folder.mkdir(exist_ok=True, parents=True)
            for source_path in fwdtransformlist:
                print('fwdtransformlist' + repr(source_path))
                source_file = pathlib.Path(source_path).name
                target_path = pathlib.Path(fwdtransforms_save_folder, source_file)
                shutil.copyfile(source_path, target_path)

            # Added this saving of inv transforms 2020 Dec 19
            invransformlist = moco["invtransforms"] # <- This is never used!!!!
            invtransforms_save_folder = pathlib.Path(
                current_path_to_save.parent,
                "{}-to-{}_invtransforms".format(moving_fly, fixed_fly),
            )
            #if True in [iso_2um_moving, iso_2um_fixed]:
            # Currently always the case but I changed the variable to iso_2um_resample
            invtransforms_save_folder = pathlib.Path(
                invtransforms_save_folder.parent, invtransforms_save_folder.name + "_2umiso"
            )
            invtransforms_save_folder.mkdir(exist_ok=True, parents=True)
            for source_path in invransformlist:
                source_file = pathlib.Path(source_path).name
                target_path = pathlib.Path(invtransforms_save_folder, source_file)
                shutil.copyfile(source_path, target_path)

        ############
        ### Save ###
        ############

        nib.Nifti1Image(moco["warpedmovout"].numpy(), np.eye(4)).to_filename(
            current_path_to_save
        )



def clean_anatomy(fly_directory, path_to_read, save_path):
    """
    Requires a surprising amount of memory.
    Only on folders called anatomy!
    Todo: Possible to save on RAM by reading brain twice instead of copying arrays!
    Make sure to not accidentaly overwrite the original file, though.
    :param args:
    :return:
    """

    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="clean_anatomy")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "clean_anatomy")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    path_to_read = utils.convert_list_of_string_to_posix_path(path_to_read)
    save_path = utils.convert_list_of_string_to_posix_path(save_path)

    printlog("Will perform clean anatomy on " + repr(path_to_read[0]))

    ##########
    ### Load brain ###
    ##########
    # Since we only have a >>>single anatomy channel<<< no need to loop!
    brain_proxy = nib.load(path_to_read[0])
    brain = np.asarray(brain_proxy.dataobj, dtype=DTYPE)

    ### Blur brain and mask small values ###
    brain_copy = brain.copy()  # Make a copy in memory of the brain data, doubles memory
    # requirement # Not necessary, already cast as float32 ".astype('float32')"

    brain_copy = ndimage.gaussian_filter(brain_copy, sigma=10)  # filter brain copy.
    # Note: it doesn't seem to be necessary to make a copy of brain before calling gaussian
    # filter. gaussian_filter makes a copy of the input array. Could be changed to save
    # memory requirement!

    threshold = skimage.filters.threshold_triangle(
        brain_copy
    )  # this is a threshold detection
    # algorithm (Similar to Otsu). https://pubmed.ncbi.nlm.nih.gov/70454/
    # "The threshold (THR) was selected by normalizing the height and dynamic range of the intensity histogram"

    brain_copy[
        np.where(brain_copy < threshold / 2)
    ] = 0  # Set every value below threshold to zero

    ### Remove blobs outside contiguous brain ###
    labels, label_nb = ndimage.label(
        brain_copy
    )  # This is done on the already thresholded brain
    brain_label = np.bincount(labels.flatten())[1:].argmax() + 1

    # Make another copy of brain
    brain_copy = (
        brain.copy()
    )  # Not necessary, already cast as float32 ".astype('float32')"
    brain_copy[
        np.where(labels != brain_label)
    ] = np.nan  # Set blobs outside brain to nan

    ### Perform quantile normalization ###
    # brain_out = sklearn.preprocessing.quantile_transform(brain_copy.flatten().reshape(-1, 1), n_quantiles=500, random_state=0, copy=True)
    # This method transforms the features to follow a uniform or a normal distribution.
    brain_out = sklearn.preprocessing.quantile_transform(
        brain_copy.ravel().reshape(-1, 1), n_quantiles=500, random_state=0, copy=True
    )
    # change flatten to ravel to avoid making a copy if possible.
    brain_out = brain_out.reshape(brain.shape)
    np.nan_to_num(brain_out, copy=False)  # nan are set to zeros here.

    ### Save brain ###
    # save_file = save_path[0].name[:-4] + '_clean.nii'
    aff = np.eye(4)
    img = nib.Nifti1Image(brain_out, aff)  # is this float32? check
    img.to_filename(save_path[0])

    printlog("Clean anatomy successfully saved in " + repr(save_path[0]))


def stim_triggered_avg_neu(args):
    """
    TODO
    :param args:
    :return:
    """
    logfile = args["logfile"]
    func_path = args["func_path"]
    printlog = getattr(brainsss.Printlog(logfile=logfile), "print_to_log")
    printlog(func_path)

    ###########################
    ### PREP VISUAL STIMULI ###
    ###########################

    vision_path = os.path.join(func_path, "visual")

    ### Load Photodiode ###
    t, ft_triggers, pd1, pd2 = brainsss.load_photodiode(vision_path)
    stimulus_start_times = brainsss.extract_stim_times_from_pd(pd2, t)

    ### Get Metadata ###
    stim_ids, angles = brainsss.get_stimulus_metadata(vision_path)
    printlog(f"Found {len(stim_ids)} presented stimuli.")

    # *100 puts in units of 10ms, which will match fictrac
    starts_angle_0 = [
        int(stimulus_start_times[i] * 100)
        for i in range(len(stimulus_start_times))
        if angles[i] == 0
    ]
    starts_angle_180 = [
        int(stimulus_start_times[i] * 100)
        for i in range(len(stimulus_start_times))
        if angles[i] == 180
    ]
    printlog(
        f"starts_angle_0: {len(starts_angle_0)}. starts_angle_180: {len(starts_angle_180)}"
    )
    list_in_ms0 = {
        "0": [i * 10 for i in starts_angle_0],
        "180": [i * 10 for i in starts_angle_180],
    }

    ##################
    ### PREP BRAIN ###
    ##################

    brain_path = os.path.join(func_path, "functional_channel_2_moco_zscore_highpass.h5")
    timestamps = brainsss.load_timestamps(
        os.path.join(func_path, "imaging"), file="functional.xml"
    )

    with h5py.File(brain_path, "r") as hf:
        dims = np.shape(hf["data"])

    for angle in list_in_ms0.keys():
        stas = []
        for slice_num in range(dims[2]):
            t0 = time.time()
            single_slice = load_slice(brain_path, slice_num)
            ynew = interpolation(slice_num, timestamps, single_slice)
            new_stim_timestamps = make_new_stim_timestamps(list_in_ms0[angle])
            chunk_edges = make_chunk_edges(new_stim_timestamps)
            sta = make_stas(ynew, new_stim_timestamps, chunk_edges)
            stas.append(sta)
            printlog(f"Slice: {slice_num}. Duration: {time() - t0}")
        stas_array = np.asarray(stas)

        ### SAVE STA ###
        savedir = os.path.join(func_path, "STA")
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savefile = os.path.join(savedir, f"sta_{angle}.npy")
        np.save(savefile, stas_array)

        ### SAVE PNG ###
        save_maxproj_img(savefile)


def correlation(
    fly_directory,
    dataset_path,
    save_path,
    fictrac_fps,
    metadata_path,
    fictrac_path
):
    """
    Correlate z-scored brain activity with behavioral activity.

    The original function from brainsss was a triple looped call to scipy>pearsonr which took quite long as
    this example shows
    | SLURM | corr | 20559045 | COMPLETED | 00:28:55 | 4 cores | 21.7GB (69.7%)

    To speed the correlation up, I used only the parts from the scipy pearsonr function we need.
    One difference is that we only work in float32 space. The scipy function would cast everything as float64,
    doubling the memory requirements.
    When I subtract the vectorized result with the looped scipy pearson result I get a max(diff) of 9.8e-8. This
    should not be relevant for us.

    See script 'pearson_correlation.py' - the vectorized version should take 0.03% of the time the
    scipy version does.

    :param fly_directory,
    :param dataset_path,
    :param save_path,
    :param fictrac_fps,
    :param metadata_path,
    :param fictrac_path
    :return:
    """
    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="correlation")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "correlation")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    save_path = utils.convert_list_of_string_to_posix_path(save_path)

    # We will have a list of functional channels here but
    # all from the same experiment, so all will have the same 'recording_metadata.xml' data
    timestamps = utils.load_timestamps(metadata_path)
    # These are timestamps PER FRAME. For example, for a volume with 49 z slices and 602 volumes
    # timestamps.shape > (602, 49).
    # and when we look at e.g. timestamps[0,:] we see times in ms it took to record that particular
    # volume (e.g. [104.6, 113.3 ... 523..27])

    # Extract 'behavior' to be correlated with neural activity from savepath
    behavior = save_path[0].name.split("corr_")[-1].split(".nii")[0]
    # This should yield for example 'dRotLabZneg'

    printlog("grey_only not implemented yet")
    """### this means only calculat correlation during periods of grey stimuli ###
    if grey_only:
        vision_path = os.path.join(load_directory, 'visual')
        stim_ids, angles = brainsss.get_stimulus_metadata(vision_path)
        t, ft_triggers, pd1, pd2 = brainsss.load_photodiode(vision_path)
        stimulus_start_times = brainsss.extract_stim_times_from_pd(pd2, t)
        grey_starts = []
        grey_stops = []
        for i, stim in enumerate(stim_ids):
            if stim == 'ConstantBackground':
                grey_starts.append(stimulus_start_times[i])
                grey_stops.append(stimulus_start_times[i] + 60)
        grey_starts = [i * 1000 for i in grey_starts]  # convert from s to ms
        grey_stops = [i * 1000 for i in grey_stops]  # convert from s to ms
        idx_to_use = []
        for i in range(len(grey_starts)):
            idx_to_use.extend(np.where((grey_starts[i] < timestamps[:, 0]) & (timestamps[:, 0] < grey_stops[i]))[0])
        ### this is now a list of indices where grey stim was presented
    #else:"""

    ####################
    ### Load fictrac ###
    ####################
    fictrac_raw = fictrac_utils.load_fictrac(fictrac_path)
    expt_len = (fictrac_raw.shape[0] / fictrac_fps) * 1000
    # how many datapoints divide by how many times per seconds,in ms

    fictrac_resolution = (1/fictrac_fps) * 1000 # in ms
    ### interpolate fictrac to match the timestamps from the microscope!
    fictrac_interp = fictrac_utils.smooth_and_interp_fictrac(
        fictrac_raw, fictrac_fps, fictrac_resolution, expt_len, behavior, timestamps=timestamps
    )
    # Originally, there was a z parameter which was used as timestamps[:,z] to return the fictrac
    # data for a given z slice. We're not using it in the vectorized verison

    # It's possible to have more than one channel as the functional channel
    # Since we are memory limited, do correlation for both channels consecutively!
    for current_dataset_path, current_save_path in zip(dataset_path, save_path):
        if "nii" in current_dataset_path.name:
            brain_proxy = nib.load(current_dataset_path)
            brain = np.asarray(brain_proxy.dataobj, dtype=DTYPE)
            printlog(
                "Loaded nii file - note that this was only tested with h5 files. nii seems to work, though."
            )
        elif ".h5" in current_dataset_path.name:
            with h5py.File(current_dataset_path, "r") as hf:
                brain = hf["data"][:]  # load everything into memory!

        printlog("Brain loaded")

        ### Correlate ###
        printlog(
            "Performing correlation on {}; behavior: {}".format(
                current_dataset_path.name, behavior
            )
        )
        # Preallocate an array filled with zeros
        corr_brain = np.zeros(
            (brain.shape[0], brain.shape[1], brain.shape[2]), dtype=DTYPE
        )
        # Fill array with NaN to rule out that we interpret a missing value as a real value when it should be NaN
        corr_brain.fill(np.nan)
        # Loop through each slice:
        for z in range(brain.shape[2]):
            # Vectorized correlation - see 'dev/pearson_correlation.py' for development and timing info
            # The formula is:
            # r = (sum(x-m_x)*(y-m_y) / sqrt(sum((x-m_x)^2)*sum((y-m_y)^2)
            brain_mean = brain[:, :, z, :].mean(axis=-1)  # , dtype=np.float64)

            # When I plot the data plt.plot(fictrac_interp) and plt.plot(fictrac_interp.astype(np.float32) I
            # can't see a difference. Should be ok to do this as float.
            # This is advantagous because we have to do the dot product with the brain. np.dot will
            # default to higher precision leading to making a copy of brain data which costs a lot of memory
            # Unfortunately fictrac_interp comes in [time, z] as opposed to brain that comes in [x,y,z,t]
            fictrac_mean = fictrac_interp[:, z].mean(dtype=DTYPE)

            # >> Typical values for z scored brain seem to be between -25 and + 25.
            # It shouldn't be necessary to cast as float64. This then allows us
            # to do in-place operation!
            brain_mean_m = brain[:, :, z, :] - brain_mean[:, :, None]
            # fictrac data is small, so working with float64 shouldn't cost much memory!
            # Correction - if we cast as float64 and do dot product with brain, we'll copy the
            # brain to float64 array, balloning the memory requirement
            fictrac_mean_m = fictrac_interp[:, z].astype(DTYPE) - fictrac_mean

            # Note from scipy pearson docs: This can overflow if brain_mean_m is, for example [-5e210, 5e210, 3e200, -3e200]
            # I doubt we'll ever get close to numbers like this.
            normbrain = np.linalg.norm(
                brain_mean_m, axis=-1
            )  # Make a copy, but since there's no time dimension it's quite small
            normfictrac = np.linalg.norm(fictrac_mean_m)
            try:
                # Calculate correlation
                corr_brain[:, :, z] = np.dot(
                    brain_mean_m / normbrain[:, :, None], fictrac_mean_m / normfictrac
                )
            except ValueError:
                print('Likely aborted scan.')
                print('Attempting to run correlation')
                # Remove the last timepoint
                fictrac_mean_m = fictrac_mean_m[0:fictrac_mean_m.shape[0]-1]
                # Calculate correlation

                corr_brain[:, :, z] = np.dot(
                    brain_mean_m / normbrain[:, :, None], fictrac_mean_m / normfictrac
                )
        ### SAVE ###
        current_save_path.parent.mkdir(exist_ok=True, parents=True)
        '''# if 'warp' in full_load_path:
        if "warp" in current_dataset_path.parts:
            warp_str = "_warp"
        else:
            warp_str = ""
        printlog("grey_only not implemented yet")'''
        '''if grey_only:
            grey_str = '_grey'
        else:
            grey_str = '''''
        '''if "zscore" not in current_dataset_path.parts:
            no_zscore_highpass_str = "_mocoonly"
        else:
            no_zscore_highpass_str = ""'''

        # Prepare Nifti file for saving
        aff = np.eye(4)
        object_to_save = nib.Nifti1Image(corr_brain, aff)
        # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
        object_to_save.to_filename(current_save_path)

        # Save easy-to-view png image of correlation
        printlog("Saved {}".format(current_save_path))
        corr_utils.save_maxproj_img(image_to_max_project=corr_brain, path=current_save_path)
        printlog("Saved png plot")

        ###
        # LOG SUCCESS
        ###
        printlog(
            "Successfully finished calculating correlation on {}; behavior: {}".format(
                current_dataset_path.name, behavior
            )
        )
def temporal_high_pass_filter(
        fly_directory,
        dataset_path,
        temporal_high_pass_filtered_path
):
    """
    Filters z-scored brain with scipy.ndimage.gaussian_filter1d in time dimension.

    Todo: Currently sigma is set to 200. It should be dependent on the recording frequency (per volume)!

    Undertanding what the filter does:
    import scipy.ndimage
    box = np.zeros((10,10,10)) # pretend this is x,y and t
    box[0:5,0:5,:] = 10
    box[5:10,5:10,:] = 10
    smooth = scipy.ndimage.gaussian_filter1d(box, sigma=3, axis=-1)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.imshow(box[0,:,:], vmin=0, vmax=10)
    ax2.imshow(smooth[0,:,:], vmin=0, vmax=10) # plot e.g. [:,0,:] and [:,:,0]
    # There is no smoothing across x or y indicating that gaussian_filter1d
    # does what it says on the tin: It only filters across the last dimension,
    # or only as a temporal filter as Bella called it.
    box = np.zeros((10,10,10),dtype=np.float32) # pretend this is x,y and t
    box[:,:,0:10] = np.arange(0,10)
    smooth = scipy.ndimage.gaussian_filter1d(box, sigma=3, axis=-1)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.imshow(box[0,:,:], vmin=0, vmax=10)
    ax2.imshow(smooth[0,:,:], vmin=0, vmax=10)

    # Play with test data
    data_proxy = nib.load('/Users/dtadres/Documents/func1/imaging/channel_1.nii')
    data = np.asarray(data_proxy.dataobj, dtype=np.float32)
    smooth = scipy.ndimage.gaussian_filter1d(data,sigma=200,axis=-1,truncate=1)
    :param args:
    :return:
    """

    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(
        fly_directory, function_name="temporal_high_pass_filter"
    )
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "temporal_high_pass_filter")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    temporal_high_pass_filtered_path = utils.convert_list_of_string_to_posix_path(
        temporal_high_pass_filtered_path
    )

    printlog("Beginning high pass")

    # dataset_path might be a list of 2 channels (or a list with one channel only)
    for current_dataset_path, current_temporal_high_pass_filtered_path in zip(
        dataset_path, temporal_high_pass_filtered_path
    ):
        printlog("Working on " + repr(current_dataset_path.name))

        # Dynamically define sigma for filtering! This was recently introduced to brainsss
        timestamps = utils.load_timestamps(pathlib.Path(current_dataset_path.parent, 'imaging', 'recording_metadata.xml'))
        hz = (1 / np.diff(timestamps[:, 0])[0]) * 1000 # This is volumes per second!
        sigma = int(hz / 0.01)  # gets a sigma of ~100 second

        if '.nii' in current_dataset_path.name:
            # this doesn't actually LOAD the data - it is just a proxy
            current_dataset_proxy = nib.load(current_dataset_path)
            # Now load into RAM
            data = np.asarray(current_dataset_proxy.dataobj, dtype=DTYPE)
            printlog("Data shape is {}".format(data.shape))

            # Calculate mean
            data_mean = np.mean(data, axis=-1)
            # Using filter to smoothen data. This gets rid of high frequency noise.
            smoothed_data = ndimage.gaussian_filter1d(
                data, sigma=sigma, axis=-1, truncate=1
            )  # This for sure makes a copy of
            # the array, doubling memory requirements

            # To save memory, do in-place operations where possible
            # data_high_pass = data - smoothed_data + data_mean[:,:,:,None]
            data -= smoothed_data
            # to save memory: data-= ndimage.gaussian_filter1d(...output=data) to do everything in-place?
            data += data_mean[:, :, :, None]

            ##########################
            ### SAVE DATA AS NIFTY ###
            ##########################
            aff = np.eye(4)
            temp_highpass_nifty = nib.Nifti1Image(data, aff)
            temp_highpass_nifty.to_filename(current_temporal_high_pass_filtered_path)
            printlog('Successfully saved ' + current_temporal_high_pass_filtered_path.as_posix())

        else:
            printlog('Function currently only works with nii as output')
            printlog('this will probably break')

    printlog("high pass done")


def zscore(fly_directory, dataset_path, zscore_path):
    """
    z-scoring of a brain.
    z-score = (x-mu)/sigma with:
        x=observed value
        mu=mean of sample
        sigma=standard deviation of the sample.

    Z-scoring helps normalize our recording and make is comparable across samples where
    different animals will have different baseline fluorescence, different amounts of response
    etc.

    We only z-score the functional channels (defined in snakefile)

    Expected memory requirement for mostly in place operations
    2x Filesize + 2~10 MB for the meanbrain and std (and more for larger images)

    Reason:
    https://ipython-books.github.io/45-understanding-the-internals-of-numpy-to-avoid-unnecessary-array-copying/
    def aid(x):
         # This function returns the memory
         # block address of an array.
         return x.__array_interface__['data'][0]
    fakebrain = np.zeros((128,256,49,1000))
    meanbrain = np.nanmean(fakebrain, axis=3)
    stdbrain = np.std(fakebrain, axis=3)
    zscore = (fakebrain-meanbrain[:,:,:,np.newaxis])/stdbrain[:,:,:,np.newaxis]

    aid(fake_brain), aid(zscore)
    >(11379671040, 30647255040) # Different memory location!

    # in place operation possible?
    np.subtract.at(fakebrain, [], meanbrain[:,:,:,np.newaxis)
    Or just?
    fakebrain = fakebrain-meanbrain[:,:,:,np.newaxis])/stdbrain[:,:,:,np.newaxis]
    -> Nope that leads to different memory locations when using 'aid' func.

    Just use the normal in place operators:
    data = np.zeros((128,256,49,1000))
    meanbrain = np.nanmean(fakebrain, axis=3)
    stdbrain = np.std(fakebrain, axis=3)
    aid(data)
    > 10737418240
    # First subtract meanbrain from everythin
    data-=meanbrain[:,:,:,np.newaxis]
    aid(data)
    > 10737418240
    # Then divide everything by std to get zscore
    data/=stdbrain[:,:,:,np.newaxis]
    aid(data)
    >10737418240

    :param fly_directory: a pathlib.Path object to a 'fly' (or 'larva') folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param dataset_path: Full path as a list of pathlib.Path objects to the nii to be read
    :param zscore_path: Full path as a list of pathlib.Path objects to the nii to be saved
    """
    # To reproduce Bella's script
    #RUN_LOOPED = False
    #if RUN_LOOPED:
    #    stepsize=100

    ##############
    ### ZSCORE ###
    ##############
    logfile = utils.create_logfile(fly_directory, function_name="zscore")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "zscore")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    zscore_path = utils.convert_list_of_string_to_posix_path(zscore_path)

    printlog("Beginning ZSCORE")

    # we might get a second functional channel in the future!
    for current_dataset_path, current_zscore_path in zip(dataset_path, zscore_path):
        if 'nii' in current_dataset_path.name:
            dataset_proxy = nib.load(current_dataset_path)
            data = np.asarray(dataset_proxy.dataobj, dtype=DTYPE)

            printlog("Data shape is {}".format(data.shape))

            # Expect a 4D array, xyz and the fourth dimension is time!
            mean_brain = np.nanmean(data, axis=3, dtype=DTYPE_CACLULATIONS)
            printlog('Calculated mean of data')
            std_brain = np.std(data, axis=3, dtype=DTYPE_CACLULATIONS)  # With float64 need much more memory
            printlog('Calculated standard deviation')

            ### Calculate zscore and save ###
            # Calculate z-score
            # z_scored = (data - mean_brain[:,:,:,np.newaxis])/std_brain[:,:,:,np.newaxis]
            # The above works, is easy to read but makes a copy in memory. Since brain data is
            # huge (easily 20Gb) we'll avoid making a copy by doing in place operations to save
            # memory! See docstring for more information

            # data will be data-meanbrain after this operation
            data -= mean_brain[:, :, :, np.newaxis]
            # Then it will be divided by std which leads to zscore
            data /= std_brain[:, :, :, np.newaxis]
            # convert nan to zeros - from brainsss
            data = np.nan_to_num(data)
            printlog('Calculated z-score')

            # Prepare nifti file
            aff = np.eye(4)
            zscore_nifty = nib.Nifti1Image(data, aff)
            zscore_nifty.to_filename(current_zscore_path)

            printlog('Saved z-score image as ' + current_zscore_path.as_posix())
        else:
            printlog('Function currently only works with nii as output')
            printlog('this will probably break')

    printlog("z-scoring done")

def make_mean_brain(
    fly_directory, meanbrain_n_frames, path_to_read, path_to_save, rule_name
):
    """
    Function to calculate meanbrain.
    This is based on Bella's meanbrain script.
    :param fly_directory: pathlib.Path object to a 'fly' (or 'larva') folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param meanbrain_n_frames: First n frames to average over when computing mean/fixed brain | Default None (average over all frames).
    :param path_to_read: Full path as a list of pathlib.Path objects to the nii to be read
    :param path_to_save: Full path as a list of pathlib.Path objects to the nii to be saved
    :param rule_name: a string used to save the log file
    """

    ####
    # LOGGING
    ####
    logfile = utils.create_logfile(fly_directory, function_name=rule_name)
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, rule_name)

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    path_to_read = utils.convert_list_of_string_to_posix_path(path_to_read)
    path_to_save = utils.convert_list_of_string_to_posix_path(path_to_save)

    for current_path_to_read, current_path_to_save in zip(path_to_read, path_to_save):
        brain_data = None  # make sure the array is empty before starting another iteration!
        ###
        # Read imaging file
        ###
        printlog("Currently looking at: " + repr(current_path_to_read.name))
        if current_path_to_read.suffix == ".nii":
            # Doesn't load anything, just points to a given location
            brain_proxy = nib.load(current_path_to_read)
            # Load data, it's np.uint16 at this point, no point changing it.
            brain_data = np.asarray(brain_proxy.dataobj, dtype=np.uint16)
        elif current_path_to_read.suffix == ".h5":
            # Original: Not great because moco brains are saved as float32
            #with h5py.File(current_path_to_read, "r") as hf:
            #    brain_data = np.asarray(hf["data"][:], dtype="uint16")
            with h5py.File(current_path_to_read, "r") as hf:
                brain_data = np.asarray(hf["data"][:], dtype=DTYPE)
        else:
            printlog("Current file has suffix " + current_path_to_read.suffix)
            printlog("Can currently only handle .nii and .h5 files!")
        # brain_data = np.asarray(nib.load(path_to_read).get_fdata(), dtype='uint16')
        # get_fdata() loads data into memory and sometimes doesn't release it.

        ###
        # CREATE MEANBRAIN
        ###
        if meanbrain_n_frames is not None:
            # average over first meanbrain_n_frames frames
            meanbrain = np.mean(brain_data[..., : int(meanbrain_n_frames)], axis=-1, dtype=DTYPE_CACLULATIONS)
        else:  # average over all frames
            meanbrain = np.mean(brain_data, axis=-1, dtype=DTYPE_CACLULATIONS)

        printlog("Datatype of meanbrain: " + repr(meanbrain.dtype))
        ###
        # SAVE MEANBRAIN
        ###
        aff = np.eye(4)
        meanbrain_nifty = nib.Nifti1Image(
            meanbrain, aff
        )
        meanbrain_nifty.to_filename(current_path_to_save)

        ###
        # LOG SUCCESS
        ###
        fly_print = pathlib.Path(fly_directory).name
        func_print = str(current_path_to_read).split("/imaging")[0].split("/")[-1]
        # func_print = current_path_to_read.name.split('/')[-2]
        printlog(
            f"meanbrn | COMPLETED | {fly_print} | {func_print} | {brain_data.shape} ===> {meanbrain.shape}"
        )


def bleaching_qc(
    fly_directory,
    path_to_read,
    path_to_save,
):
    """
    Perform bleaching quality control
    This is based on Bella's 'bleaching_qc.py' script
    Input are all nii files per folder (e.g. channel_1.nii and channel_2.nii) and output is single 'bleaching.png' file.
    Bleaching is defined as overall decrease of fluorescence in a given nii file.

    :param fly_directory: a pathlib.Path object to a 'fly' (or 'larva') folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param path_to_read: list of paths to images to read, can be more than one
    :param path_to_save: list of paths to the 'bleaching.png' file
    :param functional_channel_list: list with channels marked as functional channels by experimenter
    :param anatomical_channel: the channel marked as the anatomy channel by the experimenter
    """
    ####
    # LOGGING
    ####
    logfile = utils.create_logfile(fly_directory, function_name="bleaching_qc")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "bleaching_qc")

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    path_to_read = utils.convert_list_of_string_to_posix_path(path_to_read)
    path_to_save = utils.convert_list_of_string_to_posix_path(path_to_save)
    # There is only one path to save! It comes as a list
    path_to_save = path_to_save[0]

    #####
    # CALCULATE MEAN FLUORESCENCE PER TIMEPOINT
    #####
    data_mean = {}
    # For each path in the list
    for current_path_to_read in path_to_read:
        printlog(f"Currently reading: {current_path_to_read.name:.>{WIDTH - 20}}")
        # Doesn't load anything to memory, just a pointer
        brain_proxy = nib.load(current_path_to_read)
        # Load data into memory, brain at this point is half of uint14, no point doing float
        brain = np.asarray(brain_proxy.dataobj, dtype=np.uint16)
        utils.check_for_nan_and_inf_func(brain)
        # calculate mean over time
        data_mean[current_path_to_read.name] = np.mean(brain, axis=(0, 1, 2))

    ##############################
    ### OUTPUT BLEACHING CURVE ###
    ##############################
    # plotting params
    plt.rcParams.update({"font.size": 24})
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    signal_loss = {}

    for filename in data_mean:
        xs = np.arange(len(data_mean[filename]))
        color = "k"
        # I slightly changed the colornames to make it obvious that data was analyzed
        # with a pipeline different from the normal one
        if "channel_1" in filename:
            color = "tomato"  # on our scope this is red
        elif "channel_2" in filename:
            color = "lime"  # on our scope this is green
        elif "channel_3" in filename:
            color = "darkred"  # on our scope this should be an IR channel
        ax.plot(data_mean[filename], color=color, label=filename)
        try:
            # Fit polynomial to mean fluorescence.
            linear_fit = np.polyfit(xs, data_mean[filename], 1)
            # and plot it
            ax.plot(np.poly1d(linear_fit)(xs), color="k", linewidth=3, linestyle="--")
            # take the linear fit to calculate how much signal is lost and report it as the title
            signal_loss[filename] = (
                linear_fit[0] * len(data_mean[filename]) / linear_fit[1] * -100
            )
        except:
            print("unable to perform fit because of error: ")
            print(traceback.format_exc())
            print(
                "\n Checking for nan and inf as possible cause - if no output, no nan and inf found"
            )
            print(utils.check_for_nan_and_inf_func(data_mean[filename]))

    ax.set_xlabel("Frame Num")
    ax.set_ylabel("Avg signal")
    try:
        loss_string = ""
        for filename in data_mean:
            loss_string = (
                loss_string
                + filename
                + " lost"
                + f"{int(signal_loss[filename])}"
                + "%\n"
            )
    except:  # This happens when unable to peform fit (see try..except above).
        pass
    ax.set_title(loss_string, ha="center", va="bottom")

    ###
    # SAVE PLOT
    ###
    save_file = pathlib.Path(path_to_save)
    fig.savefig(save_file, dpi=300, bbox_inches="tight")

    ###
    # LOG SUCCESS
    ###
    printlog(f"Prepared plot and saved as: {str(save_file):.>{WIDTH - 20}}")


def fictrac_qc(fly_directory, fictrac_file_path, fictrac_fps):
    """
    Perform fictrac quality control.
    This is based on Bella's fictrac_qc.py  script.
    :param fly_directory: a pathlib.Path object to a 'fly' (or 'larva') folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param fictrac_file_paths: a list of paths as pathlib.Path objects
    :param fictrac_fps: frames per second of the videocamera recording the fictrac data, an integer
    """
    ####
    # LOGGING
    ####
    logfile = utils.create_logfile(fly_directory, function_name="fictrac_qc")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "fictrac_qc")

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    fictrac_file_path = utils.convert_list_of_string_to_posix_path(fictrac_file_path)
    # Organize path - there is only one, but it comes as a list
    fictrac_file_path = fictrac_file_path[0]

    ####
    # QUALITY CONTROL
    ####
    #for current_file in fictrac_file_paths:
    printlog("Currently looking at: " + repr(fictrac_file_path))
    fictrac_raw = fictrac_utils.load_fictrac(fictrac_file_path)
    # This should yield something like 'fly_001/func0/fictrac
    full_id = ", ".join(fictrac_file_path.parts[-4:-2])

    resolution = 10  # desired resolution in ms # Comes from Bella!
    expt_len = fictrac_raw.shape[0] / fictrac_fps * 1000
    behaviors = ["dRotLabY", "dRotLabZ"]
    fictrac = {}
    for behavior in behaviors:
        if behavior == "dRotLabY":
            short = "Y"
        elif behavior == "dRotLabZ":
            short = "Z"
        fictrac[short] = fictrac_utils.smooth_and_interp_fictrac(
            fictrac_raw, fictrac_fps, resolution, expt_len, behavior
        )
    time_for_plotting = np.arange(0, expt_len, resolution) # comes in ms

    # Call these helper functions for plotting
    fictrac_utils.make_2d_hist(
        fictrac, fictrac_file_path, full_id,  fixed_crop=True
    )
    fictrac_utils.make_2d_hist(
        fictrac, fictrac_file_path, full_id, fixed_crop=False
    )
    fictrac_utils.make_velocity_trace(
        fictrac, fictrac_file_path, full_id, time_for_plotting,
    )

    ###
    # LOG SUCCESS
    ###
    printlog(f"Prepared fictrac QC plot and saved in: {str(fictrac_file_path.parent):.>{WIDTH - 20}}")

def copy_to_scratch(fly_directory, paths_on_oak, paths_on_scratch):
    """
    CURRENTLY NOT USED
    For faster reading and writing, it might be worth putting the nii
    files (and other large files) on $SCRATCH and only save the result
    on oak:
    https://www.sherlock.stanford.edu/docs/storage/filesystems/#scratch
        Each compute node has a low latency, high-bandwidth Infiniband
        link to $SCRATCH. The aggregate bandwidth of the filesystem is
        about 75GB/s. So any job with high data performance requirements
         will take advantage from using $SCRATCH for I/O.
    :return:
    """
    # printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    logfile = utils.create_logfile(fly_directory, function_name="copy_to_scratch")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "copy_to_scratch")

    for current_file_src, current_file_dst in zip(paths_on_oak, paths_on_scratch):
        # make folder if not exist
        pathlib.Path(current_file_dst).parent.mkdir(exist_ok=True, parents=True)
        # copy file
        shutil.copy(current_file_src, current_file_dst)
        printlog("Copied: " + repr(current_file_dst))