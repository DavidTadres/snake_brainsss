import json
import numpy as np
from xml.etree import ElementTree as ET
from lxml import etree, objectify
from openpyxl import load_workbook
import pathlib
import sys
import time
import traceback
import natsort
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
scripts_path = pathlib.Path(
    __file__
).parent.resolve()
sys.path.insert(0, pathlib.Path(scripts_path, "workflow"))
# This just imports '*.py' files from the folder 'brainsss'.
from brainsss import moco_utils
from brainsss import utils
from brainsss import fictrac_utils
from brainsss import corr_utils


####################
# GLOBAL VARIABLES #
####################
WIDTH = 120  # This is used in all logging files


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
    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="make_supervoxels")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, WIDTH, "make_supervoxels")

    # func_path = args['func_path']
    # logfile = args['logfile']
    # printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    print("path_to_read " + repr(path_to_read))
    path_to_read = utils.convert_list_of_string_to_posix_path(path_to_read)
    print("path_to_read " + repr(path_to_read))
    save_path_cluster_labels = utils.convert_list_of_string_to_posix_path(
        save_path_cluster_labels
    )
    print("save_path_cluster_labels " + repr(save_path_cluster_labels))
    save_path_cluster_signals = utils.convert_list_of_string_to_posix_path(
        save_path_cluster_signals
    )
    print("save_path_cluster_signals " + repr(save_path_cluster_signals))

    # Can have more than one functional channel!
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
        with h5py.File(current_path_to_read, "r+") as file:
            # Load everything into memory, cast as float 32
            brain = file["data"][:].astype(np.float32)
            # Convert nan to num, ideally in place to avoid duplication of data
            brain = np.nan_to_num(brain, copy=False)
            # brain = np.nan_to_num(h5_file.get("data")[:].astype('float32'))
        printlog("brain shape: {}".format(brain.shape))
        printlog("load duration: {} sec".format(time.time() - t0))

        ### MAKE CLUSTER DIRECTORY ###

        # cluster_dir = os.path.join(func_path, 'clustering')
        # if not os.path.exists(cluster_dir):
        #    os.mkdir(cluster_dir)
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
        # WHY NOT CLUSTER EVERYTHING? Why z slices?
        # Won't this limit supervoxel to a given slice? If we did neural activity = brain.reshape(-1,brain.shape[3])
        # it probably takes much longer (and more memory) but supervoxels would be in 3D?
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
        printlog("cluster average duration: {} sec".format(time.time() - t0))


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
    utils.print_function_start(logfile, WIDTH, "func2anat")

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
        fixed_brain_proxy.dataobj, dtype=np.float32
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
        moving_brain = np.asarray(moving_brain_proxy.dataobj, dtype=np.float32)

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
    path_to_read_fixed,
    path_to_read_moving,
    path_to_save,
    type_of_transform,
    resolution_of_fixed,
    resolution_of_moving,
    rule_name,
    iso_2um_fixed=True,
    iso_2um_moving=False,
    grad_step=0.2,
    flow_sigma=3,
    total_sigma=0,
    syn_sampling=32,
):
    """

    :param args:
    :return:
    """
    """
    Hardcoded stuff from preprocessing of brainsss
    res_anat = (0.653, 0.653, 1)
    res_func = (2.611, 2.611, 5)

    for fly in fly_dirs:
        fly_directory = os.path.join(dataset_path, fly)

        if loco_dataset:
            moving_path = os.path.join(fly_directory, 'func_0', 'imaging', 'functional_channel_1_mean.nii')
        else:
            moving_path = os.path.join(fly_directory, 'func_0', 'moco', 'functional_channel_1_moc_mean.nii')
        moving_fly = 'func'
        moving_resolution = res_func

        if loco_dataset:
            fixed_path = os.path.join(fly_directory, 'anat_0', 'moco', 'stitched_brain_red_mean.nii')
        else:
            fixed_path = os.path.join(fly_directory, 'anat_0', 'moco', 'anatomy_channel_1_moc_mean.nii')
        fixed_fly = 'anat'
        fixed_resolution = res_anat

        save_directory = os.path.join(fly_directory, 'warp')
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        type_of_transform = 'Affine'
        save_warp_params = True
        flip_X = False
        flip_Z = False

        low_res = False
        very_low_res = False

        iso_2um_fixed = True
        iso_2um_moving = False

        grad_step = 0.2
        flow_sigma = 3
        total_sigma = 0
        syn_sampling = 32

    """

    ###
    # Logging
    ###
    logfile = utils.create_logfile(fly_directory, function_name=rule_name)
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, WIDTH, rule_name)

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    path_to_read_fixed = utils.convert_list_of_string_to_posix_path(path_to_read_fixed)
    path_to_read_moving = utils.convert_list_of_string_to_posix_path(
        path_to_read_moving
    )
    path_to_save = utils.convert_list_of_string_to_posix_path(path_to_save)

    print("path_to_read_fixed" + repr(path_to_read_fixed))
    print("path_to_read_moving" + repr(path_to_read_moving))
    print("path_to_save" + repr(path_to_save))

    # logfile = args['logfile']
    # save_directory = args['save_directory']
    flip_X = (
        False  # args['flip_X'] # Todo - what does this do? Was set to false in brainsss
    )
    flip_Z = (
        False  # args['flip_Z'] # Todo - what does this do? Was set to false in brainsss
    )
    # type_of_transform = transform_type # args['type_of_transform']  # SyN or Affine,
    save_warp_params = True  # copy-paste from brainsss. args['save_warp_params']

    # There can only be one fixed brain, of course
    path_to_read_fixed = path_to_read_fixed[0]

    fixed_path = path_to_read_fixed  # args['fixed_path']
    # fixed_fly = 'anat' #args['fixed_fly']
    fixed_fly = path_to_read_fixed.name
    fixed_resolution = resolution_of_fixed  # args['fixed_resolution']

    # low_res = False # args['low_res']
    # very_low_res = False # args['very_low_res']

    iso_2um_fixed = (
        iso_2um_fixed  # args['iso_2um_fixed'] True for func2anat, False for anat2atlas
    )
    iso_2um_moving = iso_2um_moving  # args['iso_2um_moving'] False for func2anat, True for anat2atlas

    grad_step = grad_step  # args['grad_step']
    flow_sigma = flow_sigma  # args['flow_sigma']
    total_sigma = total_sigma  # args['total_sigma']
    syn_sampling = syn_sampling  # args['syn_sampling']

    # try:
    #    mimic_path = args['mimic_path']
    #    mimic_fly = args['mimic_fly']
    #    mimic_resolution = args['mimic_resolution']
    # except:
    mimic_path = None
    mimic_fly = None
    mimic_resolution = None

    ###################
    ### Load Brains ###
    ###################

    ### Fixed
    # fixed = np.asarray(nib.load(fixed_path).get_data().squeeze(), dtype='float32')
    # Doesn't load to memory
    fixed_brain_proxy = nib.load(path_to_read_fixed)
    # Load to memory
    fixed_brain = np.asarray(
        fixed_brain_proxy.dataobj, dtype=np.float32
    )  # I'm not using squeeze here! Might introduce
    # a bug so important to keep if statement below!
    utils.check_for_nan_and_inf_func(fixed_brain)

    if len(fixed_brain.shape) > 3:
        printlog(
            "WARNING: Here we should only have 3 dimensions not "
            + repr(fixed_brain.shape)
        )

    fixed_brain = ants.from_numpy(fixed_brain)
    fixed_brain.set_spacing(fixed_resolution)
    # if low_res:
    #    fixed = ants.resample_image(fixed, (256, 128, 49), 1, 0)
    # elif very_low_res:
    #    fixed = ants.resample_image(fixed, (128, 64, 49), 1, 0)
    # elif iso_2um_fixed:
    fixed_brain = ants.resample_image(fixed_brain, (2, 2, 2), use_voxels=False)

    # It's possible to have to channels for the 'moving' brain. Do this in a loop
    for current_path_to_read_moving, current_path_to_save in zip(
        path_to_read_moving, path_to_save
    ):
        # moving_path = current_path_to_read_moving  # args['moving_path']
        # moving_fly = 'func' #args['moving_fly']
        moving_fly = current_path_to_read_moving.name
        moving_resolution = resolution_of_moving  # args['moving_resolution']

        ### Moving
        # moving = np.asarray(nib.load(moving_path).get_data().squeeze(), dtype='float32')\
        moving_brain_proxy = nib.load(current_path_to_read_moving)
        moving_brain = np.asarray(moving_brain_proxy.dataobj, dtype=np.float32)
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
        moving_brain.set_spacing(moving_resolution)
        # if low_res:
        #    moving = ants.resample_image(moving, (256, 128, 49), 1, 0)
        # elif very_low_res:
        #    moving = ants.resample_image(moving, (128, 64, 49), 1, 0)
        # elif iso_2um_moving:
        moving_brain = ants.resample_image(moving_brain, (2, 2, 2), use_voxels=False)

        ### Mimic
        # if mimic_path is not None:
        #    mimic = np.asarray(nib.load(mimic_path).get_data().squeeze(), dtype='float32')
        #    if flip_X:
        #        mimic = mimic[::-1, :, :]
        #    if flip_Z:
        #        mimic = mimic[:, :, ::-1]
        #    mimic = ants.from_numpy(mimic)
        #    mimic.set_spacing(mimic_resolution)
        #    printlog('Starting {} to {}, with mimic {}'.format(moving_fly, fixed_fly, mimic_fly))
        # else:
        printlog("Starting registration of {} to {}".format(moving_fly, fixed_fly))

        #############
        ### Align ###
        #############

        t0 = time.time()
        # with stderr_redirected():  # to prevent dumb itk gaussian error bullshit infinite printing > Ohoh, hopefully doesn't
        # fill my log up
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
                fixed_path.name.split("/")[-1],
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
            # fwdtransforms_save_dir = os.path.join(save_directory, '{}-to-{}_fwdtransforms'.format(moving_fly, fixed_fly))
            fwdtransforms_save_folder = pathlib.Path(
                current_path_to_save.parent,
                "{}-to-{}_fwdtransforms".format(moving_fly, fixed_fly),
            )
            # if low_res:
            #    fwdtransforms_save_dir += '_lowres'
            if True in [iso_2um_moving, iso_2um_fixed]:
                fwdtransforms_save_path = pathlib.Path(
                    fwdtransforms_save_folder, "_2umiso"
                )
                # fwdtransforms_save_dir += '_2umiso'
            # if not os.path.exists(fwdtransforms_save_dir):
            #    os.mkdir(fwdtransforms_save_dir)
            fwdtransforms_save_folder.mkdir(exist_ok=True, parents=True)
            for source_path in fwdtransformlist:
                # source_file = source_path.split('/')[-1] # This should be correct - this comes from moco from ants
                source_file = pathlib.Path(source_path).name
                # target_path = os.path.join(fwdtransforms_save_dir, source_file)
                target_path = pathlib.Path(fwdtransforms_save_folder, source_file)
                shutil.copyfile(source_path, target_path)

        # Added this saving of inv transforms 2020 Dec 19
        if save_warp_params:
            invransformlist = moco["invtransforms"]
            # fwdtransforms_save_dir = os.path.join(save_directory, '{}-to-{}_invtransforms'.format(moving_fly, fixed_fly))
            invtransforms_save_folder = pathlib.Path(
                current_path_to_save.parent,
                "{}-to-{}_invtransforms".format(moving_fly, fixed_fly),
            )
            # if low_res:
            #    fwdtransforms_save_dir += '_lowres'
            if True in [iso_2um_moving, iso_2um_fixed]:
                invtransforms_save_folder = pathlib.Path(
                    invtransforms_save_folder, "_2umiso"
                )
                # fwdtransforms_save_dir += '_2umiso'
            # if not os.path.exists(fwdtransforms_save_dir):
            #    os.mkdir(fwdtransforms_save_dir)
            invtransforms_save_folder.mkdir(exist_ok=True, parents=True)
            for source_path in fwdtransformlist:
                # source_file = source_path.split('/')[-1]
                source_file = pathlib.Path(source_path).name
                # target_path = os.path.join(fwdtransforms_save_dir, source_file)
                target_path = pathlib.Path(invtransforms_save_folder, source_file)
                shutil.copyfile(source_path, target_path)

        ##################################
        ### Apply warp params to mimic ###
        ##################################

        # if mimic_path is not None:
        #    mimic_moco = ants.apply_transforms(fixed, mimic, moco['fwdtransforms'])

        ############
        ### Save ###
        ############

        # NOT SAVING MIMIC <------ MAY NEED TO CHANGE
        # if flip_X:
        #    save_file = os.path.join(save_directory, moving_fly + '_m' + '-to-' + fixed_fly)
        #    # save_file = os.path.join(save_directory, mimic_fly + '_m' + '-to-' + fixed_fly + '.nii')
        # else:
        #    save_file = os.path.join(save_directory, moving_fly + '-to-' + fixed_fly)
        #    # save_file = os.path.join(save_directory, mimic_fly + '-to-' + fixed_fly + '.nii')
        # nib.Nifti1Image(mimic_moco.numpy(), np.eye(4)).to_filename(save_file)
        # if low_res:
        #    save_file += '_lowres'
        # save_file += '.nii'
        nib.Nifti1Image(moco["warpedmovout"].numpy(), np.eye(4)).to_filename(
            current_path_to_save
        )

        # if flip_X:
        #     save_file = os.path.join(save_directory, moving_fly + '_m' + '-to-' + fixed_fly + '.nii')
        # else:
        #     save_file = os.path.join(save_directory, moving_fly + '-to-' + fixed_fly + '.nii')
        # nib.Nifti1Image(moco['warpedmovout'].numpy(), np.eye(4)).to_filename(save_file)


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
    utils.print_function_start(logfile, WIDTH, "clean_anatomy")

    # logfile = args['logfile']
    # directory = args['directory'] # directory will be a full path anat/moco
    # width = 120
    # printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')

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
    brain = np.asarray(brain_proxy.dataobj, dtype=np.float32)

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
    # behavior,
    fictrac_fps,
    metadata_path,
    fictrac_path,
):
    """
    Correlate z-scored brain activity with behavioral activity.

    The original function from brainsss was a triple looped call to scipy>pearsonr which took quite long as
    this example shows
    | SLURM | corr | 20559045 | COMPLETED | 00:28:55 | 4 cores | 21.7GB (69.7%)

    To speed the correlation up, I used only the parts from the scipy pearsonr function we need.
    One difference is that we only work in float32 space. The scipy function would cast everything as float64,
    doubling the memory requirements.
    When I subtract the vectorized result with the looped scipy pearson result I gat a max(diff) of 9.8e-8. This
    should not be relevant for us.

    See script 'pearson_correlation.py' - the vectorized version should take 0.03% of the time the
    scipy version does.

    :param args:
    :return:
    """
    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="correlation")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, WIDTH, "correlation")

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
    # Always define path in snakefile as it makes sure the file exists before even submitting a job
    # uses the same function as in fictrac_qc...
    fictrac_raw = fictrac_utils.load_fictrac(fictrac_path)
    resolution = 10  # desired resolution in ms
    expt_len = (fictrac_raw.shape[0] / fictrac_fps) * 1000
    # how many datapoints divide by how many times per seconds,in ms

    ### interpolate fictrac to match the timestamps from the microscope!
    fictrac_interp = fictrac_utils.smooth_and_interp_fictrac(
        fictrac_raw, fictrac_fps, resolution, expt_len, behavior, timestamps=timestamps
    )
    # Originally, there was a z parameter which was used as timestamps[:,z] to return the fictrac
    # data for a given z slice. We're not using it in the vectorized verison

    # It's possible to have more than one channel as the functional channel
    # Since we are memory limited, do correlation for both channels consecutively!
    for current_dataset_path, current_save_path in zip(dataset_path, save_path):
        if "nii" in current_dataset_path.name:
            # Avoid using get_fdata in loop. Something about cache is filling up memory quite fast!
            brain = np.asarray(
                nib.load(current_dataset_path).get_fdata().sqeeeze(), dtype="float32"
            )  # Never tested this
            # Also not sure it makes sense to define it as float32 because the h5>nii converter seems to save as int16.
            printlog(
                "Loaded nii file - BEWARE, I have not tested this. Better to use h5 files!"
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
            (brain.shape[0], brain.shape[1], brain.shape[2]), dtype=np.float32
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
            fictrac_mean = fictrac_interp[:, z].mean(dtype=np.float32)

            # >> Typical values for z scored brain seem to be between -25 and + 25.
            # It shouldn't be necessary to cast as float64. This then allows us
            # to do in-place operation!
            brain_mean_m = brain[:, :, z, :] - brain_mean[:, :, None]
            # fictrac data is small, so working with float64 shouldn't cost much memory!
            # Correction - if we cast as float64 and do dot product with brain, we'll copy the
            # brain to float64 array, balloning the memory requirement
            fictrac_mean_m = fictrac_interp[:, z].astype(np.float32) - fictrac_mean

            # Note from scipy pearson docs: This can overflow if brain_mean_m is, for example [-5e210, 5e210, 3e200, -3e200]
            # I doubt we'll ever get close to numbers like this.
            normbrain = np.linalg.norm(
                brain_mean_m, axis=-1
            )  # Make a copy, but since there's no time dimension it's quite small
            normfictrac = np.linalg.norm(fictrac_mean_m)

            # Calculate correlation
            corr_brain[:, :, z] = np.dot(
                brain_mean_m / normbrain[:, :, None], fictrac_mean_m / normfictrac
            )

        printlog(
            "Finished calculating correlation on {}; behavior: {}".format(
                current_dataset_path.name, behavior
            )
        )

        ### SAVE ###
        # if not os.path.exists(save_directory):
        #    os.mkdir(save_directory)
        current_save_path.parent.mkdir(exist_ok=True, parents=True)

        # if 'warp' in full_load_path:
        if "warp" in current_dataset_path.parts:
            warp_str = "_warp"
        else:
            warp_str = ""
        printlog("grey_only not implemented yet")
        # if grey_only:
        #    grey_str = '_grey'
        # else:
        #    grey_str = ''
        if "zscore" not in current_dataset_path.parts:
            no_zscore_highpass_str = "_mocoonly"
        else:
            no_zscore_highpass_str = ""

        save_file = pathlib.Path(current_save_path)  # ,
        #                        '{}_corr_{}{}{}{}.nii'.format(date, behavior, warp_str, grey_str, no_zscore_highpass_str))
        # Commented longer filename because we must define the output filename already in the snakefile.
        # save_file = os.path.join(save_directory,
        #                         '{}_corr_{}{}{}{}.nii'.format(date, behavior, warp_str, grey_str, no_zscore_highpass_str))
        aff = np.eye(4)
        object_to_save = nib.Nifti1Image(corr_brain, aff)
        # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
        object_to_save.to_filename(save_file)

        printlog("Saved {}".format(save_file))
        corr_utils.save_maxproj_img(image_to_max_project=corr_brain, path=save_file)
        printlog("Saved png plot")

        TESTING = False
        if TESTING:
            del brain  # remove brain from memory
            del corr_brain
            time.sleep(2)
            with h5py.File(current_dataset_path, "r") as hf:
                brain = hf["data"][:]  # load everything into memory!

            from scipy.stats import pearsonr

            # Keep for a few tests for now
            ##### BELLAS LOOP CODE BELOW ####
            # Get brain size
            x_dim = brain.shape[0]
            y_dim = brain.shape[1]
            z_dim = brain.shape[2]

            idx_to_use = list(range(timestamps.shape[0]))
            # timestamps.shape > (602, 49)
            # So we'd get a list going from 0 - 601

            corr_brain = np.zeros((x_dim, y_dim, z_dim))
            # For z dimension
            for z in range(z_dim):
                ### interpolate fictrac to match the timestamps of this slice
                printlog(f"{z}")
                # Why in here and what does z do?
                fictrac_interp = fictrac_utils.smooth_and_interp_fictrac(
                    fictrac_raw,
                    fictrac_fps,
                    resolution,
                    expt_len,
                    behavior,
                    timestamps=timestamps,
                    z=z,
                )
                # for x dimension
                for i in range(x_dim):
                    # for y dimension
                    for j in range(y_dim):
                        # nan to num should be taken care of in zscore, but checking here for some already processed brains
                        if np.any(np.isnan(brain[i, j, z, :])):
                            printlog(f"warning found nan at x = {i}; y = {j}; z = {z}")
                            corr_brain[i, j, z] = 0
                        elif len(np.unique(brain[i, j, z, :])) == 1:
                            #     if np.unique(brain[i,j,z,:]) != 0:
                            #         printlog(F'warning found non-zero constant value at x = {i}; y = {j}; z = {z}')
                            corr_brain[i, j, z] = 0
                        else:
                            # idx_to_use can be used to select a subset of timepoints
                            corr_brain[i, j, z] = pearsonr(
                                fictrac_interp[idx_to_use],
                                brain[i, j, z, :][idx_to_use],
                            )[0]

            save_file = pathlib.Path(
                current_save_path.parent,
                current_save_path.name.split(".nii")[0] + "_testing.nii",
            )
            printlog("Saving testfile to" + repr(save_file))
            aff = np.eye(4)
            object_to_save = nib.Nifti1Image(corr_brain, aff)
            # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
            object_to_save.to_filename(save_file)


def temporal_high_pass_filter(
    fly_directory, dataset_path, temporal_high_pass_filtered_path
):
    """
    Filters z-scored brain with scipt.ndimage.gaussian_filter1d in chunks due to memory demand of the large files
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
    utils.print_function_start(logfile, WIDTH, "temporal_high_pass_filter")
    # args = {'logfile': logfile,
    # 'load_directory': load_directory,
    # 'save_directory': save_directory,
    # 'brain_file': brain_file}
    #
    # load_directory = args['load_directory']
    # save_directory = args['save_directory']
    # brain_file = args['brain_file']
    # stepsize = 2

    # full_load_path = os.path.join(load_directory, brain_file)
    # save_file = os.path.join(save_directory, brain_file.split('.')[0] + '_highpass.h5')

    #####################
    ### SETUP LOGGING ###
    #####################

    # width = 120
    # logfile = args['logfile']
    # printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')

    #################
    ### HIGH PASS ###
    #################

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    temporal_high_pass_filtered_path = utils.convert_list_of_string_to_posix_path(
        temporal_high_pass_filtered_path
    )

    # From Bella, why so low???
    # stepsize = 2
    # stepsize = 500 # Doesn't seem to work - probably because loop is set up to work only with stepsize=2

    printlog("Beginning high pass")
    # dataset_path might be a list of 2 channels (or a list with one channel only)
    for current_dataset_path, current_temporal_high_pass_filtered_path in zip(
        dataset_path, temporal_high_pass_filtered_path
    ):
        printlog("Working on " + repr(current_dataset_path.name))
        with h5py.File(current_dataset_path, "r") as hf:
            data = hf[
                "data"
            ]  # this doesn't actually LOAD the data - it is just a proxy
            dims = np.shape(data)
            printlog("Data shape is {}".format(dims))

            # steps = list(range(0, dims[-1], stepsize))
            # steps.append(dims[-1])
            # Here we create a document we are going to write to in the loop
            with h5py.File(current_temporal_high_pass_filtered_path, "w") as f:
                # dset = f.create_dataset('data', dims, dtype='float32', chunks=True) # Original
                _ = f.create_dataset("data", dims, dtype="float32")

                data_mean = np.mean(data, axis=-1)
                smoothed_data = ndimage.gaussian_filter1d(
                    data, sigma=200, axis=-1, truncate=1
                )  # This for sure makes a copy of
                # the array, doubling memory requirements

                # To save memory, do in-place operations where possible
                # data_high_pass = data - smoothed_data + data_mean[:,:,:,None]
                data -= smoothed_data
                data += data_mean[:, :, :, None]
                f["data"][:, :, :, :] = data
                """for chunk_num in range(len(steps)):
                    print('cunk_num' + repr(chunk_num))
                    #t0 = time.time()
                    if chunk_num + 1 <= len(steps) - 1:
                        chunkstart = steps[chunk_num]
                        chunkend = steps[chunk_num + 1]
                        chunk = data[:, :, chunkstart:chunkend, :]
                        # Check if we really are getting a [128,256,2,3000] chunk
                        # over the z dimension. Interesting choice? Why not over time?
                        # > because we want to filter over time. Could choose any dimension except time!
                        chunk_mean = np.mean(chunk, axis=-1)
                        print("chunk_mean" + repr(chunk_mean))

                        ### SMOOTH ###
                        #t0 = time.time()
                        # I'm pretty sure this is identical to just filtering over the whole brain at once
                        smoothed_chunk = gaussian_filter1d(chunk, sigma=200, axis=-1, truncate=1)

                        ### Apply Smooth Correction ###
                        #t0 = time.time()
                        chunk_high_pass = chunk - smoothed_chunk + chunk_mean[:, :, :, None]  # need to add back in mean to preserve offset

                        ### Save ###
                        time.t0 = time()
                        f['data'][:, :, chunkstart:chunkend, :] = chunk_high_pass
                        """

    printlog("high pass done")


def zscore(fly_directory, dataset_path, zscore_path):
    """
    Remember, only the functional channel is z scored of course!!!!

    Expected memory needs:
    2x Filesize + 2~10 MB for the meanbrain and std

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
    :param args:
    :return:
    """
    # To reproduce Bella's script
    RUN_LOOPED = False
    if RUN_LOOPED:
        stepsize=100

    ##############
    ### ZSCORE ###
    ##############
    logfile = utils.create_logfile(fly_directory, function_name="zscore")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, WIDTH, "zscore")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    zscore_path = utils.convert_list_of_string_to_posix_path(zscore_path)

    printlog("Beginning ZSCORE")

    # we might get a second functional channel in the future!
    for current_dataset_path, current_zscore_path in zip(dataset_path, zscore_path):
        ####
        # testing!!!
        #####

        # load_directory = args['load_directory']
        # save_directory = args['save_directory']
        # brain_file = args['brain_file']
        # stepsize = 100

        # full_load_path = os.path.join(load_directory, brain_file)
        # save_file = os.path.join(save_directory, brain_file.split('.')[0] + '_zscore.h5')

        #####################
        ### SETUP LOGGING ###
        #####################

        # Open file - Must keep file open while accessing it.
        # I'm pretty sure we don't overwrite it because we open it as r.
        # This should mean that we are reading stuff into memory, of course
        with h5py.File(current_dataset_path, "r") as hf:
            data = hf[
                "data"
            ]  # this doesn't actually LOAD the data - it is just a proxy
            dims = np.shape(data)

            printlog("Data shape is {}".format(dims))

            if RUN_LOOPED:
                save_loop = pathlib.Path(current_zscore_path.parent, current_zscore_path.name + 'loop.h5')
                ####
                # Bella's code that allows chunking
                ###
                running_sum = np.zeros(dims[:3])
                running_sumofsq = np.zeros(dims[:3])
                steps = list(range(0, dims[-1], stepsize))
                steps.append(dims[-1])

                for chunk_num in range(len(steps)):
                    if chunk_num + 1 <= len(steps) - 1:
                        chunkstart = steps[chunk_num]
                        chunkend = steps[chunk_num + 1]
                        chunk = data[:, :, :, chunkstart:chunkend]
                        running_sum += np.sum(chunk, axis=3)
                        # printlog(F"vol: {chunkstart} to {chunkend} time: {time()-t0}")
                meanbrain = running_sum / dims[-1]

                np.save('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/loop_meanbrain.npy',
                        meanbrain)

                for chunk_num in range(len(steps)):
                    if chunk_num + 1 <= len(steps) - 1:
                        chunkstart = steps[chunk_num]
                        chunkend = steps[chunk_num + 1]
                        chunk = data[:, :, :, chunkstart:chunkend]
                        running_sumofsq += np.sum((chunk - meanbrain[..., None]) ** 2, axis=3)
                        # printlog(F"vol: {chunkstart} to {chunkend} time: {time()-t0}")
                final_std = np.sqrt(running_sumofsq / dims[-1])
                np.save('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/loopfinal_std.npy',
                        final_std)

                ### Calculate zscore and save ###

                with h5py.File(save_loop, 'w') as f:
                    dset = f.create_dataset('data', dims, dtype='float32', chunks=True)

                    for chunk_num in range(len(steps)):
                        if chunk_num + 1 <= len(steps) - 1:
                            chunkstart = steps[chunk_num]
                            chunkend = steps[chunk_num + 1]
                            chunk = data[:, :, :, chunkstart:chunkend]
                            running_sumofsq += np.sum((chunk - meanbrain[..., None]) ** 2, axis=3)
                            zscored = (chunk - meanbrain[..., None]) / final_std[..., None]
                            f['data'][:, :, :, chunkstart:chunkend] = np.nan_to_num(
                                zscored)  ### Added nan to num because if a pixel is a constant value (over saturated) will divide by 0
                            # printlog(F"vol: {chunkstart} to {chunkend} time: {time()-t0}")
            else:
                # Then do vectorized version
                # I think we don't have to worry about memory too much - since we only work
                # with one h5 file at a time and 30 minutes at float32 is ~20Gb
                # Expect a 4D array, xyz and the fourth dimension is time!
                meanbrain = np.nanmean(data, axis=3, dtype=np.float64)
                np.save('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/vect_meanbrain.npy',
                        meanbrain)
                # Might get out of memory error, test!
                final_std = np.std(data, axis=3, dtype=np.float64) # With float64 get memory error!
                np.save('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/vect_final_std.npy',
                        final_std)

                ### Calculate zscore and save ###

                # Calculate z-score
                # z_scored = (data - meanbrain[:,:,:,np.newaxis])/final_std[:,:,:,np.newaxis]
                # The above works, is easy to read but makes a copy in memory. Since brain data is
                # huge (easily 20Gb) we'll avoid making a copy by doing in place operations to save
                # memory! See docstring for more information

                # data will be data-meanbrain after this operation
                data -= meanbrain[:, :, :, np.newaxis]
                # Then it will be divided by std which leads to zscore
                data /= final_std[:, :, :, np.newaxis]
                # From the docs:
                # Chunking has performance implications. It’s recommended to keep the total size
                # of your chunks between 10 KiB and 1 MiB, larger for larger datasets. Also
                # keep in mind that when any element in a chunk is accessed, the entire chunk
                # is read from disk
                with h5py.File(current_zscore_path, "w") as file:
                    dset = file.create_dataset(
                        "data", data=data, dtype=np.float32
                    )  # , dims, dtype='float32', chunks=False)

                if len(dataset_path) > 1:
                    del data
                    printlog(
                        "Sleeping for 10 seconds before loading the next functional channel"
                    )
                    time.sleep(
                        10
                    )  # allow garbage collector to start cleaning up memory before potentially loading
                    # the other functional channel!

    printlog("zscore done")


def motion_correction(
    fly_directory,
    dataset_path,
    meanbrain_path,
    type_of_transform,
    output_format,
    flow_sigma,
    total_sigma,
    aff_metric,
    h5_path,
    anatomy_channel,
    functional_channels,
):
    """
    Motion correction function, essentially a copy from Bella's motion_correction.py script.
    The 'anatomy' and 'functional' channel(s) are defined in the 'fly.json' file of a given experimental folder.
    Here, we assume that there is a single 'anatomy' channel per experiment but it's possible to have zero, one or two
    functional channels.

    TODO: - ants seems to have a motion correction function. Try it.

    :param dataset_path: A list of paths
    :return:
    :param fly_directory:
    :param dataset_path:
    :param meanbrain_path:
    :param type_of_transform:
    :param output_format:
    :param flow_sigma:
    :param total_sigma:
    :param aff_metric:
    :param h5_path:
    :param anatomy_channel:
    :param functional_channels:
    :return:
    """

    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="motion_correction")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, WIDTH, "motion_correction")

    print("dataset_path" + repr(dataset_path))

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    meanbrain_path = utils.convert_list_of_string_to_posix_path(meanbrain_path)
    h5_path = utils.convert_list_of_string_to_posix_path(h5_path)

    parent_path = dataset_path[
        0
    ].parent  # the path witout filename, i.e. ../fly_001/func1/imaging/

    ####
    # Identify which channel is the anatomy, or 'master' channel
    ####

    # This first loop assigns a pathlib.Path object to the path_brain_anatomy variable
    # Here we assume that there is ONLY ONE anatomy channel
    path_brain_anatomy = None
    for current_brain_path, current_meanbrain_path in zip(dataset_path, meanbrain_path):
        if (
            "channel_1.nii" in current_brain_path.name
            and "channel_1" in anatomy_channel
        ):
            if path_brain_anatomy is None:
                path_brain_anatomy = current_brain_path
                path_meanbrain_anatomy = current_meanbrain_path
            else:
                printlog(
                    "!!!! ANATOMY CHANNEL AREADY DEFINED AS "
                    + repr(path_brain_anatomy)
                    + "!!! MAKE SURE TO HAVE ONLY ONE ANATOMY CHANNEL IN FLY.JSON"
                )
        elif (
            "channel_2.nii" in current_brain_path.name
            and "channel_2" in anatomy_channel
        ):
            if path_brain_anatomy is None:
                path_brain_anatomy = current_brain_path
                path_meanbrain_anatomy = current_meanbrain_path
            else:
                printlog(
                    "!!!! ANATOMY CHANNEL AREADY DEFINED AS "
                    + repr(path_brain_anatomy)
                    + "!!! MAKE SURE TO HAVE ONLY ONE ANATOMY CHANNEL IN FLY.JSON"
                )
        elif (
            "channel_3.nii" in current_brain_path.name
            and "channel_3" in anatomy_channel
        ):
            if path_brain_anatomy is None:
                path_brain_anatomy = current_brain_path
                path_meanbrain_anatomy = current_meanbrain_path
            else:
                printlog(
                    "!!!! ANATOMY CHANNEL AREADY DEFINED AS "
                    + repr(path_brain_anatomy)
                    + "!!! MAKE SURE TO HAVE ONLY ONE ANATOMY CHANNEL IN FLY.JSON"
                )

    # Next, check if there are any functional channels.
    path_brain_functional = []
    path_meanbrain_functional = []
    for current_brain_path, current_meanbrain_path in zip(dataset_path, meanbrain_path):
        if (
            "channel_1.nii" in current_brain_path.name
            and "channel_1" in functional_channels
        ):
            path_brain_functional.append(current_brain_path)
            path_meanbrain_functional.append(current_meanbrain_path)
        if (
            "channel_2.nii" in current_brain_path.name
            and "channel_2" in functional_channels
        ):
            path_brain_functional.append(current_brain_path)
            path_meanbrain_functional.append(current_meanbrain_path)
        if (
            "channel_3.nii" in current_brain_path.name
            and "channel_3" in functional_channels
        ):
            path_brain_functional.append(current_brain_path)
            path_meanbrain_functional.append(current_meanbrain_path)

    ####
    # DEFINE SAVEPATH
    ####

    # This could be integrated in loop above but for readability I'll make extra loops here
    # First, path to resulting anatomy motion corrected file.
    for current_h5_path in h5_path:
        if "channel_1" in current_h5_path.name and "channel_1" in anatomy_channel:
            path_h5_anatomy = current_h5_path
        elif "channel_2" in current_h5_path.name and "channel_2" in anatomy_channel:
            path_h5_anatomy = current_h5_path
        elif "channel_3" in current_h5_path.name and "channel_3" in anatomy_channel:
            path_h5_anatomy = current_h5_path

    # functional path is optional - to empty list in case no functional channel is provided
    path_h5_functional = []
    for current_h5_path in h5_path:
        if "channel_1" in current_h5_path.name and "channel_1" in functional_channels:
            path_h5_functional.append(current_h5_path)
        if "channel_2" in current_h5_path.name and "channel_2" in functional_channels:
            path_h5_functional.append(current_h5_path)
        if "channel_3" in current_h5_path.name and "channel_3" in functional_channels:
            path_h5_functional.append(current_h5_path)

    # Print useful info to log file
    printlog(f"Dataset path{parent_path.name:.>{WIDTH - 12}}")
    printlog(f"Brain anatomy{path_brain_anatomy.name:.>{WIDTH - 12}}")
    if len(path_brain_functional) > 0:
        printlog(f"Brain functional{str(path_brain_functional[0].name):.>{WIDTH - 12}}")
        if len(path_brain_functional) > 1:
            printlog(
                f"Brain functional#2{str(path_brain_functional[1].name):.>{WIDTH - 12}}"
            )
    else:
        printlog("No functional path found {:.>{WIDTH - 12}}")

    printlog(f"type_of_transform{type_of_transform:.>{WIDTH - 17}}")
    printlog(f"output_format{output_format:.>{WIDTH - 13}}")
    printlog(f"flow_sigma{flow_sigma:.>{WIDTH - 10}}")
    printlog(f"total_sigma{total_sigma:.>{WIDTH - 11}}")


    # This can't really happen with snakemake as it won't even start the job without
    # the input files present!

    ########################################
    ### Read Channel 1 imaging data ###
    ########################################
    ### Get Brain Shape ###
    img_anatomy = nib.load(path_brain_anatomy)  # this loads a proxy
    anatomy_shape = img_anatomy.header.get_data_shape()
    brain_dims = anatomy_shape
    printlog(f"Master brain shape{str(brain_dims):.>{WIDTH - 18}}")

    ################################
    ### DEFINE STEPSIZE/CHUNKING ###
    ################################
    # Ideally, we define the number of slices on the (x,y) size of the images!

    path_brain_anatomy
    if "func" in path_brain_anatomy.parts:
        scantype = "func"
        stepsize = 100  # Bella had this at 100
    elif "anat" in path_brain_anatomy.parts:
        scantype = "anat"
        stepsize = 5  # Bella had this at 5
    else:
        scantype = "Unknown"
        stepsize = 100
        printlog(
            f"{'   Could not determine scantype. Using default stepsize of 100   ':*^{WIDTH}}"
        )
    printlog(f"Scantype{scantype:.>{WIDTH - 8}}")
    printlog(f"Stepsize{stepsize:.>{WIDTH - 8}}")


    ########################################
    ### Read Meanbrain of Channel 1 ###
    ########################################

    meanbrain_proxy = nib.load(path_meanbrain_anatomy)
    # TODO change this to np.float32 (from np.uint16) to minimze precision loss
    meanbrain = np.asarray(
        meanbrain_proxy.dataobj, dtype=np.uint16
    )  # <<<<!!! This is quite strange - meanbrain is float64!!
    # meanbrain = np.asarray(nib.load(path_meanbrain_anatomy).get_fdata(), dtype='uint16')
    # get_fdata() loads data into memory and sometimes doesn't release it.
    fixed_ants = ants.from_numpy(np.asarray(meanbrain, dtype="float32"))
    printlog(f"Loaded meanbrain{path_meanbrain_anatomy.name:.>{WIDTH - 16}}")

    #########################
    ### Load Mirror Brain ###
    #########################

    if len(path_brain_functional) > 0:  # is not None:
        img_functional_one_proxy = nib.load(path_brain_functional[0])  # this loads a proxy
        # make sure channel anatomy and functional have same shape
        functional_one_shape = img_functional_one_proxy.header.get_data_shape()
        if anatomy_shape != functional_one_shape:
            printlog(
                f"{'   WARNING Channel anatomy and functional do not have the same shape!   ':*^{WIDTH}}"
            )
            printlog("{} and {}".format(anatomy_shape, functional_one_shape))

        # Once we have the 3 detector channel in the bruker, we'll have 2 functional channels
        if len(path_brain_functional) > 1:
            img_functional_two = nib.load(
                path_brain_functional[1]
            )  # this loads a proxy
            # make sure both functional channels have same dims
            functional_two_shape = img_functional_two.header.get_data_shape()
            if functional_one_shape != functional_two_shape:
                printlog(
                    f"{'   WARNING Channel functional one and functional two do not have the same shape!   ':*^{WIDTH}}"
                )
                printlog("{} and {}".format(functional_one_shape, functional_two_shape))
    ############################################################
    ### Make Empty MOCO files that will be filled vol by vol ###
    ############################################################
    MEMORY_ONLY = True
    if not MEMORY_ONLY:
        # This should most likely live on scratch as it is accessed several times.
        # h5_file_name = f"{path_brain_master.name.split('.')[0]}_moco.h5"
        # moco_dir, savefile_master = brainsss.make_empty_h5(h5_path_scratch, h5_file_name, brain_dims)#, save_type)
        # Make 'moco' dir in imaging path
        path_h5_anatomy.parent.mkdir(parents=True, exist_ok=True)
        # Create empty h5 file
        with h5py.File(path_h5_anatomy, "w") as file:
            _ = file.create_dataset("data", brain_dims, dtype="float32", chunks=True)
        printlog(f"Created empty hdf5 file{path_h5_anatomy.name:.>{WIDTH - 23}}")

        if len(path_h5_functional) > 0:
            with h5py.File(path_h5_functional[0], "w") as file:
                _ = file.create_dataset("data", brain_dims, dtype="float32", chunks=True)
            printlog(f"Created empty hdf5 file{path_h5_functional[0].name:.>{WIDTH - 23}}")

            if len(path_h5_functional) > 1:
                with h5py.File(path_h5_functional[1], "w") as file:
                    _ = file.create_dataset(
                        "data", brain_dims, dtype="float32", chunks=True
                    )
                printlog(
                    f"Created empty hdf5 file{path_h5_functional[1].name:.>{WIDTH - 23}}"
                )

    #################################
    ### Perform Motion Correction ###
    #################################
    printlog(f"{'   STARTING MOCO   ':-^{WIDTH}}")
    transform_matrix = []

    ### prepare chunks to loop over ###
    # the chunks defines how many vols to moco before saving them to h5 (this save is slow, so we want to do it less often)
    steps = list(range(0, brain_dims[-1], stepsize))
    # add the last few volumes that are not divisible by stepsize
    if brain_dims[-1] > steps[-1]:
        steps.append(brain_dims[-1])

    # loop over all brain vols, motion correcting each and insert into hdf5 file on disk
    # for i in range(brain_dims[-1]):
    start_time = time.time()
    print_timer = time.time()

    # We should have enough RAM to keep everything in memory -
    # It seems to be slow to write to h5py unless it's optimized - not sure if that's one reason why moco is so slow:
    # https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage

    if MEMORY_ONLY:
        # Input be 4D array! Else fails here. This is preallocation of memory for the resulting motion-corrected
        # anatomy channel
        moco_anatomy = np.zeros((brain_dims[0], brain_dims[1], brain_dims[2], brain_dims[3]), dtype=np.float32)
        moco_anatomy.fill(np.nan) # avoid that missing values end up as 0!
        #
        transform_matrix = np.zeros((12, brain_dims[3])) # Lets see if that's correct

        if len(path_brain_functional) > 0:
            # This is preallocation of memory for the resulting motion-corrected functional channel
            moco_functional_one = np.zeros((brain_dims[0], brain_dims[1], brain_dims[2], brain_dims[3]), dtype=np.float32)
            moco_functional_one.fill(np.nan)

        # Register one frame after another
        for current_frame in range(brain_dims[-1]): # that's the t-dimension
            current_moving_frame = img_anatomy.dataobj[:,:,:,current_frame] # load a single frame into memory (memory-cheap, but i/o heavy)
            current_moving_frame_ants = ants.from_numpy(np.asarray(current_moving_frame, dtype=np.float32)) # cast np.uint16 as float32 because ants requires it

            moco = ants.registration(
                    fixed_ants,
                    current_moving_frame_ants,
                    type_of_transform=type_of_transform,
                    flow_sigma=flow_sigma,
                    total_sigma=total_sigma,
                    aff_metric=aff_metric,
                    #outputprefix='' # MAYBE writing on scratch will make this faster?
            )
            moco_anatomy[:,:,:,current_frame] = moco["warpedmovout"].numpy()
            # Get transform info, for saving and applying transform to functional channel
            transformlist = moco["fwdtransforms"]

            ### APPLY TRANSFORMS TO FUNCTIONAL CHANNELS ###
            if len(path_brain_functional) > 0:
                current_functional_frame = img_functional_one_proxy.dataobj[:,:,:, current_frame]
                current_functional_frame_ants = ants.from_numpy(
                    np.asarray(current_functional_frame, dtype=np.float32)
                )
                moco_functional = ants.apply_transforms(
                    fixed_ants, current_functional_frame_ants, transformlist
                )
                print("fixed_ants " + repr(fixed_ants))
                moco_functional_one[:,:,:,current_frame] = moco_functional.numpy()
                # TODO add more channels here!!!

            # Delete transform info - might be worth keeping instead of huge resulting file? TBD
            for x in transformlist:
                if ".mat" in x:
                    temp = ants.read_transform(x)
                    transform_matrix[:, current_frame] = temp.parameters
                # lets' delete all files created by ants - else we quickly create thousands of files!
                pathlib.Path(x).unlink()

            ### Print progress ###
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 * 60:  # if less than 1 min has elapsed
                print_frequency = (
                    1  # print every sec if possible, but will be every vol
                )
            elif elapsed_time < 5 * 60:
                print_frequency = 1 * 60
            elif elapsed_time < 30 * 60:
                print_frequency = 5 * 60
            else:
                print_frequency = 60 * 60
            if time.time() - print_timer > print_frequency:
                print_timer = time.time()
                moco_utils.print_progress_table_moco(
                    total_vol=brain_dims[-1],
                    complete_vol=current_frame,
                    printlog=printlog,
                    start_time=start_time,
                    width=WIDTH,
                )

        aff = np.eye(4)
        anatomy_save_object = nib.Nifti1Image(moco_anatomy, aff)
        anatomy_save_object.to_filename(path_h5_anatomy)
        if len(path_brain_functional) > 0:
            functional_one_save_object = nib.Nifti1Image(moco_functional_one)
            functional_one_save_object.to_filename(path_h5_functional[0])

    if not MEMORY_ONLY:
        # For timepoints / stepsize. e.g. if have 300 timepoints and stepsize 100 I get len(steps)=3
        for j in range(len(steps) - 1):
            # printlog(F"j: {j}")

            ### LOAD A SINGLE BRAIN VOL ###
            moco_anatomy_chunk = [] # This really should be a preallocated array!!!
            moco_functional_one_chunk = [] # This really should be a preallocated array!!!
            moco_functional_two_chunk = [] # This really should be a preallocated array!!!
            # for each timePOINT!
            for i in range(stepsize):
                # that's a number
                index = steps[j] + i
                # for the very last j, adding the step size will go over the dim, so need to stop here
                if index == brain_dims[-1]:
                    break
                # that's a single slice in time
                vol = img_anatomy.dataobj[..., index] #
                moving = ants.from_numpy(np.asarray(vol, dtype="float32"))

                ### MOTION CORRECT ###
                # This step doesn't seem to take very long - ~2 seconds on a 256,128,49 volume
                moco = ants.registration(
                    fixed_ants,
                    moving,
                    type_of_transform=type_of_transform,
                    flow_sigma=flow_sigma,
                    total_sigma=total_sigma,
                    aff_metric=aff_metric,
                )
                # moco_ch1 = moco['warpedmovout'].numpy()
                moco_anatomy = moco["warpedmovout"].numpy()
                # moco_ch1_chunk.append(moco_ch1)
                moco_anatomy_chunk.append(moco_anatomy) # This really should be a preallocated array!!!
                transformlist = moco["fwdtransforms"]
                # printlog(F'vol, ch1 moco: {index}, time: {time.time()-t0}')

                ### APPLY TRANSFORMS TO FUNCTIONAL CHANNELS ###
                if len(path_brain_functional) > 0:
                    vol = img_functional_one_proxy.dataobj[..., index]
                    functional_one_moving = ants.from_numpy(
                        np.asarray(vol, dtype="float32")
                    )
                    moco_functional_one = ants.apply_transforms(
                        fixed_ants, functional_one_moving, transformlist
                    )
                    moco_functional_one = moco_functional_one.numpy()
                    moco_functional_one_chunk.append(moco_functional_one)
                    # If a second functional channel exists, also apply to this one
                    if len(path_brain_functional) > 1:
                        vol = img_functional_two.dataobj[..., index]
                        functional_two_moving = ants.from_numpy(
                            np.asarray(vol, dtype="float32")
                        )
                        moco_functional_two = ants.apply_transforms(
                            fixed_ants, functional_two_moving, transformlist
                        )
                        moco_functional_two = moco_functional_two.numpy()
                        moco_functional_two_chunk.append(moco_functional_two)
                ### APPLY TRANSFORMS TO CHANNEL 2 ###
                # t0 = time.time()
                # if path_brain_mirror is not None:
                #    vol = img_ch2.dataobj[..., index]
                #    ch2_moving = ants.from_numpy(np.asarray(vol, dtype='float32'))
                #    moco_ch2 = ants.apply_transforms(fixed, ch2_moving, transformlist)
                #    moco_ch2 = moco_ch2.numpy()
                #    moco_ch2_chunk.append(moco_ch2)
                # printlog(F'moco vol done: {index}, time: {time.time()-t0}')

                ### SAVE AFFINE TRANSFORM PARAMETERS FOR PLOTTING MOTION ###
                transformlist = moco["fwdtransforms"]
                for x in transformlist:
                    if ".mat" in x:
                        temp = ants.read_transform(x)
                        transform_matrix.append(temp.parameters)

                ### DELETE FORWARD TRANSFORMS ###
                transformlist = moco["fwdtransforms"]
                for x in transformlist:
                    if ".mat" not in x:
                        # print('Deleting fwdtransforms ' + x) # Yes, these are files
                        pathlib.Path(x).unlink()
                        # os.remove(x) # todo Save memory? # I think that because otherwise we'll
                        # quickly have tons of files in the temp folder which will make it hard to
                        # know which one we are looking for.

                ### DELETE INVERSE TRANSFORMS ###
                transformlist = moco[
                    "invtransforms"
                ]  # I'm surprised this doesn't lead to an error because it doesn't seem taht moco['invtransforms'] is defined anywhere
                for x in transformlist:
                    if ".mat" not in x:
                        # print('Deleting invtransforms ' + x)
                        pathlib.Path(x).unlink()
                        # os.remove(x) # todo Save memory? # I think that because otherwise we'll
                        # quickly have tons of files in the temp folder which will make it hard to
                        # know which one we are looking for.
                        # No it seems mainly a memory/filenumber issue: for each registration call (so several
                        # 1000 times per call of this function) a >10Mb file is created. That adds of course

                ### Print progress ###
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 * 60:  # if less than 1 min has elapsed
                    print_frequency = (
                        1  # print every sec if possible, but will be every vol
                    )
                elif elapsed_time < 5 * 60:
                    print_frequency = 1 * 60
                elif elapsed_time < 30 * 60:
                    print_frequency = 5 * 60
                else:
                    print_frequency = 60 * 60
                if time.time() - print_timer > print_frequency:
                    print_timer = time.time()
                    moco_utils.print_progress_table_moco(
                        total_vol=brain_dims[-1],
                        complete_vol=index,
                        printlog=printlog,
                        start_time=start_time,
                        width=WIDTH,
                    )

            moco_anatomy_chunk = np.moveaxis(np.asarray(moco_anatomy_chunk), 0, -1)
            if len(path_brain_functional) > 0:
                moco_functional_one_chunk = np.moveaxis(
                    np.asarray(moco_functional_one_chunk), 0, -1
                )
                if len(path_brain_functional) > 1:
                    moco_functional_two_chunk = np.moveaxis(
                        np.asarray(moco_functional_two_chunk), 0, -1
                    )
            # moco_ch1_chunk = np.moveaxis(np.asarray(moco_ch1_chunk), 0, -1)
            # if path_brain_mirror is not None:
            #    moco_ch2_chunk = np.moveaxis(np.asarray(moco_ch2_chunk), 0, -1)
            # printlog("chunk shape: {}. Time: {}".format(moco_ch1_chunk.shape, time.time()-t0))

            ### APPEND WARPED VOL TO HD5F FILE - CHANNEL 1 ###
            with h5py.File(path_h5_anatomy, "a") as f:
                f["data"][..., steps[j] : steps[j + 1]] = moco_anatomy_chunk
            # t0 = time.time()
            # with h5py.File(path_h5_master, 'a') as f:
            #    f['data'][..., steps[j]:steps[j + 1]] = moco_ch1_chunk
            # printlog(F'Ch_1 append time: {time.time-t0}')

            ### APPEND WARPED VOL TO HD5F FILE - CHANNEL 2 ###
            if len(path_brain_functional) > 0:
                with h5py.File(path_h5_functional[0], "a") as f:
                    f["data"][..., steps[j] : steps[j + 1]] = moco_functional_one_chunk
                if len(path_brain_functional) > 1:
                    with h5py.File(path_h5_functional[1], "a") as f:
                        f["data"][..., steps[j] : steps[j + 1]] = moco_functional_two_chunk
            # t0 = time.time()
            # if path_brain_mirror is not None:
            #    with h5py.File(path_h5_mirror, 'a') as f:
            #        f['data'][..., steps[j]:steps[j + 1]] = moco_ch2_chunk
            # printlog(F'Ch_2 append time: {time.time()-t0}')

    ### SAVE TRANSFORMS ###
    printlog("saving transforms")
    printlog(f"path_h5_master: {path_h5_anatomy}")
    transform_matrix = np.array(transform_matrix)
    # save_file = os.path.join(moco_dir, 'motcorr_params')
    save_file = pathlib.Path(path_h5_anatomy.parent, "motcorr_params")
    np.save(save_file, transform_matrix)

    ### MAKE MOCO PLOT ###
    printlog("making moco plot")
    printlog(f"moco_dir: {path_h5_anatomy.parent}")
    moco_utils.save_moco_figure(
        transform_matrix=transform_matrix,
        parent_path=parent_path,
        moco_dir=path_h5_anatomy.parent,
        printlog=printlog,
    )

    ### OPTIONAL: SAVE REGISTERED IMAGES AS NII ###
    if output_format == "nii":
        printlog("saving .nii images")

        # Save master:
        nii_savefile_master = moco_utils.h5_to_nii(path_h5_anatomy)
        printlog(f"nii_savefile_master: {str(nii_savefile_master.name)}")
        if (
            nii_savefile_master is not None
        ):  # If .nii conversion went OK, delete h5 file
            printlog("deleting .h5 file at {}".format(path_h5_anatomy))
            path_h5_anatomy.unlink()  # delete file
        else:
            printlog("nii conversion failed for {}".format(path_h5_anatomy))
        # Save mirror:
        if len(path_h5_functional) > 0:
            nii_savefile_functional_one = moco_utils.h5_to_nii(path_h5_functional[0])
            printlog(f"nii_savefile_mirror: {str(nii_savefile_functional_one)}")
            if (
                nii_savefile_functional_one is not None
            ):  # If .nii conversion went OK, delete h5 file
                printlog("deleting .h5 file at {}".format(path_h5_functional[0]))
                # os.remove(savefile_mirror)
                path_h5_functional[0].unlink()
            else:
                printlog("nii conversion failed for {}".format(path_h5_functional[0]))

            if len(path_h5_functional) > 1:
                nii_savefile_functional_two = moco_utils.h5_to_nii(
                    path_h5_functional[1]
                )
                printlog(f"nii_savefile_mirror: {str(nii_savefile_functional_two)}")
                if (
                    nii_savefile_functional_two is not None
                ):  # If .nii conversion went OK, delete h5 file
                    printlog("deleting .h5 file at {}".format(path_h5_functional[1]))
                    # os.remove(savefile_mirror)
                    path_h5_functional[1].unlink()
                else:
                    printlog(
                        "nii conversion failed for {}".format(path_h5_functional[1])
                    )
        # Save mirror:
        # if path_brain_mirror is not None:
        #    nii_savefile_mirror = moco_utils.h5_to_nii(path_h5_functional[0])
        #    printlog(F"nii_savefile_mirror: {str(nii_savefile_mirror)}")
        #    if nii_savefile_mirror is not None:  # If .nii conversion went OK, delete h5 file
        #        printlog('deleting .h5 file at {}'.format(path_h5_mirror))
        #        #os.remove(savefile_mirror)
        #        path_h5_mirror.unlink()
        #    else:
        #        printlog('nii conversion failed for {}'.format(path_h5_mirror))


def copy_to_scratch(fly_directory, paths_on_oak, paths_on_scratch):
    """
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
    utils.print_function_start(logfile, WIDTH, "copy_to_scratch")

    for current_file_src, current_file_dst in zip(paths_on_oak, paths_on_scratch):
        # make folder if not exist
        pathlib.Path(current_file_dst).parent.mkdir(exist_ok=True, parents=True)
        # copy file
        shutil.copy(current_file_src, current_file_dst)
        printlog("Copied: " + repr(current_file_dst))


def make_mean_brain(
    fly_directory, meanbrain_n_frames, path_to_read, path_to_save, rule_name
):
    """
    Function to calculate meanbrain.
    This is based on Bella's meanbrain script.
    :param fly_directory: pathlib.Path object to a 'fly' folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
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
    utils.print_function_start(logfile, WIDTH, rule_name)

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
            brain_proxy = nib.load(
                current_path_to_read
            )  # Doesn't load anything, just points to a given location
            brain_data = np.asarray(
                brain_proxy.dataobj, dtype="uint16"
            )  # loads data to memory.
        elif current_path_to_read.suffix == ".h5":
            with h5py.File(current_path_to_read, "r") as hf:
                brain_data = np.asarray(hf["data"][:], dtype="uint16") #<- CHECK THIS, i think moco_brains are saved as float 32!!!
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
            meanbrain = np.mean(brain_data[..., : int(meanbrain_n_frames)], axis=-1)#, dtype=np.float32) <- change back!
        else:  # average over all frames
            meanbrain = np.mean(brain_data, axis=-1)#, dtype=np.float32) <- change back!

        printlog("Datatype of meanbrain: " + repr(meanbrain.dtype))
        ###
        # SAVE MEANBRAIN
        ###
        aff = np.eye(4)
        object_to_save = nib.Nifti1Image(
            meanbrain, aff
        )
        object_to_save.to_filename(current_path_to_save)

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

    :param logfile: logfile to be used for all errors (stderr) and console outputs (stdout)
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
    utils.print_function_start(logfile, WIDTH, "bleaching_qc")

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
        # Load data into memory
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


def fictrac_qc(fly_directory, fictrac_file_paths, fictrac_fps):
    """
    Perform fictrac quality control.
    This is based on Bella's fictrac_qc.py  script.
    :param fly_directory: a pathlib.Path object to a 'fly' folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param fictrac_file_paths: a list of paths as pathlib.Path objects
    :param fictrac_fps: frames per second of the videocamera recording the fictrac data, an integer
    """
    ####
    # LOGGING
    ####
    logfile = utils.create_logfile(fly_directory, function_name="fictrac_qc")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, WIDTH, "fictrac_qc")

    ####
    # QUALITY CONTROL
    ####
    for current_file in fictrac_file_paths:
        printlog("Currently looking at: " + repr(current_file))
        fictrac_raw = fictrac_utils.load_fictrac(current_file)
        # This should yield something like 'fly_001/func0/fictrac
        full_id = ", ".join(current_file.parts[-4:-2])
        #full_id = ", ".join(str(current_file).split("/")[-3:1])

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
            fictrac, current_file, full_id,  fixed_crop=True
        )
        fictrac_utils.make_2d_hist(
            fictrac, current_file, full_id, fixed_crop=False
        )
        fictrac_utils.make_velocity_trace(
            fictrac, current_file, full_id, time_for_plotting,
        )

    ###
    # LOG SUCCESS
    ###
    printlog(f"Prepared fictrac QC plot and saved in: {str(current_file.parent):.>{WIDTH - 20}}")


def fly_builder(user, import_dirs, dataset_dirs):
    """
    Move folders from imports to fly dataset - need to restructure folders.
    This is based on Bella's 'fly_builder.py' script

    # Note: I removed the discrepancy between 'anat' and 'func' folders. All files
    are now just called 'channel_1.nii' or 'channel_2.nii' as it makes handling filenames
    much, much simpler, especially when going through snakemake

    :param logfile: logfile to be used for all errors (stderr) and console outputs (stdout)
    :param user: your SUnet ID as a string
    :param dirs_to_build: a list of folders to build. e.g. dir_to_build = ['20231301'] or  dir_to_build = ['20232101', '20231202']
    :param target_folder:
    :return:
    """
    try:
        # Loop through the list of dirs
        for current_import_dir, current_dataset_dir in zip(import_dirs, dataset_dirs):
            ###
            # Logging
            ###
            logfile = utils.create_logfile(
                current_dataset_dir, function_name="fly_builder"
            )
            printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
            utils.print_function_start(logfile, WIDTH, "fly_builder")
            printlog(f"Building flies from: {str(current_import_dir):.>{WIDTH - 22}}")

            printlog(
                f"Building fly directory:{str(current_dataset_dir):.>{WIDTH - 22}}"
            )
            printlog(
                f"\n{'   Building ' + current_import_dir.name + ' as ' + str(current_dataset_dir.name) + '   ':-^{WIDTH}}"
            )

            ###
            # Dict to keep track of filepaths
            ###
            fly_dirs_dict = {}
            fly_dirs_dict["fly ID"] = current_dataset_dir.name
            ###
            # Copy fly data
            ####
            fly_dirs_dict = copy_fly(
                current_import_dir, current_dataset_dir, printlog, user, fly_dirs_dict
            )

            ###
            # Avoid for now - it's going to be tricky getting the correct time for fly.json.
            # much easier to rely on user input here.
            # We have cleaner way of sorting this anyway with the xls file
            # Add date to fly.json file
            try:
                add_date_to_fly(current_dataset_dir)
            except Exception as e:
                printlog(str(e))
                printlog(str(e))
                printlog(traceback.format_exc())

            # Add json metadata to master dataset
            try:
                add_fly_to_xlsx(current_dataset_dir, printlog)
            except Exception as e:
                printlog("Could not add xls data because of error:")
                printlog(str(e))
                printlog(traceback.format_exc())

            # Save json file with all relevant paths
            with open(
                pathlib.Path(
                    current_dataset_dir, current_dataset_dir.name + "_dirs.json"
                ),
                "w",
            ) as outfile:
                json.dump(fly_dirs_dict, outfile)

            # If we are here it should mean that everything above has been copied as expected.
            # We can therefore delete the 'incomplete' file in this folder
            try:
                pathlib.Path(current_dataset_dir, "incomplete").unlink()
                printlog(
                    f'Deleted "incomplete" file in :{str(current_dataset_dir):.>{WIDTH}}'
                )
            except FileNotFoundError:
                printlog(
                    f'"Incomplete" file not found in! :{str(current_dataset_dir):.>{WIDTH}}'
                )
            # In case part of the import folder is being copied and the program crashes, it might
            # lead to accidentally copying all flies from import to new flies in data which might lead
            # to data duplication.
            # To avoid this, we write in the import folder a file called 'data_transfered_to.txt'
            # with the path of the target
            already_copied_path = pathlib.Path(
                current_import_dir, "data_transfered_to.txt"
            )
            with open(already_copied_path, "w") as outputfile:
                outputfile.write(str(current_dataset_dir))
            printlog(
                f"Wrote data_already_transfered_to.txt file in :{str(current_import_dir):.>{WIDTH}}"
            )

    except Exception as error_stack:
        printlog('!!! ERROR !!! -> check error file in "log" folder')
        logfile = utils.create_logfile(
            current_dataset_dir, function_name="ERROR_fly_builder"
        )
        utils.write_error(logfile=logfile, error_stack=error_stack, width=WIDTH)


def add_date_to_fly(fly_folder):
    """get date from xml file and add to fly.json"""

    # Check if there are func folders:
    candidate_folders = [
        pathlib.Path(fly_folder, x) for x in fly_folder.iterdir() if "func" in x.name
    ]
    # if not...
    if len(candidate_folders) == 0:
        # ...check if there are anat folders:
        candidate_folders = [
            pathlib.Path(fly_folder, x)
            for x in fly_folder.iterdir()
            if "anat" in x.name
        ]

    if len(candidate_folders) > 0:
        candidate_folder = candidate_folders[0]
        xml_file = pathlib.Path(candidate_folder, "imaging", "recording_metadata.xml")
        print("xml_file path" + repr(xml_file))
        # Extract datetime
        datetime_str, _, _ = get_datetime_from_xml(xml_file)
        # Get just date
        date = datetime_str.split("-")[0]
        time = datetime_str.split("-")[1]

        ### Add to fly.json
        json_file = pathlib.Path(fly_folder, "fly.json")
        # json_file = os.path.join(destination_fly, 'fly.json')
        with open(json_file, "r+") as f:
            metadata = json.load(f)
            metadata["date"] = str(date)
            metadata["time"] = str(time)
            f.seek(0)
            json.dump(metadata, f, indent=4)
            f.truncate()

    else:
        print(
            'Unable to find folder called "anat" or "func" to read "recording_metadata.xml'
        )


def copy_fly(import_dir, dataset_dir, printlog, user, fly_dirs_dict):
    """
    #####
    # Todo - make sure the scratch folder is empty!!!!!
    #####
    Otherwise it's easy to imagine a situation where a user creates ../fly002/
    deletes it and records another fly which is then called fly002. All analysis
    would then be done on the initial fly002!

    There will be two types of folders in a fly folder.
    1) func_x folder
    2) anat_x folder
    For functional folders, need to copy fictrac and visual as well
    For anatomy folders, only copy folder. There will also be
    3) fly json data
    4) the recording metadata was called 'anatomy.xml' and 'functional.xml' in the past.
       this is changed to always be called 'recording_metadata.xml'
    """
    # look at every item in source fly folder
    # This is a folder like: '/oak/stanford/groups/trc/data/David/Bruker/imports/20231207/fly1'
    for current_import_file_or_folder in import_dir.iterdir():
        # This should be e.g. directory such as anat1 or func0 or fly.json
        print(
            "Currently looking at source_item: {}".format(
                current_import_file_or_folder.name
            )
        )
        # Handle folders
        if current_import_file_or_folder.is_dir():
            # Call this folder source expt folder
            current_import_imaging_folder = current_import_file_or_folder
            # Make the same folder in destination fly folder
            current_dataset_folder = pathlib.Path(
                dataset_dir, current_import_file_or_folder.name
            )
            # current_dataset_folder.mkdir(parents=True, exist_ok=True)

            # Is this folder an anatomy or functional folder?
            if "anat" in current_import_imaging_folder.name:
                # If anatomy folder, just copy everything
                # Make imaging folder and copy
                # imaging_destination = os.path.join(expt_folder, 'imaging')
                imaging_destination = pathlib.Path(current_dataset_folder, "imaging")
                # os.mkdir(imaging_destination)
                imaging_destination.mkdir(parents=True, exist_ok=True)
                copy_bruker_data(
                    current_import_imaging_folder, imaging_destination, "anat", printlog
                )
                current_fly_dir_dict = str(imaging_destination).split(
                    imaging_destination.parents[1].name
                )[-1]
                # json_key = current_imaging_folder.name + ' Imaging'
                # utils.append_json(path=fly_dirs_dict_path, key=json_key, value=current_fly_dir_dict)
                fly_dirs_dict[
                    current_import_imaging_folder.name + " Imaging"
                ] = current_fly_dir_dict
                ######################################################################
                print(
                    f"anat:{current_dataset_folder}"
                )  # IMPORTANT - FOR COMMUNICATING WITH MAIN
                ######################################################################
            elif "func" in current_import_imaging_folder.name:
                # Make imaging folder and copy
                # imaging_destination = os.path.join(expt_folder, 'imaging')
                imaging_destination = pathlib.Path(current_dataset_folder, "imaging")
                # os.mkdir(imaging_destination)
                imaging_destination.mkdir(parents=True, exist_ok=True)
                copy_bruker_data(
                    current_import_imaging_folder, imaging_destination, "func", printlog
                )
                # Update fly_dirs_dict
                current_fly_dir_dict = str(imaging_destination).split(
                    imaging_destination.parents[1].name
                )[-1]
                # json_key = current_imaging_folder.name + ' Imaging'
                # utils.append_json(path=fly_dirs_dict_path, key=json_key, value=current_fly_dir_dict)
                fly_dirs_dict[
                    current_import_imaging_folder.name + " Imaging"
                ] = current_fly_dir_dict
                # Copy fictrac data based on timestamps
                try:
                    fly_dirs_dict = copy_fictrac(
                        current_dataset_folder,
                        printlog,
                        user,
                        current_import_imaging_folder,
                        fly_dirs_dict,
                    )
                    # printlog('Fictrac data copied')
                except Exception as e:
                    printlog("Could not copy fictrac data because of error:")
                    printlog(str(e))
                    printlog(traceback.format_exc())
                # Copy visual data based on timestamps, and create visual.json
                try:
                    copy_visual(current_dataset_folder, printlog)
                except Exception as e:
                    printlog("Could not copy visual data because of error:")
                    printlog(str(e))

                ######################################################################
                # print(f"func:{expt_folder}")  # IMPORTANT - FOR COMMUNICATING WITH MAIN
                ######################################################################
                # REMOVED TRIGGERING

            else:
                printlog(
                    "Invalid directory in fly folder (skipping): {}".format(
                        current_import_imaging_folder.name
                    )
                )

        # Copy fly.json file
        else:
            current_import_file = current_import_file_or_folder
            if current_import_file_or_folder.name == "fly.json":
                target_path = pathlib.Path(dataset_dir, current_import_file.name)
                shutil.copyfile(current_import_file, target_path)
            else:
                printlog(
                    "Invalid file in fly folder (skipping): {}".format(
                        current_import_file.name
                    )
                )

    return fly_dirs_dict


def copy_bruker_data(source, destination, folder_type, printlog, fly_dirs_dict=None):
    # Do not update destination - download all files into that destination
    # for item in os.listdir(source):
    for source_path in source.iterdir():
        # Check if item is a directory
        if source_path.is_dir():
            # Do not update destination - download all files into that destination
            copy_bruker_data(source_path, destination, folder_type, printlog)
            # In my case this leads to /oak/stanford/groups/trc/data/David/Bruker/imports/20231201/fly2/func1
            # The code then calls itself with this path which then goes one folder deeper

        # If the item is a file
        else:
            target_path = None
            # Don't copy these files
            if "SingleImage" in source_path.name:
                continue  # skip rest of the 'else' term
            # elif '.nii' in source_path.name and folder_type == 'func': # Shouldn't be necessary with '_s' check!
            #    continue  # do not copy!! Else we'll copy all the split nii files as well.
            #    # This is an artifact of splitting the nii file on Brukerbridge and might not be
            #    # relevant in the future/for other users!
            # each source path file can only be a single file - why if..if instead of if..elif?
            ### Change file names and filter various files
            # This is for split files from brukerbridge
            elif "concat.nii" in source_path.name and folder_type == "anat": # <this should be 'func'
                target_name = (
                    "channel_" + source_path.name.split("ch")[1].split("_")[0] + ".nii"
                )
                target_path = pathlib.Path(destination, target_name)
            # This is for non-split files from Brukerbridge
            elif (
                ".nii" in source_path.name
                and "_s" not in source_path.name
                and folder_type == "anat" # <this should be 'func'
            ):
                target_name = (
                    "channel_" + source_path.name.split("channel")[1].split("_")[1]
                )  # this already gets us the '.nii'!
                print("target name" + repr(target_name))
                target_path = pathlib.Path(destination, target_name)
            # Rename anatomy file to anatomy_channel_x.nii
            elif ".nii" in source_path.name and folder_type == "anat":
                target_name = (
                    "channel_" + source_path.name.split("channel")[1].split("_")[1]
                )
                if ".nii" not in target_name:
                    target_name += "nii"  # Data I got from Yandan had double .nii!
                target_path = pathlib.Path(destination, target_name)
            # Special copy for photodiode since it goes in visual folder
            # To be tested once I have such data!!
            elif ".csv" in source_path.name:
                source_name = "photodiode.csv"
                visual_folder_path = pathlib.Path(destination, "visual")
                visual_folder_path.mkdir(exist_ok=True)
                target_path = pathlib.Path(visual_folder_path, source_name)
            # Special copy for visprotocol metadata since it goes in visual folder
            # To be tested once I have such data!!
            elif ".hdf5" in source_path.name:
                # Create folder 'visual'
                visual_folder_path = pathlib.Path(destination.name, "visual")
                visual_folder_path.mkdir(exist_ok=True)
                target_path = pathlib.Path(visual_folder_path, source_path.name)
            # Rename to recording_metadata.xml if appropriate
            elif ".xml" in source_path.name and "Voltage" not in source_path.name:
                target_name = "recording_metadata.xml"
                target_path = pathlib.Path(destination, target_name)
                if folder_type == "func":
                    copy_file_func(source_path, target_path, printlog)
                    # Create json file
                    create_imaging_json(target_path, printlog)
                    continue
            # Rename to anatomy.xml if appropriate
            # if '.xml' in source_path.name and folder_type == 'anat' and \
            #        'Voltage' not in source_path.name:
            #    target_name = 'anatomy.xml'
            #    target_path = pathlib.Path(destination, target_name)

            # Rename to functional.xml if appropriate, copy immediately, then make scan.json
            # if ('.xml' in source_path.name and folder_type == 'func' and
            #        'Voltage' not in source_path.name):
            #    # TOdo: rename to something like 'microscope.xml'. Similar to ch, keep filenames consistent!
            #    target_path = pathlib.Path(destination, 'functional.xml')
            #    #target_item = os.path.join(destination, item)
            #    copy_file(source_path, target_path, printlog)
            #    # Create json file
            #    create_imaging_json(target_path, printlog)
            #    continue
            elif ".xml" in source_path.name and "VoltageOutput" in source_path.name:
                target_path = pathlib.Path(destination, "voltage_output.xml")

            if target_path is not None:
                # Actually copy the file
                copy_file_func(source_path, target_path, printlog)


def copy_file_func(source, target, printlog):
    # printlog('Transfering file {}'.format(target))
    # to_print = ('/').join(target.split('/')[-4:])
    # print('source: ' + str(source))
    # print('target: ' + str(target))
    to_print = str(source.name + " to " + target.name)
    # width = 120
    printlog(f"Transfering file{to_print:.>{WIDTH - 16}}")
    ##sys.stdout.flush()
    shutil.copyfile(source, target)


def copy_visual(destination_region, printlog):
    print("copy_visual NOT IMPLEMENTED YET")
    """width = 120
    printlog(F"Copying visual stimulus data{'':.^{width - 28}}")
    visual_folder = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/imports/visual'
    visual_destination = os.path.join(destination_region, 'visual')

    # Find time of experiment based on functional.xml
    true_ymd, true_total_seconds = get_expt_time(os.path.join(destination_region, 'imaging'))

    # Find visual folder that has the closest datetime
    # First find all folders with correct date, and about the correct time
    folders = []
    for folder in os.listdir(visual_folder):
        test_ymd = folder.split('-')[1]
        test_time = folder.split('-')[2]
        test_hour = test_time[0:2]
        test_minute = test_time[2:4]
        test_second = test_time[4:6]
        test_total_seconds = int(test_hour) * 60 * 60 + \
                             int(test_minute) * 60 + \
                             int(test_second)

        if test_ymd == true_ymd:
            time_difference = np.abs(true_total_seconds - test_total_seconds)
            if time_difference < 3 * 60:
                folders.append([folder, test_total_seconds])
                printlog('Found reasonable visual folder: {}'.format(folder))

    # if more than 1 folder, use the oldest folder
    if len(folders) == 1:
        correct_folder = folders[0]
    # if no matching folder,
    elif len(folders) == 0:
        printlog(F"{'No matching visual folders found; continuing without visual data':.<{width}}")
        return
    else:
        printlog('Found more than 1 visual stimulus folder within 3min of expt. Picking oldest.')
        correct_folder = folders[0]  # set default to first folder
        for folder in folders:
            # look at test_total_seconds entry. If larger, call this the correct folder.
            if folder[-1] > correct_folder[-1]:
                correct_folder = folder

    # now that we have the correct folder, copy it's contents
    printlog('Found correct visual stimulus folder: {}'.format(correct_folder[0]))
    try:
        os.mkdir(visual_destination)
    except:
        pass
        ##print('{} already exists'.format(visual_destination))
    source_folder = os.path.join(visual_folder, correct_folder[0])
    printlog('Copying from: {}'.format(source_folder))
    for file in os.listdir(source_folder):
        target_path = os.path.join(visual_destination, file)
        source_path = os.path.join(source_folder, file)
        ##print('Transfering from {} to {}'.format(source_path, target_path))
        ##sys.stdout.flush()
        shutil.copyfile(source_path, target_path)

    # Create visual.json metadata
    # Try block to prevent quiting if visual stimuli timing is wonky (likely went too long)
    try:
        unique_stimuli = brainsss.get_stimuli(visual_destination)
    except:
        unique_stimuli = 'brainsss.get_stimuli failed'
    with open(os.path.join(visual_destination, 'visual.json'), 'w') as f:
        json.dump(unique_stimuli, f, indent=4)"""


def copy_fictrac(destination_region, printlog, user, source_fly, fly_dirs_dict):
    # The target file will be called 'fictrac_behavior_data.dat' because it makes
    # handling files much easier in the snakefile.
    # Make fictrac folder
    fictrac_destination = pathlib.Path(destination_region, "fictrac")
    fictrac_destination.mkdir(exist_ok=True)
    # Different users have different rule on what to do with the data
    if user == "brezovec":
        user = "luke"
    if user == "yandanw":
        user = "luke"
    if user == "ilanazs":
        user = "luke"
    if user == "dtadres":
        # TODO!!!
        fictrac_folder = pathlib.Path(
            "/oak/stanford/groups/trc/data/David/Bruker/Fictrac"
        )
        # when doing post-hoc fictrac, Bella's code where one compare the recording
        # timestamps of imaging and fictrac doesn't work anymore.
        # I instead use a deterministic file structure:
        # for example for fly 20231201\fly2\func1 imaging data, fictrac data must
        # be in the folder 20231201_fly2_func1. There must only be a single dat file in that folder.
        source_path = pathlib.Path(
            fictrac_folder,
            source_fly.parts[-3]
            + "_"
            + source_fly.parts[-2]
            + "_"
            + source_fly.parts[-1],
        )
        for current_file in source_path.iterdir():
            if "dat" in current_file.name:
                # width = 120
                # source_path = os.path.join(source_path, file)
                dat_path = current_file
                # target_path = pathlib.Path(fictrac_destination, current_file.name)
                target_path = pathlib.Path(
                    fictrac_destination, "fictrac_behavior_data.dat"
                )
                to_print = str(target_path)
                printlog(f"Transfering file{to_print:.>{WIDTH - 16}}")

                # put fictrac file path in into fly_dirs_dict
                current_fly_dir_dict = str(target_path).split(
                    fictrac_destination.parents[1].name
                )[-1]
                fly_dirs_dict[
                    destination_region.name + " Fictrac "
                ] = current_fly_dir_dict
    else:
        # fictrac_folder = os.path.join("/oak/stanford/groups/trc/data/fictrac", user)
        fictrac_folder = pathlib.Path("/oak/stanford/groups/trc/data/fictrac", user)

        # Find time of experiment based on functional.xml
        # true_ymd, true_total_seconds = get_expt_time(os.path.join(destination_region, 'imaging'))
        true_ymd, true_total_seconds = get_expt_time(
            pathlib.Path(destination_region, "imaging")
        )

        # printlog(f'true_ymd: {true_ymd}; true_total_seconds: {true_total_seconds}')

        # Find .dat file of 1) correct-ish time, 2) correct-ish size
        correct_date_and_size = []
        time_differences = []
        # for file in os.listdir(fictrac_folder):
        for file in fictrac_folder.iterdir():
            file = str(file)  # To be changed in the future
            # but I'm currently to lazy to change everything to
            # pathlib object below.

            # must be .dat file
            if ".dat" not in file:
                continue

            # Get datetime from file name
            datetime = file.split("-")[1][:-4]
            test_ymd = datetime.split("_")[0]
            test_time = datetime.split("_")[1]
            test_hour = test_time[0:2]
            test_minute = test_time[2:4]
            test_second = test_time[4:6]
            test_total_seconds = (
                int(test_hour) * 60 * 60 + int(test_minute) * 60 + int(test_second)
            )

            # Year/month/day must be exact
            if true_ymd != test_ymd:
                continue
            # printlog('Found file from same day: {}'.format(file))

            # Must be correct size
            # fp = os.path.join(fictrac_folder, file)
            fp = pathlib.Path(fictrac_folder, file)
            file_size = fp.stat().st_size
            # file_size = os.path.getsize(fp)
            if (
                file_size < 1000000
            ):  # changed to 1MB to accomidate 1 min long recordings. #30000000: #30MB
                # width = 120
                # printlog(F"Found correct .dat file{file:.>{width-23}}")
                # datetime_correct = datetime
                # break
                continue

            # get time difference from expt
            time_difference = np.abs(true_total_seconds - test_total_seconds)
            # Time must be within 10min
            if time_difference > 10 * 60:
                continue

            # if correct date and size append to list of potential file
            correct_date_and_size.append(file)
            time_differences.append(time_difference)

        # now that we have all potential files, pick the one with closest time
        # except clause will happen if empty list
        try:
            datetime_correct = correct_date_and_size[np.argmin(time_differences)]
        except:
            # width = 120
            printlog(
                f"{'   No fictrac data found --- continuing without fictrac data   ':*^{WIDTH}}"
            )
            return

        # Collect all fictrac files with correct datetime
        correct_time_files = [
            file for file in fictrac_folder.iterdir() if datetime_correct in file.name
        ]
        # correct_time_files = [file for file in os.listdir(fictrac_folder) if datetime_correct in file]

        # correct_time_files = []
        # for file in os.listdir(fictrac_folder):
        #     if datetime_correct in file:
        #         correct_time_files.append(file)

        # printlog('Found these files with correct times: {}'.format(correct_time_files))
        ##sys.stdout.flush()

        # Now transfer these 4 files to the fly
        fictrac_folder.mkdir()
        # os.mkdir(fictrac_destination)
        for file in correct_time_files:
            # width = 120
            target_path = pathlib.Path(fictrac_folder, file)
            source_path = pathlib.Path(fictrac_folder, file)
            # target_path = os.path.join(fictrac_destination, file)
            # source_path = os.path.join(fictrac_folder, file)
            # to_print = ('/').join(target_path.split('/')[-4:])
            to_print = str(target_path)
            printlog(f"Transfering file{to_print:.>{WIDTH - 16}}")
            # printlog('Transfering {}'.format(target_path))
            ##sys.stdout.flush()

    shutil.copyfile(dat_path, target_path)

    ### Create empty xml file.
    # Update this later
    root = etree.Element("root")
    fictrac = objectify.Element("fictrac")
    root.append(fictrac)
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)
    tree = etree.ElementTree(fictrac)
    # with open(os.path.join(fictrac_destination, 'fictrac.xml'), 'wb') as file:
    with open(pathlib.Path(fictrac_destination, "fictrac.xml"), "wb") as file:
        tree.write(file, pretty_print=True)

    return fly_dirs_dict


def create_imaging_json(xml_source_file, printlog):
    # Make empty dict
    source_data = {}

    # Get datetime
    try:
        datetime_str, _, _ = get_datetime_from_xml(xml_source_file)
    except:
        printlog("No xml or cannot read.")
        ##sys.stdout.flush()
        return
    date = datetime_str.split("-")[0]
    time = datetime_str.split("-")[1]
    source_data["date"] = str(date)
    source_data["time"] = str(time)

    # Get rest of data
    tree = objectify.parse(xml_source_file)
    source = tree.getroot()
    statevalues = source.findall("PVStateShard")[0].findall("PVStateValue")
    for statevalue in statevalues:
        key = statevalue.get("key")
        if key == "micronsPerPixel":
            indices = statevalue.findall("IndexedValue")
            for index in indices:
                axis = index.get("index")
                if axis == "XAxis":
                    source_data["x_voxel_size"] = float(index.get("value"))
                elif axis == "YAxis":
                    source_data["y_voxel_size"] = float(index.get("value"))
                elif axis == "ZAxis":
                    source_data["z_voxel_size"] = float(index.get("value"))
        if key == "laserPower":
            # I think this is the maximum power if set to vary by z depth - WRONG
            indices = statevalue.findall("IndexedValue")
            laser_power_overall = int(float(indices[0].get("value")))
            source_data["laser_power"] = laser_power_overall
        if key == "pmtGain":
            indices = statevalue.findall("IndexedValue")
            for index in indices:
                index_num = index.get("index")
                if index_num == "0":
                    source_data["PMT_red"] = int(float(index.get("value")))
                if index_num == "1":
                    source_data["PMT_green"] = int(float(index.get("value")))
        if key == "pixelsPerLine":
            source_data["x_dim"] = int(float(statevalue.get("value")))
        if key == "linesPerFrame":
            source_data["y_dim"] = int(float(statevalue.get("value")))
    sequence = source.findall("Sequence")[0]
    last_frame = sequence.findall("Frame")[-1]
    source_data["z_dim"] = int(last_frame.get("index"))

    # Need this try block since sometimes first 1 or 2 frames don't have laser info...
    # try:
    #     # Get laser power of first and last frames
    #     last_frame = sequence.findall('Frame')[-1]
    #     source_data['laser_power'] = int(last_frame.findall('PVStateShard')[0].findall('PVStateValue')[1].findall('IndexedValue')[0].get('value'))
    #     #first_frame = sequence.findall('Frame')[0]
    #     #source_data['laser_power_min'] = int(first_frame.findall('PVStateShard')[0].findall('PVStateValue')[1].findall('IndexedValue')[0].get('value'))
    # except:
    #     source_data['laser_power_min'] = laser_power_overall
    #     source_data['laser_power_max'] = laser_power_overall
    #     #printlog('Used overall laser power.')
    #     # try:
    #     #     first_frame = sequence.findall('Frame')[2]
    #     #     source_data['laser_power_min'] = int(first_frame.findall('PVStateShard')[0].findall('PVStateValue')[1].findall('IndexedValue')[0].get('value'))
    #     #     printlog('Took min laser data from frame 3, not frame 1, due to bruker metadata error.')
    #     # # Apparently sometimes the metadata will only include the
    #     # # laser value at the very beginning
    #     # except:
    #     #     source_data['laser_power_min'] = laser_power_overall
    #     #     source_data['laser_power_max'] = laser_power_overall
    #     #     printlog('Used overall laser power.')

    # Save data
    # with open(os.path.join(os.path.split(xml_source_file)[0], 'scan.json'), 'w') as f:
    with open(pathlib.Path(xml_source_file.parent, "scan.json"), "w") as f:
        json.dump(source_data, f, indent=4)


def get_expt_time(directory):
    """Finds time of experiment based on functional.xml"""
    xml_file = pathlib.Path(directory, "functional.xml")
    # xml_file = os.path.join(directory, 'functional.xml')
    _, _, datetime_dict = get_datetime_from_xml(xml_file)
    true_ymd = datetime_dict["year"] + datetime_dict["month"] + datetime_dict["day"]
    true_total_seconds = (
        int(datetime_dict["hour"]) * 60 * 60
        + int(datetime_dict["minute"]) * 60
        + int(datetime_dict["second"])
    )

    ##print('dict: {}'.format(datetime_dict))
    ##print('true_ymd: {}'.format(true_ymd))
    ##print('true_total_seconds: {}'.format(true_total_seconds))
    ##sys.stdout.flush()
    return true_ymd, true_total_seconds


"""def get_fly_time(fly_folder):
    # need to read all xml files and pick oldest time
    # find all xml files
    xml_files = []
    xml_files = get_xml_files(fly_folder, xml_files)

    ##print('found xml files: {}'.format(xml_files))
    ##sys.stdout.flush()
    datetimes_str = []
    datetimes_int = []
    for xml_file in xml_files:
        datetime_str, datetime_int, _ = get_datetime_from_xml(xml_file)
        datetimes_str.append(datetime_str)
        datetimes_int.append(datetime_int)

    # Now pick the oldest datetime
    datetimes_int = np.asarray(datetimes_int)
    ##print('Found datetimes: {}'.format(datetimes_str))
    ##sys.stdout.flush()
    index_min = np.argmin(datetimes_int)
    datetime = datetimes_str[index_min]
    ##print('Found oldest datetime: {}'.format(datetime))
    ##sys.stdout.flush()
    return datetime"""

"""def get_xml_files(fly_folder, xml_files):
    # Look at items in fly folder
    for item in os.listdir(fly_folder):
        full_path = os.path.join(fly_folder, item)
        if os.path.isdir(full_path):
            xml_files = get_xml_files(full_path, xml_files)
        else:
            if '.xml' in item and \
            '_Cycle' not in item and \
            'fly.xml' not in item and \
            'scan.xml' not in item and \
            'expt.xml' not in item:
                xml_files.append(full_path)
                ##print('Found xml file: {}'.format(full_path))
                ##sys.stdout.flush()
    return xml_files"""


def get_datetime_from_xml(xml_file):
    ##print('Getting datetime from {}'.format(xml_file))
    ##sys.stdout.flush()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    datetime = root.get("date")
    # will look like "4/2/2019 4:16:03 PM" to start

    # Get dates
    date = datetime.split(" ")[0]
    month = date.split("/")[0]
    day = date.split("/")[1]
    year = date.split("/")[2]

    # Get times
    time = datetime.split(" ")[1]
    hour = time.split(":")[0]
    minute = time.split(":")[1]
    second = time.split(":")[2]

    # Convert from 12 to 24 hour time
    am_pm = datetime.split(" ")[-1]
    if am_pm == "AM" and hour == "12":
        hour = str(00)
    elif am_pm == "AM":
        pass
    elif am_pm == "PM" and hour == "12":
        pass
    else:
        hour = str(int(hour) + 12)

    # Add zeros if needed
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    if len(hour) == 1:
        hour = "0" + hour

    # Combine
    datetime_str = year + month + day + "-" + hour + minute + second
    datetime_int = int(year + month + day + hour + minute + second)
    datetime_dict = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
    }

    return datetime_str, datetime_int, datetime_dict


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def load_xml(file):
    tree = objectify.parse(file)
    root = tree.getroot()
    return root


def add_fly_to_xlsx(fly_folder, printlog):
    printlog("Adding fly to master_2P excel log")

    ### TRY TO LOAD ELSX ###
    try:
        xlsx_path = pathlib.Path(fly_folder.parent, "master_2P.xlsx")
        wb = load_workbook(filename=xlsx_path, read_only=False)
        ws = wb.active
        printlog("Successfully opened master_2P log")
    except Exception as e:
        printlog(
            "FYI you have no excel metadata sheet found, so unable to append metadata for this fly."
        )
        printlog(traceback.format_exc())
        return

    ### TRY TO LOAD FLY METADATA ###
    try:
        fly_file = pathlib.Path(fly_folder, "fly.json")
        # fly_file = os.path.join(fly_folder, 'fly.json')
        fly_data = load_json(fly_file)
        printlog("Successfully loaded fly.json")
    except:
        printlog(
            "FYI no *fly.json* found; this will not be logged in your excel sheet."
        )
        fly_data = {}
        fly_data["circadian_on"] = None
        fly_data["circadian_off"] = None
        fly_data["gender"] = None
        fly_data["age"] = None
        fly_data["temp"] = None
        fly_data["notes"] = None
        fly_data["date"] = None
        fly_data["genotype"] = None

    # Write in master xlsx only if we have a func folder. Ignore anatomical data!
    expt_folders = [
        pathlib.Path(fly_folder, x) for x in fly_folder.iterdir() if "func" in x.name
    ]
    # brainsss.sort_nicely(expt_folders)
    expt_folders = natsort.natsorted(expt_folders)
    for expt_folder in expt_folders:
        print("expt_folder" + repr(expt_folder))

        ### TRY TO LOAD EXPT METADATA ###
        try:
            # expt_file = os.path.join(expt_folder, 'expt.json')
            expt_file = pathlib.Path(expt_folders, "expt.json")
            expt_data = load_json(expt_file)
            printlog("Sucessfully loaded expt.json")
        except:
            printlog(
                "FYI no *expt.json* found; this will not be logged in your excel sheet."
            )
            expt_data = {}
            expt_data["brain_area"] = None
            expt_data["notes"] = None
            expt_data["time"] = None

        ### TRY TO LOAD SCAN DATA ###
        try:
            scan_file = pathlib.Path(expt_folder, "imaging", "scan.json")
            # scan_file = os.path.join(expt_folder, 'imaging', 'scan.json')
            scan_data = load_json(scan_file)
            scan_data["x_voxel_size"] = "{:.1f}".format(scan_data["x_voxel_size"])
            scan_data["y_voxel_size"] = "{:.1f}".format(scan_data["y_voxel_size"])
            scan_data["z_voxel_size"] = "{:.1f}".format(scan_data["z_voxel_size"])
            printlog("Sucessfully loaded scan.json")
        except:
            printlog(
                "FYI no *scan.json* found; this will not be logged in your excel sheet."
            )
            scan_data = {}
            scan_data["laser_power"] = None
            scan_data["PMT_green"] = None
            scan_data["PMT_red"] = None
            scan_data["x_dim"] = None
            scan_data["y_dim"] = None
            scan_data["z_dim"] = None
            scan_data["x_voxel_size"] = None
            scan_data["y_voxel_size"] = None
            scan_data["z_voxel_size"] = None

        visual_file = pathlib.Path(expt_folder, "visual", "visual.json")
        # visual_file = os.path.join(expt_folder, 'visual', 'visual.json')
        try:
            visual_data = load_json(visual_file)
            visual_input = visual_data[0]["name"] + " ({})".format(len(visual_data))
        except:
            visual_input = None

        # Get fly_id
        # fly_folder = expt_folder.parent
        # fly_folder = os.path.split(os.path.split(expt_folder)[0])[-1]
        fly_id = fly_folder.name.split("_")[-1]
        printlog(f"Got fly ID as {fly_id}")

        # Get expt_id
        expt_id = expt_folder.name  # probably 'func1' etc.
        printlog(f"Got expt ID as {expt_id}")
        # expt_id = 'NA' # Not sure what this is, NA for now

        # Append the new row
        # new_row = []
        new_row = [
            int(fly_id),
            str(expt_id),
            fly_data["date"],
            expt_data["brain_area"],
            fly_data["genotype"],
            visual_input,
            None,
            fly_data["notes"],
            expt_data["notes"],
            expt_data["time"],
            fly_data["circadian_on"],
            fly_data["circadian_off"],
            fly_data["gender"],
            fly_data["age"],
            fly_data["temp"],
            scan_data["laser_power"],
            scan_data["PMT_green"],
            scan_data["PMT_red"],
            scan_data["x_dim"],
            scan_data["y_dim"],
            scan_data["z_dim"],
            scan_data["x_voxel_size"],
            scan_data["y_voxel_size"],
            scan_data["z_voxel_size"],
        ]

        ws.append(new_row)
        printlog(f"Appended {fly_id} {expt_id}")

    # Save the file
    wb.save(xlsx_path)
    printlog("master_2P successfully updated")
