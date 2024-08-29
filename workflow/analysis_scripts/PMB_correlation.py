

"""
Problem: If we are looking for PMB and we have a very active fly that turns 2 times per second while
imaging with something like GCaMP7b the off kinetics of the fluorescent sensor can lead to a hard to
decipher mixing of signal.

To tease out the ramping activity it's likely necessary to:
1) Define saccadic turns
2) identify where we have no turns for x seconds before a given turn
3) only use those timepoints
"""

import pathlib
import sys
import nibabel as nib
import numpy as np
import scipy
from scipy import signal

parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
# Console: sys.path.insert(0, 'C:\\Users\\David\\snake_brainsss\\workflow')

from brainsss import fictrac_utils
from brainsss import utils
from scripts import analysis_functions

# The original brainsss usually saved everything in float32 but some
# calculations were done with float64 (e.g. np.mean w/o defining dtype).
# To be more explicit, I define here two global variables.
# Dtype is what was mostly called when saving and loading data
DTYPE = np.float32

#dataset_path = pathlib.Path('F:\\FS152_x_FS200\\fly_002\\func0')
#behavior='dRotLabZpos'
#save_path = pathlib.Path('F:\\FS152_x_FS200\\fly_002\\func0\\turn_corr', behavior + '_corr.nii')
def PMB_correlation(
        fly_log_folder,
        fictrac_path,
        fictrac_fps,
        metadata_path,
        moco_zscore_highpass_path,
        save_path,

):

    #fps = 100
    resolution = 10  # desired resolution in ms for fictrac. Should really just be 1/fps I think...
    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_log_folder, function_name="correlation")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    # utils.print_function_start(logfile, "correlation")

    printlog('Will use ' + repr(moco_zscore_highpass_path) + ' for input.')
    printlog('Will save as ' + repr(save_path))
    # Then extract the desired behavior. Since there's only one output per run, the list
    # has length 1.
    behavior = save_path[0].name.split("corr_")[-1].split("_PMB.nii")[0]
    time_before_turn = int(save_path[0].name.split("PMB_")[-1].split('s_no_turn.nii')[0])

    turn_indeces = analysis_functions.extract_turn_bouts(moco_zscore_highpass_path.parent,
                                                         minimal_time_between_turns=0.5, turn_thresh=200)
    cleaned_up_turns = {}
    cleaned_up_turns['L']=analysis_functions.extract_isolated_turns(direction=turn_indeces['L'],
                                                 other_direction=turn_indeces['R'],
                                                 time_before_turn=time_before_turn)

    cleaned_up_turns['R'] = analysis_functions.extract_isolated_turns(direction=turn_indeces['R'],
                                                   other_direction=turn_indeces['L'],
                                                   time_before_turn=time_before_turn)

    # only take the time before and during turn for correlation...
    #neural_timestamps = utils.load_timestamps(pathlib.Path(dataset_path, 'imaging\\recording_metadata.xml'))
    neural_timestamps = utils.load_timestamps(metadata_path)
    # search for relevant timestamps - treat volume as one timepoint (which is incorrect of course but might be ok here)
    neural_timestamps_seconds = neural_timestamps/1000

    indeces_to_take = []


    indeces_to_take= analysis_functions.get_time_index_neural_data(cleaned_up_turns['R'],
                                                                   neural_timestamps_seconds,
                                                                   indeces_to_take,
                                                                   time_before_turn)

    indeces_to_take= analysis_functions.get_time_index_neural_data(cleaned_up_turns['L'],
                                                                   neural_timestamps_seconds,
                                                                   indeces_to_take,
                                                                   time_before_turn)
    # Sort index
    indeces_to_take = np.array(sorted(indeces_to_take))
    # Make sure each index only exists once
    indeces_to_take = np.unique(indeces_to_take)

    #neural_data = nib.load(pathlib.Path(dataset_path, 'channel_2_moco_zscore_highpass.nii'))
    neural_data = nib.load(moco_zscore_highpass_path)

    x_dim, y_dim, z_dim, t_dim = neural_data.shape
    neural_data = np.asarray(neural_data.dataobj)

    #fictrac_raw = fictrac_utils.load_fictrac(pathlib.Path(dataset_path, 'stimpack\\loco\\fictrac_behavior_data.dat'))
    fictrac_raw = fictrac_utils.load_fictrac(fictrac_path)

    expt_len = fictrac_raw.shape[0] / fictrac_fps * 1000 # experiment length in ms

    # We will have a list of functional channels here but
    # all from the same experiment, so all will have the same 'recording_metadata.xml' data
    # neural_timestamps = utils.load_timestamps(pathlib.Path(dataset_path, 'imaging\\recording_metadata.xml'))
    neural_timestamps = utils.load_timestamps(metadata_path)
    corr_brain = np.zeros((x_dim, y_dim, z_dim))

    fictrac_interp = fictrac_utils.smooth_and_interp_fictrac(
        fictrac_raw, fictrac_fps, resolution, expt_len, behavior, timestamps=neural_timestamps
        )
    # do correlation only with
    for z in range(z_dim):

        '''
        ### interpolate fictrac to match the timestamps of this slice
        print(F"{z}")
        fictrac_interp = fictrac_utils.smooth_and_interp_fictrac(fictrac_raw,
                                                                 fps,
                                                                 resolution,
                                                                 expt_len,
                                                                 behavior,
                                                                 timestamps=neural_timestamps,
                                                                 z=z)
    
        for x in range(x_dim):
            for y in range(y_dim):
                # nan to num should be taken care of in zscore, but checking here for some already processed brains
                #if np.any(np.isnan(brain[i, j, z, :])):
                    #printlog(F'warning found nan at x = {i}; y = {j}; z = {z}')
                #    corr_brain[i, j, z] = 0
                if len(np.unique(neural_data[x, y, z, :])) == 1:
                    #     if np.unique(brain[i,j,z,:]) != 0:
                    print(F'warning found non-zero constant value at x = {x}; y = {y}; z = {z}')
                    corr_brain[x, y, z] = 0
                else:
                    # idx_to_use can be used to select a subset of timepoints
                    corr_brain[x, y, z] = scipy.stats.pearsonr(fictrac_interp[idx_to_use], neural_data[x, y, z, idx_to_use])[0]
                    '''

        # Vectorized correlation - see 'dev/pearson_correlation.py' for development and timing info
        # The formula is:
        # r = (sum(x-m_x)*(y-m_y) / sqrt(sum((x-m_x)^2)*sum((y-m_y)^2)
        brain_mean = neural_data[:, :, z, indeces_to_take].mean(axis=-1)  # , dtype=np.float64)

        # When I plot the data plt.plot(fictrac_interp) and plt.plot(fictrac_interp.astype(np.float32) I
        # can't see a difference. Should be ok to do this as float.
        # This is advantagous because we have to do the dot product with the brain. np.dot will
        # default to higher precision leading to making a copy of brain data which costs a lot of memory
        # Unfortunately fictrac_interp comes in [time, z] as opposed to brain that comes in [x,y,z,t]
        fictrac_mean = fictrac_interp[indeces_to_take, z].mean(dtype=DTYPE)

        # >> Typical values for z scored brain seem to be between -25 and + 25.
        # It shouldn't be necessary to cast as float64. This then allows us
        # to do in-place operation!
        brain_mean_m = neural_data[:, :, z, indeces_to_take] - brain_mean[:, :, None]

        # fictrac data is small, so working with float64 shouldn't cost much memory!
        # Correction - if we cast as float64 and do dot product with brain, we'll copy the
        # brain to float64 array, balloning the memory requirement
        fictrac_mean_m = fictrac_interp[indeces_to_take, z].astype(DTYPE) - fictrac_mean

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
            fictrac_mean_m = fictrac_mean_m[0:fictrac_mean_m.shape[0] - 1]
            # Calculate correlation

            corr_brain[:, :, z] = np.dot(
                brain_mean_m / normbrain[:, :, None], fictrac_mean_m / normfictrac
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare Nifti file for saving
    aff = np.eye(4)
    object_to_save = nib.Nifti1Image(corr_brain, aff)
    # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
    object_to_save.to_filename(save_path)

'''       
def PMB_correlation(
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
        )'''