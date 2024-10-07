"""
Sometimes correlation does not work well to tease out signal.

Instead, we can go the other way: extract the brain around a saccade and
just take the mean.
"""
import numpy as np
import nibabel as nib
import pathlib
import scipy
import time

import sys
parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
# This just imports '*.py' files from the folder 'brainsss'.
from brainsss import utils
from brainsss import fictrac_utils

def find_saccades(fictrac_path):
    """
    This is essentially copy-paste from Bella's scripts.

    Might not work well for different recordings!
    """

    behavior = 'dRotLabZ'
    minimal_time_between_turns = 0.5
    fictrac_fps = 100
    turn_thresh = 200

    fictrac_raw = fictrac_utils.load_fictrac(fictrac_path)
    fictrac_smoothed = scipy.signal.savgol_filter(np.asarray(fictrac_raw[behavior]), 25, 3)
    fictrac_smoothed = fictrac_smoothed * 180 / np.pi * fictrac_fps  # now in deg/sec
    fictrac_timestamps = np.arange(0, fictrac_smoothed.shape[0] / fictrac_fps,
                                   1 / fictrac_fps)  # This is correct assuming we really
    # always get exactly 100 fps!

    minimal_time_between_turns = minimal_time_between_turns * fictrac_fps  # convert from seconds to fps!

    ###########################
    ### Identify turn bouts ###
    ###########################

    peaks = {'L': [], 'R': []}
    heights = {'L': [], 'R': []}

    for turn, scalar in zip(['L', 'R'], [1, -1]):
        found_peaks = scipy.signal.find_peaks(fictrac_smoothed * scalar, height=turn_thresh,
                                              distance=minimal_time_between_turns)
        pks = found_peaks[0]
        pk_height = found_peaks[1]['peak_heights']

        # convert index to time in seconds!
        peaks[turn] = fictrac_timestamps[pks]
        heights[turn] = pk_height

    return (peaks)


def extract_saccade_triggered_neural_activity(imaging_data,
                                              neural_timestamps,
                                              turns,
                                              turn_side='L'):
    """

    """

    # While it seems that Bruker is not super consistent with timestamps, the first
    # volume should be the slowest. Hence, this should be conservative and always work.
    first_volume_time = (neural_timestamps[1,0] - neural_timestamps[0,0])/1e3

    time_before_saccade = 0.5  # seconds
    time_after_saccade = 0.5  # seconds

    # Get brain size
    x_dim = imaging_data.shape[0]
    y_dim = imaging_data.shape[1]
    z_dim = imaging_data.shape[2]

    # How many stacks can we maximally collect in the total time?
    # Should be this: i.e. 1s/0.53s/V = 1.8V so.
    max_number_of_stacks_per_saccade = int(1 + np.ceil((time_before_saccade+time_after_saccade)/first_volume_time))
    print('first_volume_time: ' + repr(first_volume_time))
    print('max_number_of_stacks_per_saccade: ' + repr(max_number_of_stacks_per_saccade))

    brain_activity_turns = np.zeros((x_dim, y_dim, z_dim, max_number_of_stacks_per_saccade, len(turns[turn_side])),
                                    dtype=np.float32)
    brain_activity_turns.fill(np.nan)
    # Need to flatten the neural timestamps array as searchsorted only works on 1D array
    flat_neural_timestamps = neural_timestamps.flatten()

    brain_activity_no_turns = imaging_data.copy() # Expensive...lots of RAM!


    for turn_counter, current_turn in enumerate(turns[turn_side]):
        print('turn counter: ' + repr(turn_counter))
        time_before_current_saccade = current_turn - time_before_saccade
        time_before_current_saccade *= 1000  # turn to ms which is what neural_timestamps is in

        time_after_current_saccade = current_turn + time_after_saccade
        time_after_current_saccade *= 1000  # turn to ms which is what neural_timestamps is in

        try:
            # Find first index to use in flat array
            flat_first_index_to_find = np.searchsorted(flat_neural_timestamps, time_before_current_saccade)
            # Then in 2D array
            first_index_in_neural_timstamps = np.where(
                neural_timestamps == flat_neural_timestamps[flat_first_index_to_find])

            flat_last_index_to_find = np.searchsorted(flat_neural_timestamps, time_after_current_saccade)
            # Then in 2D array
            last_index_in_neural_timstamps = np.where(neural_timestamps == flat_neural_timestamps[flat_last_index_to_find])

            # Define index for for loop start!
            # This is the first index of the imaging_data z_slice for the current saccade
            z_slice_counter_img_data = first_index_in_neural_timstamps[1][0]
            # This is the first index of the imaging_data t_slice for the current sacced
            t_volume_counter_img_data = first_index_in_neural_timstamps[0][0]

            t_volume_counter_saccade_data = 0

            #print("brain_activity_turns.shape: " + repr(brain_activity_turns.shape))
            #print('will collect ' + repr(flat_last_index_to_find - flat_first_index_to_find) + ' z-slices')

            for i in range(flat_last_index_to_find - flat_first_index_to_find):
                # Grab the imaging data
                brain_activity_turns[:, :, z_slice_counter_img_data, t_volume_counter_saccade_data, turn_counter] = \
                    imaging_data[:, :, z_slice_counter_img_data, t_volume_counter_img_data]

                # Set the data that we are copying out to 'nan' in the copied array in order to
                # get array with neural data W/O saccades!
                try:
                    brain_activity_no_turns[:, :, z_slice_counter_img_data, t_volume_counter_img_data] = np.nan
                except Exception as e:
                    print('z_slice_counter_img_data: ' + repr(z_slice_counter_img_data))
                    print('t_volume_counter_img_data: ' + repr(t_volume_counter_img_data))
                    print(e)
                # Go to next z_slice
                z_slice_counter_img_data += 1

                # If z_slice done, set back to zero and go to next t_volume_counter
                if z_slice_counter_img_data == z_dim:
                    z_slice_counter_img_data = 0
                    t_volume_counter_img_data += 1
                    t_volume_counter_saccade_data += 1
        except IndexError as e:
            print(e)


    return(brain_activity_turns, brain_activity_no_turns)

# Run the actual code:
def calc_sac_trig_activity(fly_folder_to_process_oak,
                           fictrac_path,
                           brain_path,
                           metadata_path,
                           savepath):
    """

    """
    logfile = utils.create_logfile(fly_folder_to_process_oak, function_name="saccade_triggered_activity")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    #fictrac_path = utils.convert_list_of_string_to_posix_path(fictrac_path)
    #brain_path = utils.convert_list_of_string_to_posix_path(brain_path)
    #metadata_path = utils.convert_list_of_string_to_posix_path(metadata_path)
    #savepath = utils.convert_list_of_string_to_posix_path(savepath)


    printlog("fictrac_path: " + repr(fictrac_path))
    printlog("brain_path: " + repr(brain_path))
    printlog("metadata_path: " + repr(metadata_path))
    printlog("savepath: " + repr(savepath))

    side_to_analyze = str(savepath).split('_sac_trig_act.nii')[0][-1]
    printlog("side_to_analyze: " + repr(side_to_analyze))

    # Find Saccades
    turns = find_saccades(fictrac_path)

    # Extract neural timestamps
    neural_timestamps = utils.load_timestamps(metadata_path)
    # Load brain data
    brain_data = nib.load(brain_path)
    brain_data = np.array(brain_data.dataobj)
    saccade_triggered_brain_activity, brain_activity_no_saccade = extract_saccade_triggered_neural_activity(
        brain_data, neural_timestamps,turns, turn_side = side_to_analyze)

    # Release memory
    del(brain_data)
    time.sleep(1)

    mean_brain_activity_no_saccade = np.nanmean(brain_activity_no_saccade, axis=-1, dtype=np.float32)
    #del(mean_brain_activity_no_saccade) # release memory
    #time.sleep(1)

    # Calculate mean of the extracted neural activity:
    mean_saccade_triggered_brain_activity = np.nanmean(saccade_triggered_brain_activity,axis=(3,4),dtype=np.float32)
    #del(saccade_triggered_brain_activity) # to clear memory
    #time.sleep(1)

    diff = mean_saccade_triggered_brain_activity - mean_brain_activity_no_saccade

    aff = np.eye(4)
    object_to_save = nib.Nifti1Image(diff, aff)
    # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
    pathlib.Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    object_to_save.to_filename(savepath)
    print('Successfully saved ' + savepath.name)