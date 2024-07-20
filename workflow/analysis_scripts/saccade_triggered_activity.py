"""
Sometimes correlation does not work well to tease out signal.

Instead, we can go the other way: extract the brain around a saccade and
just take the mean.
"""
import numpy as np
import nibabel as nib
import pathlib
import scipy

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
    time_before_saccade = 0.5  # seconds
    time_after_saccade = 0.5  # seconds

    # But 3 full stacks should be enough
    max_number_of_stacks_per_saccade = 3

    # Get brain size
    x_dim = imaging_data.shape[0]
    y_dim = imaging_data.shape[1]
    z_dim = imaging_data.shape[2]

    brain_activity_turns = np.zeros((x_dim, y_dim, z_dim, max_number_of_stacks_per_saccade, len(turns[turn_side])),
                                    dtype=np.float32sac)
    brain_activity_turns.fill(np.nan)
    # Need to flatten the neural timestamps array as searchsorted only works on 1D array
    flat_neural_timestamps = neural_timestamps.flatten()
    for turn_counter, current_turn in enumerate(turns[turn_side]):
        print('turn counter: ' + repr(turn_counter))
        time_before_current_saccade = current_turn - time_before_saccade
        time_before_current_saccade *= 1000  # turn to ms which is what neural_timestamps is in

        time_after_current_saccade = current_turn + time_after_saccade
        time_after_current_saccade *= 1000  # turn to ms which is what neural_timestamps is in

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

        for i in range(flat_last_index_to_find - flat_first_index_to_find):
            # Grab the imaging data
            brain_activity_turns[:, :, z_slice_counter_img_data, t_volume_counter_saccade_data, turn_counter] = \
                imaging_data[:, :, z_slice_counter_img_data, t_volume_counter_img_data]

            # Go to next z_slice
            z_slice_counter_img_data += 1

            # If z_slice done, set back to zero and go to next t_volume_counter
            if z_slice_counter_img_data == z_dim:
                z_slice_counter_img_data = 0
                t_volume_counter_img_data += 1
                t_volume_counter_saccade_data += 1

    return (brain_activity_turns)

# Run the actual code:
def calc_sac_trig_activity(fictrac_path, brain_path, metadata_path, savepath):
    """

    """

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    fictrac_path = utils.convert_list_of_string_to_posix_path(fictrac_path)
    brain_path = utils.convert_list_of_string_to_posix_path(brain_path)
    metadata_path = utils.convert_list_of_string_to_posix_path(metadata_path)
    savepath = utils.convert_list_of_string_to_posix_path(savepath)

    # Find Saccades
    turns = find_saccades(fictrac_path[0])

    # Extract neural timestamps
    neural_timestamps = utils.load_timestamps(metadata_path[0])
    # Load brain data
    brain_data = nib.load(brain_path[0])
    brain_data = np.array(brain_data.dataobj)

    side_to_analyze = str(savepath[0]).str('_sac_trig_act.nii')[0][-1]
    print("side_to_analyze: " + repr(side_to_analyze))

    brain_activity_left_turns = extract_saccade_triggered_neural_activity(brain_data, neural_timestamps,turns, turn_side = side_to_analyze)
    # Calculate mean of the extracted neural activity:
    median_brain_activity_left_turns = np.nanmean(brain_activity_left_turns,axis=(3,4))

    aff = np.eye(4)
    object_to_save = nib.Nifti1Image(median_brain_activity_left_turns, aff)
    # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
    object_to_save.to_filename(savepath)
