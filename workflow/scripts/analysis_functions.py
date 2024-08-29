"""

"""

import pathlib
import sys
import numpy as np
from scipy import signal

parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
# Import local modules
from brainsss import fictrac_utils

def extract_turn_bouts(datapath, minimal_time_between_turns, turn_thresh=200):
    """
    From https://github.com/ClandininLab/brezovec_volition_2023/blob/main/predict_turn_direction.py
    """
    try:
        if datapath.is_dir():
            if pathlib.Path(datapath, 'fictrac/fictrac_behavior_data.dat').is_file():
                fictrac_path = pathlib.Path(datapath, 'fictrac/fictrac_behavior_data.dat')
            else:
                fictrac_path = pathlib.Path(datapath, 'stimpack/loco/fictrac_behavior_data.dat')
            fictrac_raw = fictrac_utils.load_fictrac(fictrac_path)
    except AttributeError: # If we instead directly provide fictrac_raw
        fictrac_raw = datapath

    #turn_thresh = 200
    behavior = 'dRotLabZ'
    fictrac_smoothed = signal.savgol_filter(np.asarray(fictrac_raw[behavior]), 25, 3)
    fps = 100
    fictrac_smoothed = fictrac_smoothed * 180 / np.pi * fps  # now in deg/sec

    #fictrac_timestamps = np.arange(0, recording_duration * 60, 1 / fps)  # this is only approximately correct!
    fictrac_timestamps = np.arange(0, fictrac_smoothed.shape[0] / fps, 1 / fps) # This is correct assuming we really
    # always get exactly 100 fps!

    minimal_time_between_turns = minimal_time_between_turns * fps # convert from seconds to fps!

    ###########################
    ### Identify turn bouts ###
    ###########################

    peaks = {'L': [], 'R': []}
    heights = {'L': [], 'R': []}

    for turn, scalar in zip(['L', 'R'], [1, -1]):
        found_peaks = signal.find_peaks(fictrac_smoothed * scalar, height=turn_thresh,
                                        distance=minimal_time_between_turns)
        pks = found_peaks[0]
        pk_height = found_peaks[1]['peak_heights']

        # Note - this was not 'end of recording' but anything after 880 seconds!
        # Just return all turns, can deal with edge case outside this function!
        ### remove peaks that are too close to beginning or end of recording
        # will do 20sec window
        # here 20sec is 1000 tps
        #ind = np.where(pks > 88000)[0]
        #pks = np.delete(pks, ind)
        #pk_height = np.delete(pk_height, ind)

        #ind = np.where(pks < 3000)[0]
        #pks = np.delete(pks, ind)
        #pk_height = np.delete(pk_height, ind)

        # convert index to time in seconds!
        peaks[turn] = fictrac_timestamps[pks]
        heights[turn] = pk_height

    return (peaks)

def extract_isolated_turns(direction, other_direction, time_before_turn=3):
    """
    We are looking for signal before a turn. If we have turns before a turn the signal will
    be harder to see.
    Have a list with turns where no other turns happened before
    """

    turns = []
    for counter in range(len(direction)):
        # Ignore initial turns
        if direction[counter] > time_before_turn:
            # check if there's a turn i.e. 5 seconds before the current turn
            current_time_before_turn = direction[counter] - time_before_turn
            current_time_of_turn = direction[counter]

            # for same direction easy to check:
            if direction[counter - 1] > current_time_before_turn:
                # there's a turn
                # print('current turn: ' + repr(direction[counter]) + ', previous turn: ' + repr(direction[counter-1] ))
                pass
            else:
                # print('current turn: ' + repr(direction[counter]) + ', previous turn: ' + repr(direction[counter-1] ))
                pass

                interesting_turn = True
                # the other direction is a bit tricker. Probably easiest to loop through and look for a turn in a defined timeframe
                for current_other_turn in range(len(other_direction)):
                    if current_time_before_turn < other_direction[current_other_turn] < current_time_of_turn:
                        # ignore these turns
                        interesting_turn = False
                if interesting_turn:
                    print('current turn: ' + repr(direction[counter]))
                    turns.append(direction[counter])

    return (np.array(turns))


def get_time_index_neural_data(cleaned_up_turns,
                               neural_timestamps_seconds,
                               indeces_to_take,
                               time_before_turn):
    """
    Grab indeces before (as defined in time_before_turn in seconds) and up to a turn.
    Make sure time_before_turn is smaller than time between turns in cleaned_up_turns
    as it's possible to get the same idnex more than once otherwise>To be taken care of
    by
    """
    for current_time in cleaned_up_turns:
        start_time = current_time-time_before_turn
        end_time = current_time

        start_index_flat = np.searchsorted(neural_timestamps_seconds.flatten(), start_time)
        start_time_index = np.where(neural_timestamps_seconds ==
                                    neural_timestamps_seconds.flatten()[start_index_flat])[0][0]

        end_index_flat = np.searchsorted(neural_timestamps_seconds.flatten(), end_time)
        end_time_index = np.where(neural_timestamps_seconds ==
                                  neural_timestamps_seconds.flatten()[end_index_flat])[0][0]

        take_index=True
        index_counter = start_time_index
        while take_index:

            indeces_to_take.append(index_counter)

            index_counter += 1

            if index_counter == end_time_index:
                take_index = False

    return(indeces_to_take)
