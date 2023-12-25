# Functions to be called from snakefiles
import numpy as np
def mem_mb_times_threads(wildcards, threads):
    """
    Returns memory in mb as 7500Mb/thread (I think we have ~8Gb/thread? to be confirmed)
    Note: wildcards is essential here!
    :param threads:
    :return:
    """
    return(threads * 7500)

def mem_mb_less_times_input(wildcards, input):
    """
    Returns memory in mb as 1.5*input memory size or 1Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return(max(input.size_mb*1.5, 1000))

def mem_mb_times_input(wildcards, input):
    """
    Returns memory in mb as 2.5*input memory size or 1Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return(max(input.size_mb*2.5, 1000))

def mem_mb_more_times_input(wildcards, input):
    """
    Returns memory in mb as 3.5*input memory size or 1Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return(max(input.size_mb*3.5, 1000))

def disk_mb_times_input(wildcards, input):
    """
    Returns memory in mb as 2.5*input memory size or 1Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return(max(input.size_mb*2.5, 1000))

def time_for_moco_input(wildcards, input):
    """
    Returns time in minutes based on input.
    I know that at 5 minute recording should take at least 30 minutes with input size ~2Gb
    I also know that a 30 minute recording should take ~7 hours with ~11Gb input size.
    And an anatomical scan also ~25Gb should take ~16 hours
    The slowest seems to be the functional 7 hours scan but even that is taken
    We'll do 45min per anatomy channel
    Note that we can have 1, 2 or 3 input files...Assume that only one of the files is the
    anatomy channel!
    :param wildcards:
    :param input:
    :return:
    """
    if input.brain_paths_ch1 != [] and input.brain_paths_ch2 != [] and input.brain_paths_ch3 != []:
        # if all three channels are used
        time_in_minutes = (input.size_mb/1000)*15 # /1000 to get Gb, then *minutes
    elif (input.brain_paths_ch1 != [] and input.brain_paths_ch2 != []) or \
        (input.brain_paths_ch1 != [] and input.brain_paths_ch3 != []) or \
        (input.brain_paths_ch2 != [] and input.brain_paths_ch3 != []):
        # if only two channels are in use
        time_in_minutes = (input.size_mb/1000)*30 # /1000 to get Gb, then *minutes
    else:
        # only one channel is provided:
        time_in_minutes = (input.size_mb/1000)*45 # /1000 to get Gb, then *minutes

    #hours = int(np.floor(time_in_minutes / 60))
    #minutes = int(np.ceil(time_in_minutes % 60))
    #string_to_return = str(hours) + ':' + str(minutes) + ':00'
    return(str(time_in_minutes*60))