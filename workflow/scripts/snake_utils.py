# Functions to be called from snakefiles
import numpy as np


def mem_mb_times_threads(wildcards, threads):
    """
    Returns memory in mb as 7500Mb/thread (I think we have ~8Gb/thread? to be confirmed)
    Note: wildcards is essential here!
    :param threads:
    :return:
    """
    return threads * 7500


def mem_mb_less_times_input(wildcards, input):
    """
    Returns memory in mb as 1.5*input memory size or 1.5Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return max(input.size_mb * 1.5, 1500)


def mem_mb_times_input(wildcards, input):
    """
    Returns memory in mb as 2.5*input memory size or 2Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return max(input.size_mb * 2.5, 2000)


def mem_mb_more_times_input(wildcards, input):
    """
    Returns memory in mb as 3.5*input memory size or 4Gb, whichever is larger,
    but not more than 256Gb which is the submission limit for sherlock.
    :param wildcards:
    :param input:
    :return:
    """
    calculated_memory = max(input.size_mb * 3.5, 4000)

    if calculated_memory < 256000:
        return(calculated_memory)
    else:
        return(256000)


def mem_mb_much_more_times_input(wildcards, input):
    """
    Returns memory in mb as 5.5*input memory size or 1Gb, whichever is larger,
    but not more than 256Gb which is the submission limit for sherlock.
    :param wildcards:
    :param input:
    :return:
    """
    calculated_memory = max(input.size_mb * 5.5, 10000)

    if calculated_memory < 256000:
        return(calculated_memory)
    else:
        return(256000)


def disk_mb_times_input(wildcards, input):
    """
    Returns memory in mb as 2.5*input memory size or 1Gb, whichever is larger
    :param wildcards:
    :param input:
    :return:
    """
    return max(input.size_mb * 2.5, 1000)

def mb_for_moco_input(wildcards, input):
    """

    :param wildcards:
    :param input:
    :return:
    """
    # We need pretty much exactly 4 times the input file size OF ONE CHANNEL.
    # If we have two channels we only need ~2 times.
    #
    if (
            input.brain_paths_ch1 != []
            and input.brain_paths_ch2 != []
            and input.brain_paths_ch3 != []
    ):
        # if all three channels are used
        mem_mb = int((input.size_mb*5.5)/3)
    elif (
        (input.brain_paths_ch1 != [] and input.brain_paths_ch2 != [])
        or (input.brain_paths_ch1 != [] and input.brain_paths_ch3 != [])
        or (input.brain_paths_ch2 != [] and input.brain_paths_ch3 != [])
    ):
        # if only two channels are in use
        mem_mb = int((input.size_mb * 5.5)/2)
    else:
        # only one channel is provided:
        mem_mb = (input.size_mb * 5.5)
    # Sherlock only allows us to request up to 256 Gb per job
    if mem_mb > 256000:
        mem_mb = 256000

    return(max(mem_mb, 5000))

# if all three channels are used

def time_for_moco_input(wildcards, input):
    """
    Note that all benchmarks were done with 'threads=32' and 'cores=8'.
    Returns time in minutes based on input. Use this with the multiprocessing motion
    correction code.
    We needs about 5 minutes for 1 Gb of for two channels. Double it to be on the safe
    side for now

    :param wildcards: Snakemake requirement
    :param input: intput from snakefile. Needed to access filesize
    :return:
    """
    if input == "<TBD>":  # This should ONLY happen during a -np call of snakemake.
        string_to_return = str(2) + "h"
    elif (
        input.brain_paths_ch1 != []
        and input.brain_paths_ch2 != []
        and input.brain_paths_ch3 != []
    ):
        # if all three channels are used
        time_in_minutes = (input.size_mb / 1000) * 2.5  # /1000 to get Gb, then *minutes
    elif (
        (input.brain_paths_ch1 != [] and input.brain_paths_ch2 != [])
        or (input.brain_paths_ch1 != [] and input.brain_paths_ch3 != [])
        or (input.brain_paths_ch2 != [] and input.brain_paths_ch3 != [])
    ):
        # if only two channels are in use
        time_in_minutes = (input.size_mb / 1000) * 5 # /1000 to get Gb, then *minutes
    else:
        # only one channel is provided:
        time_in_minutes = (input.size_mb / 1000) * 10  # /1000 to get Gb, then *minutes

    # hours = int(np.floor(time_in_minutes / 60))
    # minutes = int(np.ceil(time_in_minutes % 60))
    # string_to_return = str(hours) + ':' + str(minutes) + ':00'

    # Define a minimum time of 10 minutes
    time_in_minutes = max(time_in_minutes, 10)

    # https: // snakemake.readthedocs.io / en / stable / snakefiles / rules.html
    # If we want minutes we just add a 'm' after the number
    string_to_return = str(time_in_minutes) + "m"
    return(string_to_return)
def OLDtime_for_moco_input(wildcards, input):
    """
    ### This was used for the chunk based NOT-multiprocessed code ####
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
    if input == "<TBD>":  # This should ONLY happen during a -np call of snakemake.
        string_to_return = str(2) + "h"
    elif (
        input.brain_paths_ch1 != []
        and input.brain_paths_ch2 != []
        and input.brain_paths_ch3 != []
    ):
        # if all three channels are used
        time_in_minutes = (input.size_mb / 1000) * 15  # /1000 to get Gb, then *minutes
    elif (
        (input.brain_paths_ch1 != [] and input.brain_paths_ch2 != [])
        or (input.brain_paths_ch1 != [] and input.brain_paths_ch3 != [])
        or (input.brain_paths_ch2 != [] and input.brain_paths_ch3 != [])
    ):
        # if only two channels are in use
        time_in_minutes = (input.size_mb / 1000) * 30  # /1000 to get Gb, then *minutes
    else:
        # only one channel is provided:
        time_in_minutes = (input.size_mb / 1000) * 45  # /1000 to get Gb, then *minutes

    # hours = int(np.floor(time_in_minutes / 60))
    # minutes = int(np.ceil(time_in_minutes % 60))
    # string_to_return = str(hours) + ':' + str(minutes) + ':00'

    # https: // snakemake.readthedocs.io / en / stable / snakefiles / rules.html
    # If we want minutes we just add a 'm' after the number - TEST!!!mem_mb_more_times_input
    string_to_return = str(time_in_minutes) + "m"
    return string_to_return


'''def time_for_correlation(wildcards, input):
    """
    returns time in based on input - for a 5 min test case we need just under 5 minutes.
    That's about 4 Gb of h5 file for a single functional channel. Lets go with 10 minutes for 4Gb so 2.5 minutes
    for each Gb

    # >DOESNT WORKThat's about 10 Mb of fictrac data. So each Mb of fictrac data is ~1 minute of compute time
    :param wildcards:
    :param input
    :return: time in seconds
    """
    # return(input.fictrac_path.size_mb*60) DOESNT WORK!
    time_in_minutes = (input.size_mb/1000)*2.5
    return(time_in_minutes*60) # This turned out to give minutes...'''
