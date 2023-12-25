# Functions to be called from snakefiles

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
    care of with 45minutes per Gb
    :param wildcards:
    :param input:
    :return:
    """
    return(input.size_mb/1000*45)