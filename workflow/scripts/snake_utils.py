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