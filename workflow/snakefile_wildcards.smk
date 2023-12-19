"""
Here the idea is to do preprocessing a little bit differently:

We assume that the fly_builder already ran and that the fly_dir

Note:
    Although jobs can directly read and write to $OAK during execution,
    it is recommended to first stage files from $OAK to $SCRATCH at the
    beginning of a series of jobs, and save the desired results back from
    $SCRATCH to $OAK at the end of the job campaign.

    We strongly recommend using $OAK to reference your group home directory in
    scripts, rather than its explicit path.
AND:
    Each compute node has a low latency, high-bandwidth Infiniband link to $SCRATCH.
    The aggregate bandwidth of the filesystem is about 75GB/s. So any job with high
    data performance requirements will take advantage from using $SCRATCH for I/O.
"""

# with config file type:
# ml python/3.9.0
# source .env_snakemake/bin/activate
# cd snake_brainsss/workflow
# snakemake -s snakefile_wildcards.smk --profile config_sherlock

fly_folder_to_process = 'fly_004' # folder to be processed
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly

# do follow up analysis, enter the fly folder to be analyzed here.
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly
# YOUR SUNET ID
current_user = 'dtadres'

# First n frames to average over when computing mean/fixed brain | Default None
# (average over all frames). A
meanbrain_n_frames =  None
# SCRATCH_DIR
SCRATCH_DIR = '/scratch/users/' + current_user
# Maybe in the future - if I implement this now I won't be able to test code locally on my computer
# Do this - if it's really much faster it's great for debugging actually. Since it's stays for 90 days
# I only need to copy a test dataset once and hopefully will be able to use data on there very quickly

import pathlib
import json
import brainsss
import sys
from scripts import preprocessing
import itertools

settings = brainsss.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])
fly_folder_to_process = pathlib.Path(dataset_path, fly_folder_to_process)

# Needed for logging
width = 120 # can go into a config file as well.
"""
########
# LOGGING - because this should run acyclical, no big logfile with everything but many smaller ones!
########
pathlib.Path('./logs').mkdir(exist_ok=True)
# Have one log file per fly! This will make everything super traceable!
logfile = './logs/' + fly_folder_to_process.name + '.txt'

# Not sure what this does exactly, from Bella's code
printlog = getattr(brainsss.Printlog(logfile=logfile),'print_to_log')
# Pipe all errors to the logfile
sys.stderr = brainsss.LoggerRedirect(logfile)
# Pipe all print statements (and other console output) to the logfile
sys.stdout = brainsss.LoggerRedirect(logfile)
# Problem: Snakemake runs twice. Seems to be a bug: https://github.com/snakemake/snakemake/issues/2350
# Only print title and fly if logfile doesn't yet exist
if not pathlib.Path(logfile).is_file():
    brainsss.print_title(logfile, width)
    printlog(F"{fly_folder_to_process.name:^{width}}")
    brainsss.print_datetime(logfile, width)
#######"""

fly_dirs_dict_path = pathlib.Path(fly_folder_to_process, fly_folder_to_process.name + '_dirs.json')

with open(pathlib.Path(fly_dirs_dict_path),'r') as file:
    fly_dirs_dict = json.load(file)

#####
# Prepare filepaths to be used
#####
# Snakemake runs acyclical, meaning it checks which input depends on the output of which rule
# in order to parallelize a given snakefile.
# I'll therefore keep variables with lists of paths that will be feed into a given rule.
# These lists of paths are created here.

# Imaging data paths
func_file_paths = []
anat_file_paths = []
fictrac_file_paths = []
for key in fly_dirs_dict:
    #print(key)
    if 'func' in key and 'Imaging' in key:
        func_file_paths.append(fly_dirs_dict[key])
    elif 'anat' in key and 'Imaging' in key:
        anat_file_paths.append(fly_dirs_dict[key])
    elif 'Fictrac' in key:
        fictrac_file_paths.append(fly_dirs_dict[key])

def create_path_func(fly_folder_to_process, list_of_paths, filename=''):
    """
    Creates lists of path that can be feed as input/output to snakemake rules
    :param fly_folder_to_process: a folder pointing to a fly, i.e. /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_001
    :param list_of_paths: a list of path created, usually created from fly_dirs_dict (e.g. fly_004_dirs.json)
    :param filename: filename to append at the end. Can be nothing (i.e. for fictrac data).
    :return: list of full paths to a file based on the list_of_path provided
    """
    final_path = []
    for current_path in list_of_paths:
        final_path.append(str(fly_folder_to_process) + current_path + filename)

    return(final_path)

def create_output_path_func(list_of_paths, filename):
    """
    :param list_of_paths: expects a list of paths pointing to a file, for example from variable full_fictrac_file_paths
    :param filename: filename
    """
    final_path = []
    for current_path in list_of_paths:
        if isinstance(current_path, list):
            # This is for lists of lists, for example created by :func: create_paths_each_experiment
            # If there's another list assume that we only want one output file!
            # For example, in the bleaching_qc the reason we have a list of lists is because each experiment
            # is plotted in one file. Hence, the output should be only one file
            #print(pathlib.Path(pathlib.Path(current_path[0]).parent, filename))
            final_path.append(pathlib.Path(pathlib.Path(current_path[0]).parent, filename))
        else:
            final_path.append(pathlib.Path(pathlib.Path(current_path).parent, filename))

    return(final_path)

def create_paths_each_experiment(func_and_anat_paths):
    """
    get paths to imaging data as list of lists, i.e.
    [[func0/func_channel1.nii, func0/func_channel2.nii], [func1/func_channel1.nii, func1/func_channel2.nii]]
    :param func_and_anat_paths: a list with all func and anat path as defined in 'fly_004_dirs.json'
    """
    imaging_paths_by_folder_oak = []
    imaging_path_by_folder_scratch = []
    for current_path in func_and_anat_paths:
        if 'func' in current_path:
            imaging_paths_by_folder_oak.append([
                str(fly_folder_to_process) + current_path + '/functional_channel_1.nii',
                str(fly_folder_to_process) + current_path + '/functional_channel_2.nii']
                )
            imaging_path_by_folder_scratch.append([
                SCRATCH_DIR + '/data/' + imaging_paths_by_folder_oak[-1][0].split('data')[-1],
                SCRATCH_DIR + '/data/' + imaging_paths_by_folder_oak[-1][1].split('data')[-1]])
        elif 'anat' in current_path:
            imaging_paths_by_folder_oak.append([
                str(fly_folder_to_process) + current_path + '/anatomy_channel_1.nii',
                str(fly_folder_to_process) + current_path + '/anatomy_channel_2.nii']
                )
            imaging_path_by_folder_scratch.append([
                SCRATCH_DIR + '/data/' + imaging_paths_by_folder_oak[-1][0].split('data')[-1],
                SCRATCH_DIR + '/data/' + imaging_paths_by_folder_oak[-1][1].split('data')[-1]])
    return(imaging_paths_by_folder_oak, imaging_path_by_folder_scratch)

func_and_anat_paths = func_file_paths + anat_file_paths

imaging_paths_by_folder_oak, imaging_paths_by_folder_scratch = create_paths_each_experiment(func_and_anat_paths)
print('HERE' + repr(imaging_paths_by_folder_oak))
print('AND HERE' + repr(imaging_paths_by_folder_scratch))

#######
# Data path on OAK
#######
# Get imaging data paths for func
ch1_func_file_oak_paths = create_path_func(fly_folder_to_process, func_file_paths, '/functional_channel_1.nii')
ch2_func_file_oak_paths = create_path_func(fly_folder_to_process, func_file_paths, '/functional_channel_2.nii')
# and for anat data
ch1_anat_file_oak_paths = create_path_func(fly_folder_to_process, anat_file_paths, '/anatomy_channel_1.nii')
ch2_anat_file_oak_paths = create_path_func(fly_folder_to_process, anat_file_paths, '/anatomy_channel_2.nii')
# List of all imaging data
all_imaging_oak_paths = ch1_func_file_oak_paths + ch2_func_file_oak_paths + ch1_anat_file_oak_paths + ch2_anat_file_oak_paths

# Fictrac files are named non-deterministically (could be changed of course) but for now
# the full filename is in the fly_dirs_dict
full_fictrac_file_oak_paths = create_path_func(fly_folder_to_process, fictrac_file_paths)
#######
# Data path on SCRATCH
#######
def convert_oak_path_to_scratch(oak_path):
    """

    :param oak_path: expects a list of path, i.e. ch1_func_file_oak_paths
    :return:
    """
    all_scratch_paths = []
    for current_path in oak_path:
        relevant_path_part = current_path.split('data')[-1] # data seems to be the root folder everyone is using
        all_scratch_paths.append(SCRATCH_DIR + '/data/' + relevant_path_part)
    return(all_scratch_paths)

ch1_func_file_scratch_paths = convert_oak_path_to_scratch(ch1_func_file_oak_paths)
ch2_func_file_scratch_paths = convert_oak_path_to_scratch(ch2_func_file_oak_paths)
ch1_anat_file_scratch_paths = convert_oak_path_to_scratch(ch1_anat_file_oak_paths)
ch2_anat_file_scratch_paths = convert_oak_path_to_scratch(ch2_anat_file_oak_paths)
#full_fictrac_file_scratch_paths = convert_oak_path_to_scratch(full_fictrac_file_oak_paths)
all_imaging_scratch_paths = ch1_func_file_scratch_paths + ch2_func_file_scratch_paths + ch1_anat_file_scratch_paths + ch2_anat_file_scratch_paths

#####
# Output data path ON SCRATCH!!!!
#####
# Output files for fictrac_qc rule
#fictrac_output_files_2d_hist_fixed = create_output_path_func(list_of_paths=full_fictrac_file_scratch_paths,
#                                                             filename='fictrac_2d_hist_fixed.png')
fictrac_output_files_2d_hist_fixed = create_output_path_func(list_of_paths=full_fictrac_file_oak_paths,
                                                             filename='fictrac_2d_hist_fixed.png')

#bleaching_qc_output_files = create_output_path_func(list_of_paths=imaging_paths_by_folder,
#                                                    filename='bleaching.png')
bleaching_qc_output_files = create_output_path_func(list_of_paths=imaging_paths_by_folder_oak,
                                                           filename='bleaching.png')

# TESTING
# full_fictrac_file_paths = [full_fictrac_file_paths[0]]
# Output files for bleaching_qc rule

#print(fictrac_output_files_2d_hist_fixed)
# how to use expand example
# https://stackoverflow.com/questions/55776952/snakemake-write-files-from-an-array

rule all:
    input:
         expand("{fictrac_output}", fictrac_output=fictrac_output_files_2d_hist_fixed),
         expand("{bleaching_qc_png}", bleaching_qc_png=bleaching_qc_output_files)
    #input: expand("{f}", f=full_func_file_path)
            #'io_files/test.txt',

"""rule bleaching_qc_func_rule:
    "This should not run because the output is not requested in rule all"
    input:
        channel1_func = expand("{ch1_func}", ch1_func=ch1_func_file_paths),
        channel2_func = expand("{ch2_func}", ch2_func=ch2_func_file_paths)
    output:
        'io_files/test.txt'
    run:
        preprocessing.bleaching_qc_test(ch1=input.channel1_func,
                                        ch2=input.channel2,
                                        print_output = output)"""

rule copy_to_scratch_rule:
    threads: 1
    input:
        all_imaging_oak_paths
    output:
        all_imaging_scratch_paths
    run:
        try:
            preprocessing.copy_to_scratch(fly_directory = fly_folder_to_process,
                                          paths_on_oak = all_imaging_oak_paths,
                                          paths_on_scratch = all_imaging_scratch_paths
                                          )
        except Exception as error_stack:
            logfile = brainsss.create_logfile(fly_folder_to_process,function_name='ERROR_copy_to_scratch')
            brainsss.write_error(logfile=logfile,
                                 error_stack=error_stack,
                                 width=width)

rule fictrac_qc_rule:
    threads: 1
    input:
        full_fictrac_file_oak_paths
        #fictrac_file_paths = expand("{fictrac}", fictrac=full_fictrac_file_scratch_paths)
    output:
        expand("{fictrac_output}", fictrac_output=fictrac_output_files_2d_hist_fixed)
    run:
        try:
            preprocessing.fictrac_qc(fly_folder_to_process,
                                    fictrac_file_paths= input.fictrac_file_paths,
                                    fictrac_fps=50 # AUTOMATE THIS!!!! ELSE BUG PRONE!!!!
                                    )
        except Exception as error_stack:
            logfile = brainsss.create_logfile(fly_folder_to_process,function_name='ERROR_fictrac_qc_rule')
            brainsss.write_error(logfile=logfile,
                                 error_stack=error_stack,
                                 width=width)

rule bleaching_qc_rule:
    """
    Out of memory with 1 & 4 threads on sherlock.
    With 16 I had a ~45% memory utiliziation. Might be worth trying 8 or 10 cores
    """
    threads: 16 # This is parallelized so more should generally be better!
    input:
        imaging_paths_by_folder_scratch
    output:
        #expand("{bleaching_qc_png}", bleaching_qc_png=bleaching_qc_output_files)
        "{bleaching_qc_png}" # this might parallelize the operation
    run:
        try:
            preprocessing.bleaching_qc(fly_directory=fly_folder_to_process,
                                        imaging_data_path_read_from=imaging_paths_by_folder_scratch,
                                        imaging_data_path_save_to=imaging_paths_by_folder_oak
                                        #print_output = output
            )
        except Exception as error_stack:
            logfile = brainsss.create_logfile(fly_folder_to_process,function_name='ERROR_bleaching_qc_rule')
            brainsss.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)
'''
rule make_mean_brain_rule:
    """
    
    """
    threads: 16
    input:
        all_imaging_paths
    output: 'foo'
        # every nii file is made to a mean brain! Can predict how they
        # are going to be called and put them here.
    run:
        try:
            preprocessing.make_mean_brain(meanbrain_n_frames)
        except Exception as error_stack:
            logfile = brainsss.create_logfile(fly_folder_to_process,function_name='ERROR_make_mean_brain')
            brainsss.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

'''

"""
https://farm.cse.ucdavis.edu/~ctbrown/2023-snakemake-book-draft/chapter_9.html
While wildcards and expand use the same syntax, they do quite different things.

expand generates a list of filenames, based on a template and a list of values to insert
into the template. It is typically used to make a list of files that you want snakemake 
to create for you.

Wildcards in rules provide the rules by which one or more files will be actually created. 
They are recipes that say, "when you want to make a file with name that looks like THIS, 
you can do so from files that look like THAT, and here's what to run to make that happen.

"""



