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

fly_folder_to_process = 'fly_003' # folder to be processed
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly

# do follow up analysis, enter the fly folder to be analyzed here.
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly
# YOUR SUNET ID
current_user = 'dtadres'

# SCRATCH_DIR
# SCRATCH_DIR = '/scratch/users/' + current_user
# Maybe in the future - if I implement this now I won't be able to test code locally on my computer

import pathlib
import json
import brainsss
import sys
from scripts import preprocessing


settings = brainsss.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])
fly_folder_to_process = pathlib.Path(dataset_path, fly_folder_to_process)
#####
# LOGGING
#####
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
width = 120 # can go into a config file as well.
if not pathlib.Path(logfile).is_file():
    brainsss.print_title(logfile, width)
    printlog(F"{fly_folder_to_process.name:^{width}}")
    brainsss.print_datetime(logfile, width)
#######

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
    if 'func' in key and 'Folder' in key:
        func_file_paths.append(fly_dirs_dict[key])
    elif 'anat' in key and 'Folder' in key:
        anat_file_paths.append(fly_dirs_dict[key])
    elif 'Fictrac' in key:
        fictrac_file_paths.append(fly_dirs_dict_path[key])

def create_path_func(fly_folder_to_process, list_of_paths, filename):
    final_path = []
    for current_path in list_of_paths:
        final_path.append(str(fly_folder_to_process) + current_path + filename)
    return(final_path)

ch1_func_file_paths = create_path_func(fly_folder_to_process, func_file_paths, '/functional_channel_1.nii')
ch2_func_file_paths = create_path_func(fly_folder_to_process, func_file_paths, '/functional_channel_2.nii')
#

#
fictrac_file_paths = create_path_func(fly_folder_to_process, fictrac_file_paths, '/functional_channel_2.nii')
#
#func_file_paths = ['func0', 'func1', 'func2']
#full_ch1_file_path = []
#for current_file_path in func_file_paths:
#    full_ch1_file_path.append(str(fly_folder_to_process) + '/' + current_file_path + '/imaging/functional_channel_1.nii')
# how to use expand example
# https://stackoverflow.com/questions/55776952/snakemake-write-files-from-an-array

rule all:
    input: 'io_files/test.txt'
    #input: expand("{f}", f=full_func_file_path)


rule bleaching_qc_func_rule:
    input:
        channel1 = expand("{ch1}", ch1=ch1_file_paths),
        channel2 = expand("{ch2}", ch2=ch2_file_paths)
    output:
        '/Users/dtadres/snake_brainsss/workflow/io_files/test.txt'
    run:
        preprocessing.bleaching_qc_test(logfile=logfile,
                                        ch1=input.channel1,
                                        ch2=input.channel2,
                                        print_output = output)


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



