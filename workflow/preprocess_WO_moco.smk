# For sparse lines, it's possible that moco does more harm than good.
# To quickly test this, slightly modified preprocess run

import natsort
fictrac_fps = 100 # AUTOMATE THIS!!!! ELSE FOR SURE A MISTAKE WILL HAPPEN IN THE FUTURE!!!!
# TODO!!!! Instead of just believing a framerate, use the voltage signal recorded during imaging
# that defines the position of a given frame!
#<<<<

# First n frames to average over when computing mean/fixed brain | Default None
# (average over all frames).
meanbrain_n_frames =  None
# For logfile
width=120
##########################################################
import pathlib
import json
import sys
import datetime
# path of workflow i.e. /Users/dtadres/snake_brainsss/workflow
#scripts_path = pathlib.Path(__file__).resolve()
scripts_path = workflow.basedir # Exposes path to this file
from brainsss import utils
from scripts import preprocessing
from scripts import snake_utils
import os
print(os.getcwd())

current_user = config['user'] # this is whatever is entered when calling snakemake, i.e.
# snakemake --profile profiles/simple_slurm -s snaketest.smk --config user=jcsimon would
# yield 'jcsimon' here
settings = utils.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])

# Define path to imports to find fly.json!
#fly_folder_to_process_oak = pathlib.Path(dataset_path,fly_folder_to_process)
fly_folder_to_process_oak = pathlib.Path(os.getcwd())
print('Analyze data in ' + repr(fly_folder_to_process_oak.as_posix()))

# Read channel information from fly.json file
# If fails here, means the folder specified doesn't exist. Check name.
# Note: Good place to let user know to check folder and exit!
with open(pathlib.Path(fly_folder_to_process_oak, 'fly.json'), 'r') as file:
    fly_json = json.load(file)
# This needs to come from some sort of json file the experimenter
# creates while running the experiment. Same as genotype.
FUNCTIONAL_CHANNELS = fly_json['functional_channel']
# It is probably necessary to forcibly define STRUCTURAL_CHANNEL if not defined
# Would be better to have an error to be explicit!

# Throw an error if missing! User must provide this!
STRUCTURAL_CHANNEL = fly_json['structural_channel']
if STRUCTURAL_CHANNEL != 'channel_1' and \
    STRUCTURAL_CHANNEL != 'channel_2' and \
        STRUCTURAL_CHANNEL != 'channel_3':
    print('!!! ERROR !!!')
    print('You must provide "structural_channel" in the "fly.json" file for snake_brainsss to run!')
    sys.exit()
    # This would be a implicit fix. Not great as it'll
    # hide potential bugs. Better explicit
    #STRUCTURAL_CHANNEL = FUNCTIONAL_CHANNELS[0]

def ch_exists_func(channel):
    """
    Check if a given channel exists in global variables STRUCTURAL_CHANNEL and FUNCTIONAL_CHANNELS
    :param channel:
    :return:
    """
    if 'channel_' + str(channel) in STRUCTURAL_CHANNEL or 'channel_' + str(channel) in FUNCTIONAL_CHANNELS:
        ch_exists = True
    else:
        ch_exists = False
    return(ch_exists)

# Bool for which channel exists in this particular recording.
# IMPORTANT: One FLY must have the same channels per recording. This
# makes sense: If we have e.g. GCaMP and tdTomato we would always
# record from both the green and red channel, right?
CH1_EXISTS = ch_exists_func("1")
CH2_EXISTS = ch_exists_func("2")
CH3_EXISTS = ch_exists_func("3")

####
# Load fly_dir.json
####
fly_dirs_dict_path = pathlib.Path(fly_folder_to_process_oak, fly_folder_to_process_oak.name + '_dirs.json')
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
imaging_file_paths = []
fictrac_file_paths = []
for key in fly_dirs_dict:
    if 'Imaging' in key:
        imaging_file_paths.append(fly_dirs_dict[key][1::])
        # this yields for example 'func2/imaging'
    elif 'Fictrac' in key:
        fictrac_file_paths.append(fly_dirs_dict[key][1::])
        # This yields for example 'func1/fictrac/fictrac_behavior_data.dat'

#######
# Data path on OAK
#######
'''
# Maybe not used anymore. Might be useful to create paths to SCRATCH, though...
def create_file_paths(path_to_fly_folder, imaging_file_paths, filename, func_only=False):
    """
    Creates lists of path that can be feed as input/output to snakemake rules taking into account that
    different fly_00X folder might have different channels!
    :param path_to_fly_folder: a folder pointing to a fly, i.e. /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_001
    :param list_of_paths: a list of path created, usually created from fly_dirs_dict (e.g. fly_004_dirs.json)
    :param filename: filename to append at the end. Can be nothing (i.e. for fictrac data).
    :param func_only: Sometimes we need only paths from the functional channel, for example for z-scoring
    :return: list of filepaths
    """
    list_of_filepaths = []
    for current_path in imaging_file_paths:
        if func_only:
            if 'func' in current_path:
                if CH1_EXISTS:
                    list_of_filepaths.append(pathlib.Path(path_to_fly_folder, current_path, 'channel_1' + filename))
                if CH2_EXISTS:
                    list_of_filepaths.append(pathlib.Path(path_to_fly_folder, current_path, 'channel_2' + filename))
                if CH3_EXISTS:
                    list_of_filepaths.append(pathlib.Path(path_to_fly_folder, current_path, 'channel_3' + filename))
        else:
            if CH1_EXISTS:
                list_of_filepaths.append(pathlib.Path(path_to_fly_folder,current_path,'channel_1' + filename))
            if CH2_EXISTS:
                list_of_filepaths.append(pathlib.Path(path_to_fly_folder,current_path,'channel_2' + filename))
            if CH3_EXISTS:
                list_of_filepaths.append(pathlib.Path(path_to_fly_folder,current_path,'channel_3' + filename))
    return(list_of_filepaths)'''

FICTRAC_PATHS = []
for current_path in fictrac_file_paths:
    FICTRAC_PATHS.append(current_path.split('/fictrac_behavior_data.dat')[0])
#
list_of_paths_func = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        list_of_paths_func.append(current_path.split('/imaging')[0])

list_of_channels = []
if CH1_EXISTS:
    list_of_channels.append("1")
if CH2_EXISTS:
    list_of_channels.append("2")
if CH3_EXISTS:
    list_of_channels.append("3")

print("list_of_channels" + repr(list_of_channels))

# For wildcards we need lists of elements of the path for each folder.
list_of_paths = []
for current_path in imaging_file_paths:
    list_of_paths.append(current_path.split('/imaging')[0])
# This is a list of all imaging paths so something like this
# ['anat0', 'func0', 'func1']
print('list_of_paths ' +repr(list_of_paths) )

# Behaviors to correlate with neural activity
corr_behaviors = ['dRotLabZneg', 'dRotLabZpos', 'dRotLabY']

rule all:
    """
    See: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html
        By default snakemake executes the first rule in the snakefile. This gives rise to pseudo-rules at the beginning 
        of the file that can be used to define build-targets similar to GNU Make
    Or in other words: Here we define which file we want at the end. Snakemake checks which one is there and which 
    one is missing. It then uses the other rules to see how it can produce the missing files.
    """
    threads: 1 # should be sufficent
    resources: mem_mb=1000 # should be sufficient
    input:
        ###
        # Meanbrain
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{meanbr_imaging_paths}/imaging/channel_{meanbr_ch}_mean.nii",
            meanbr_imaging_paths=list_of_paths,
            meanbr_ch=list_of_channels),
        ####
        # Z-score
        ####
        expand(str(fly_folder_to_process_oak)
               + "/{zscore_imaging_paths}/NO_MOCO/channel_1_zscore.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
            zscore_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{zscore_imaging_paths}/NO_MOCO/channel_2_zscore.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
            zscore_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{zscore_imaging_paths}/NO_MOCO/channel_3_zscore.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
            zscore_imaging_paths=list_of_paths_func),
        ###
        # temporal high-pass filter
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_1_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
               temp_HP_filter_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_2_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
               temp_HP_filter_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_3_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
               temp_HP_filter_imaging_paths=list_of_paths_func),
        ###
        # correlation with fictrac behavior
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{corr_imaging_paths}/NO_MOCO/corr/channel_1_corr_{corr_behavior}.nii" if 'channel_1' in FUNCTIONAL_CHANNELS and len(FICTRAC_PATHS) > 0 else [],
               corr_imaging_paths=list_of_paths_func, corr_behavior=corr_behaviors),
        expand(str(fly_folder_to_process_oak)
               + "/{corr_imaging_paths}/NO_MOCO/corr/channel_2_corr_{corr_behavior}.nii" if 'channel_2' in FUNCTIONAL_CHANNELS and len(FICTRAC_PATHS) > 0 else [],
               corr_imaging_paths=list_of_paths_func, corr_behavior=corr_behaviors),
        expand(str(fly_folder_to_process_oak)
               + "/{corr_imaging_paths}/NO_MOCO/corr/channel_3_corr_{corr_behavior}.nii" if 'channel_3' in FUNCTIONAL_CHANNELS and len(FICTRAC_PATHS) > 0 else [],
               corr_imaging_paths=list_of_paths_func, corr_behavior=corr_behaviors),

rule make_mean_brain_rule:
    """
    Here it should be possible to parallelize quite easily as each input file creates
    one output file!

    input would be something like
    paths_to_use = ['../fly_004/func0/imaging/functional_channel_1', '../fly_004/func1/imaging/functional_channel_2']

    rule all would request the 'mean' brain of each of those
    expand("{imaging_path}_mean.nii", imaging_path=paths_to_use)

    rule make_mean_brain_rule:
        input: "{imaging_path}.nii"
        output: "{imaging_path}_mean.nii"
        run: function(imaging_path)

    which will do:
        brain = read(input)
        mean_brain = mean(brain)
        save.mean_brain(output)
    """
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input,  #snake_utils.mem_mb_times_input #mem_mb=snake_utils.mem_mb_more_times_input
        runtime='10m' # should be enough
    input:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths}/imaging/channel_{meanbr_ch}.nii"
    output:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths}/imaging/channel_{meanbr_ch}_mean.nii"
    run:
        try:
            preprocessing.make_mean_brain(fly_directory=fly_folder_to_process_oak,
                meanbrain_n_frames=meanbrain_n_frames,
                path_to_read=input,
                path_to_save=output,
                rule_name='make_mean_brain_rule')
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_make_mean_brain')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule zscore_rule:
    """
    """
    threads: snake_utils.threads_8x_more_input

    input:
        path_ch1=str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/imaging/channel_1.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        path_ch2=str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/imaging/channel_2.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        path_ch3=str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/imaging/channel_3.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],

    output:
        zscore_path_ch1=str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/NO_MOCO/channel_1_zscore.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2=str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/NO_MOCO/channel_2_zscore.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3=str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/NO_MOCO/channel_3_zscore.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    run:
        try:
            preprocessing.zscore(fly_directory=fly_folder_to_process_oak,
                dataset_path=[input.path_ch1, input.path_ch2, input.path_ch3],
                zscore_path=[output.zscore_path_ch1, output.zscore_path_ch2, output.zscore_path_ch3])
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_zscore')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule temporal_high_pass_filter_rule:
    """

    """
    threads: snake_utils.threads_per_memory_more
    resources:
        mem_mb=snake_utils.mem_mb_more_times_input,
        runtime='90m'  # The call to 1d smooth takes quite a bit of time! Todo< make dynamic for longer recordings!
    input:
        zscore_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_1_zscore.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_2_zscore.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_3_zscore.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    output:
        temp_HP_filter_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_1_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        temp_HP_filter_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_2_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        temp_HP_filter_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/NO_MOCO/channel_3_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],

    run:
        try:
            preprocessing.temporal_high_pass_filter(fly_directory=fly_folder_to_process_oak,
                dataset_path=[input.zscore_path_ch1,
                              input.zscore_path_ch2,
                              input.zscore_path_ch3],
                temporal_high_pass_filtered_path=[output.temp_HP_filter_path_ch1,
                                                  output.temp_HP_filter_path_ch2,
                                                  output.temp_HP_filter_path_ch3])
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_temporal_high_pass_filter')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule correlation_rule:
    """

    """
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input,
        runtime='60m'  # vectorization made this super fast
    input:
        corr_path_ch1=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/NO_MOCO/channel_1_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        corr_path_ch2=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/NO_MOCO/channel_2_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        corr_path_ch3=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/NO_MOCO/channel_3_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
        fictrac_path=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/fictrac/fictrac_behavior_data.dat",
        metadata_path=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/imaging/recording_metadata.xml"
    output:
        save_path_ch1=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/NO_MOCO/corr/channel_1_corr_{corr_behavior}.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        save_path_ch2=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/NO_MOCO/corr/channel_2_corr_{corr_behavior}.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        save_path_ch3=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/NO_MOCO/corr/channel_3_corr_{corr_behavior}.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    run:
        try:
            preprocessing.correlation(fly_directory=fly_folder_to_process_oak,
                dataset_path=[input.corr_path_ch1, input.corr_path_ch2, input.corr_path_ch3],
                save_path=[output.save_path_ch1, output.save_path_ch2, output.save_path_ch3],
                #behavior=input.fictrac_path,
                fictrac_fps=fictrac_fps,
                metadata_path=input.metadata_path,
                fictrac_path=input.fictrac_path,

            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_correlation')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)