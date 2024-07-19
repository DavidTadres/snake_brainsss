"""
Ironically, GCaMP7b is brighter than tdTomato and it is easier to perform moco with GCaMP7b in sparsely labelled
lines.

There is no straightforward way of telling moco to use a functional channel as 'main' and the structural channels as
'mirror' in moco.

This snakefile should allow the user to manually define which channel should be used for main and which one to mirror!
"""
import pathlib
import os
from brainsss import utils
from scripts import snake_utils
from scripts import preprocessing
import json

scripts_path = workflow.basedir # Exposes path to this file


current_user = config['user'] # this is whatever is entered when calling snakemake, i.e.
# snakemake --profile profiles/simple_slurm -s snaketest.smk --config user=jcsimon would
# yield 'jcsimon' here
settings = utils.load_user_settings(current_user)
moco_temp_folder = str(settings.get('moco_temp_folder', "/scratch/groups/trc"))


# Define path to imports to find fly.json!
#fly_folder_to_process_oak = pathlib.Path(dataset_path,fly_folder_to_process)
fly_folder_to_process_oak = pathlib.Path(os.getcwd())
print('Analyze data in ' + repr(fly_folder_to_process_oak.as_posix()))
####
# Load fly_dir.json
####
fly_dirs_dict_path = pathlib.Path(fly_folder_to_process_oak, fly_folder_to_process_oak.name + '_dirs.json')
with open(pathlib.Path(fly_dirs_dict_path),'r') as file:
    fly_dirs_dict = json.load(file)
# Imaging data paths
imaging_file_paths = []
fictrac_file_paths = []
for key in fly_dirs_dict:
    if 'Imaging' in key:
        imaging_file_paths.append(fly_dirs_dict[key][1::])
        # this yields for example 'func2/imaging'
    elif 'Fictrac' in key:
        fictrac_file_paths.append(fly_dirs_dict[key][1::])
list_of_paths_func = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        list_of_paths_func.append(current_path.split('/imaging')[0])

# User provided!
main_channel = config['main_channel']
mirror_channel = config['mirror_channel']

# Make sanity check for correct input:
if 'channel_1' in main_channel or 'channel_2' in main_channel or 'channel_3' in main_channel:
    all_channels=[main_channel]
if 'channel_1' in mirror_channel and 'channel_1' not in main_channel:
    all_channels.append('channel_1')
if 'channel_2' in mirror_channel and 'channel_2' not in main_channel:
    all_channels.append('channel_2')
if 'channel_3' in mirror_channel and 'channel_3' not in main_channel:
    all_channels.append('channel_3')


# First n frames to average over when computing mean/fixed brain | Default None
# (average over all frames).
meanbrain_n_frames =  None

rule all:
    threads: 1 # should be sufficent
    resources: mem_mb=1000 # should be sufficient
    input:
        expand(str(fly_folder_to_process_oak)
               + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch}_mean_func.nii",
            meanbr_imaging_paths_func=list_of_paths_func,
            meanbr_ch=all_channels),

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
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch_func}.nii"
    output:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch_func}_mean.nii"
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
                error_stack=error_stack)