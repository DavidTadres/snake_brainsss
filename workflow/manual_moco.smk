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

# On sherlock this is usually python3 but on a personal computer can be python
shell_python_command = str(settings.get('shell_python_command', "python3"))
print("shell_python_command" + shell_python_command)
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
MAIN_CHANNEL = config['main_channel']
MIRROR_CHANNEL = config['mirror_channel']

CH1_EXISTS = False
CH2_EXISTS = False
CH3_EXISTS = False
# Make sanity check for correct input:
if 'channel_1' in MAIN_CHANNEL:
    all_channels = ['1']
    CH1_EXISTS = True
elif 'channel_2' in MAIN_CHANNEL:
    all_channels = ['2']
    CH2_EXISTS = True
elif 'channel_3' in MAIN_CHANNEL:
    all_channels = ['3']
    CH3_EXISTS = True

if 'channel_1' in MIRROR_CHANNEL and 'channel_1' not in MAIN_CHANNEL:
    all_channels.append('1')
    CH1_EXISTS = True
if 'channel_2' in MIRROR_CHANNEL and 'channel_2' not in MAIN_CHANNEL:
    all_channels.append('2')
    CH2_EXISTS = True
if 'channel_3' in MIRROR_CHANNEL and 'channel_3' not in MAIN_CHANNEL:
    all_channels.append('3')
    CH3_EXISTS = True

# First n frames to average over when computing mean/fixed brain | Default None
# (average over all frames).
meanbrain_n_frames =  None

rule all:
    threads: 1 # should be sufficent
    resources: mem_mb=1000 # should be sufficient
    input:
        expand(str(fly_folder_to_process_oak)
               + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch}_mean_manual.nii",
            meanbr_imaging_paths_func=list_of_paths_func,
            meanbr_ch=all_channels),

        ###
        # Motion correction output FUNC
        # The idea is to use the structural channel for moco!
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/motcorr_params_manual.npy",
               moco_imaging_paths_func=list_of_paths_func),

        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/channel_1_moco_manual.nii" if CH1_EXISTS else [],
               moco_imaging_paths_func=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/channel_2_moco_manual.nii" if CH2_EXISTS else [],
               moco_imaging_paths_func=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/channel_3_moco_manual.nii" if CH3_EXISTS else [],
               moco_imaging_paths_func=list_of_paths_func),

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
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch}.nii"
    output:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch}_mean_manual.nii"
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

rule motion_correction_parallel_processing_rule_func:
    """
    OOM using an anat folder! :
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 32
    CPU Utilized: 1-08:48:23
    CPU Efficiency: 60.01% of 2-06:40:00 core-walltime
    Job Wall-clock time: 01:42:30
    Memory Utilized: 117.43 GB
    Memory Efficiency: 99.69% of 117.79 GB
    And another OOM for an anat 
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 32
    CPU Utilized: 1-11:00:39
    CPU Efficiency: 62.09% of 2-08:23:28 core-walltime
    Job Wall-clock time: 01:45:44
    Memory Utilized: 115.60 GB
    Memory Efficiency: 98.14% of 117.79 GB

    Same settings, func super happy
    Nodes: 1
    Cores per node: 32
    CPU Utilized: 16:06:07
    CPU Efficiency: 67.37% of 23:54:08 core-walltime
    Job Wall-clock time: 00:44:49
    Memory Utilized: 24.06 GB
    Memory Efficiency: 32.56% of 73.90 GB

    -> There's clearly a huge discrepancy of memory used based on the resolution of the image.
    Ideally I could set the memory depending on the resolution of the image instead of the size of the file...

    To speed motion correction up, use the multiprocessing module. This requires the target to 
    be a module (not just a function). Hence we have a 'shell' directive here.
    """
    threads: 32  # the max that we can do - check with sh_part
    resources:
        mem_mb=snake_utils.mb_for_moco_input,#.mem_mb_much_more_times_input,
        runtime=snake_utils.time_for_moco_input  # runtime takes input as seconds!
    input:
        # Only use the Channels that exists - this organizes the anatomy and functional paths inside the motion correction
        # module.
        brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_1.nii" if CH1_EXISTS else [],
        brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_2.nii" if CH2_EXISTS else [],
        brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_3.nii" if CH3_EXISTS else [],

        mean_brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_1_mean_manual.nii" if CH1_EXISTS else [],
        mean_brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_2_mean_manual.nii" if CH2_EXISTS else [],
        mean_brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_3_mean_manual.nii" if CH3_EXISTS else []
    output:
        moco_path_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/channel_1_moco_manual.nii" if CH1_EXISTS else [],
        moco_path_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/channel_2_moco_manual.nii" if CH2_EXISTS else [],
        moco_path_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/channel_3_moco_manual.nii" if CH3_EXISTS else [],
        par_output=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/motcorr_params_manual.npy"

    shell: shell_python_command + " " + scripts_path + "/scripts/moco_parallel.py "
                                                       "--fly_directory {fly_folder_to_process_oak} "
                                                       "--dataset_path {dataset_path} "
                                                       "--brain_paths_ch1 {input.brain_paths_ch1} "
                                                       "--brain_paths_ch2 {input.brain_paths_ch2} "
                                                       "--brain_paths_ch3 {input.brain_paths_ch3} "
                                                       "--mean_brain_paths_ch1 {input.mean_brain_paths_ch1} "
                                                       "--mean_brain_paths_ch2 {input.mean_brain_paths_ch2} "
                                                       "--mean_brain_paths_ch3 {input.mean_brain_paths_ch3} "
                                                       "--STRUCTURAL_CHANNEL {MAIN_CHANNEL} "
                                                       "--FUNCTIONAL_CHANNELS {MIRROR_CHANNEL} "
                                                       "--moco_path_ch1 {output.moco_path_ch1} "
                                                       "--moco_path_ch2 {output.moco_path_ch2} "
                                                       "--moco_path_ch3 {output.moco_path_ch3} "
                                                       "--par_output {output.par_output} "
                                                       "--moco_temp_folder {moco_temp_folder} "