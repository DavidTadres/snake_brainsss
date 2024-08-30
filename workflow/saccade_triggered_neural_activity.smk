import pathlib
import os
from brainsss import utils
from scripts import snake_utils
from scripts import preprocessing
import json
from analysis_scripts import saccade_triggered_activity

scripts_path = workflow.basedir # Exposes path to this file

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

CH1_FUNC = False
CH2_FUNC = False
CH3_FUNC = False
if 'channel_1' in FUNCTIONAL_CHANNELS:
    CH1_FUNC=True
if 'channel_2' in FUNCTIONAL_CHANNELS:
    CH2_FUNC=True
if 'channel_3' in FUNCTIONAL_CHANNELS:
    CH3_FUNC = True
#####
# Snakemake runs acyclical, meaning it checks which input depends on the output of which rule
# in order to parallelize a given snakefile.
# I'll therefore keep variables with lists of paths that will be feed into a given rule.
# These lists of paths are created here.
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
        # This yields for example 'func1/fictrac/fictrac_behavior_data.dat'
        # With automatic stimpack transfer it'll return "/func0/stimpack/loco/fictrac_behavior_data.dat"

list_of_paths_func = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        list_of_paths_func.append(current_path.split('/imaging')[0])

rule all:
    threads: 1 # should be sufficent
    resources: mem_mb=1000 # should be sufficient
    input:
        expand(str(fly_folder_to_process_oak)
        + "/{saccade_imaging_paths}/sac_trig_activity/channel_1_{turn_side}_sac_trig_act.nii" if CH1_FUNC else [],
        saccade_imaging_paths=list_of_paths_func,
        turn_side=['L', 'R']),

        expand(str(fly_folder_to_process_oak)
               + "/{saccade_imaging_paths}/sac_trig_activity/channel_2_{turn_side}_sac_trig_act.nii" if CH2_FUNC else [],
               saccade_imaging_paths=list_of_paths_func,
               turn_side=['L', 'R']),

        expand(str(fly_folder_to_process_oak)
               + "/{saccade_imaging_paths}/sac_trig_activity/channel_3_{turn_side}_sac_trig_act.nii" if CH3_FUNC else [],
               saccade_imaging_paths=list_of_paths_func,
               turn_side=['L', 'R'])


rule sac_trig_activity:
    threads: snake_utils.threads_per_memory_much_more
    resources:
        mem_mb = snake_utils.mem_mb_much_more_times_input,
        runtime='30m'
    input:
        fictrac_path = str(fly_folder_to_process_oak) + "/{saccade_imaging_paths}/stimpack/loco/fictrac_behavior_data.dat",
        brain_path = str(fly_folder_to_process_oak) + "/{saccade_imaging_paths}/imaging/channel_{func_ch}.nii",
        metadata_path=str(fly_folder_to_process_oak) + "/{saccade_imaging_paths}/imaging/recording_metadata.xml"
    output:
        savepath = str(fly_folder_to_process_oak) + "/{saccade_imaging_paths}/sac_trig_activity/channel_{func_ch}_{turn_side}_sac_trig_act.nii"
    run:
        try:
            saccade_triggered_activity.calc_sac_trig_activity(
                fly_folder_to_process_oak = fly_folder_to_process_oak,
                fictrac_path = input.fictrac_path,
                brain_path = input.brain_path,
                metadata_path = input.metadata_path,
                savepath = output.savepath

            )

        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_sac_trig_act')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)
