"""
An important control of the data coming out of the
correlation developed for brainsss is a scrambled
correlation.

The idea of this snakefile is that it can be run after
doing preprocessing on either a whole genotype or just
a folder.

Ideally it would only try to perform the scrambled
correlation in folders where the original correlation
already exists!

"""

import pathlib
import os
import json
from scripts import snake_utils
from scripts import utils
from analysis_scripts import scramble_correlation

scripts_path = workflow.basedir # Exposes path to this file

# the user needs to of course define the --directory which
# then puts this file into that path
print(os.getcwd()) # the user defined --directory

# Behaviors to correlate with neural activity
corr_behaviors = ['dRotLabZneg'
                  ,'dRotLabZpos'
                  #,'dRotLabY'
                  ]
fictrac_fps = 100 # TO BE CHANGED TO A DYNAMIC VERSION THAT READS ACTUAL FPS!!!

directory = pathlib.Path(os.getcwd())
# TESTING
# Works for both a genotype and a fly.
# directory = pathlib.Path('//Volumes//trc//data//David//Bruker//preprocessed//FS152_x_FS69')

list_of_corr_paths = []
# This is analog to the variable 'list_of_paths_func' variable in preprocess_fly.smk

# Currently we can only do a whole genotype if the IDENTICAL functional
# channels are used throughout. Else, call script on each experiment individually.
# To do this we use a 'trick' in python to keep track of a variable in a recursive
# function without 'return': we use a global list and append to it.
FUNCTIONAL_CHANNELS = []
def search_for_corr(folder, FUNCTIONAL_CHANNELS):
    #print(folder.name)
    for current_folder in folder.iterdir():
        if current_folder.is_dir():
            if 'corr' in current_folder.name:
                # If we find a folder named 'corr' we want the
                # **parent** folder **as a string**
                list_of_corr_paths.append(current_folder.parent.as_posix())
                #print("all_corr_paths" + repr(list_of_corr_paths))

                # Read channel information from fly.json file
                # If fails here, means the folder specified doesn't exist. Check name.
                # Note: Good place to let user know to check folder and exit!
                with open(pathlib.Path(current_folder.parent.parent,'fly.json'),'r') as file:
                    fly_json = json.load(file)
                    if len(FUNCTIONAL_CHANNELS) == 0:
                        # This needs to come from some sort of json file the experimenter
                        # creates while running the experiment. Same as genotype.
                        FUNCTIONAL_CHANNELS.append(fly_json['functional_channel'])
                    else:
                        if FUNCTIONAL_CHANNELS[0] == fly_json['functional_channel']:
                            pass
                        else:
                            print('Folder ' + current_folder.parent.as_posix() + ' has different settings for '
                                 'functional_channel compared to a previous fly!\n')
                            print('Please run snakemake on each fly individually')

                            import sys
                            sys.exit(1)
                return
            else:
                search_for_corr(current_folder, FUNCTIONAL_CHANNELS)

for current_folder in directory.iterdir():
    if current_folder.is_dir():
        search_for_corr(current_folder, FUNCTIONAL_CHANNELS)

# For some reason we get a list which contains: ["['channel_2']"]
# which seems to throw False when tested for 'channel_2.
# Flatten the list in the for loop below
temp_func = []
for i in FUNCTIONAL_CHANNELS:
    temp_func.append(i.split("['")[-1].split("']")[0])
FUNCTIONAL_CHANNELS = temp_func


# Now we have a list with all folder that have a correlation folder which
# can now be used to create rules!
print('list of found folders')
print(list_of_corr_paths)
# Helps with debugging
print("\nFUNCTIONAL_CHANNELS")
print(FUNCTIONAL_CHANNELS)

print("\ncorr_behaviors")
print(corr_behaviors)
print("\n")

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
        # Scramble correlation with fictrac behavior
        ###
        expand("/{corr_imaging_paths}/corr/channel_1_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
            corr_imaging_paths=list_of_corr_paths,corr_behavior=corr_behaviors),
        expand("{corr_imaging_paths}/corr/channel_2_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
            corr_imaging_paths=list_of_corr_paths,corr_behavior=corr_behaviors),
        expand("{corr_imaging_paths}/corr/channel_3_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
            corr_imaging_paths=list_of_corr_paths,corr_behavior=corr_behaviors),


rule scramble_correlation_rule:
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb = snake_utils.mem_mb_less_times_input,
        runtime='20m' # vectorization made this super fast
    input:
        corr_path_ch1="{corr_imaging_paths}/channel_1_moco_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        corr_path_ch2="{corr_imaging_paths}/channel_2_moco_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        corr_path_ch3="{corr_imaging_paths}/channel_3_moco_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else[],
        fictrac_path="{corr_imaging_paths}/stimpack/loco/fictrac_behavior_data.dat",
        metadata_path="{corr_imaging_paths}/imaging/recording_metadata.xml"

    output:
        savepath_ch1="{corr_imaging_paths}/corr/channel_1_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        savepath_ch2="{corr_imaging_paths}/corr/channel_2_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        savepath_ch3="{corr_imaging_paths}/corr/channel_3_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else[]
    run:
        try:
            scramble_correlation.calculate_scrambled_correlation(
                fictrac_path=input.fictrac_path,
                fictrac_fps=fictrac_fps,
                metadata_path=input.metadata_path,
                moco_zscore_highpass_path=[input.corr_path_ch1, input.corr_path_ch2, input.corr_path_ch3],
                save_path=[output.savepath_ch1, output.savepath_ch2, output.savepath_ch3]
            )
        except Exception as error_stack:
            # Here I need to make an assumption:
            # The folder structure must be
            # -genotype
            #   - fly_001
            #       - logs
            #       - func0
            #           - corr
            #           - imaging
            # Get parent folder of 'corr' which points to 'fly_001
            fly_log_folder = pathlib.Path(input.metadata_path).parent.parent
            logfile = utils.create_logfile(fly_log_folder,function_name='ERROR_scramble_correlation')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)