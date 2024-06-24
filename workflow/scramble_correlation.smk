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

scripts_path = workflow.basedir # Exposes path to this file

# the user needs to of course define the --directory which
# then puts this file into that path
print(os.getcwd()) # the user defined --directory

# Behaviors to correlate with neural activity
corr_behaviors = ['dRotLabZneg'
                  ,'dRotLabZpos'
                  #,'dRotLabY'
                  ]

directory = pathlib.Path(os.getcwd())
# TESTING
# Works for both a genotype and a fly.
# directory = pathlib.Path('//Volumes//trc//data//David//Bruker//preprocessed//FS152_x_FS69')

list_of_corr_paths = []
# This is analog to the variable 'list_of_paths_func' variable in preprocess_fly.smk

# Currently we can only do a whole genotype if the IDENTICAL functional
# channels are used throughout. Else, call script on each experiment individually.
FUNCTIONAL_CHANNELS = None
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
                    if FUNCTIONAL_CHANNELS is None:
                        # This needs to come from some sort of json file the experimenter
                        # creates while running the experiment. Same as genotype.
                        FUNCTIONAL_CHANNELS = fly_json['functional_channel']
                    else:
                        if FUNCTIONAL_CHANNELS == fly_json['functional_channel']:
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
        search_for_corr(current_folder,FUNCTIONAL_CHANNELS)

# Now we have a list with all folder that have a correlation folder which
# can now be used to create rules!
print('list of found folders')
print(list_of_corr_paths)

"""        expand("/{corr_imaging_paths}/corr/channel_1_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
               corr_imaging_paths=list_of_corr_paths, corr_behavior=corr_behaviors),
        expand("{corr_imaging_paths}/corr/channel_2_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
               corr_imaging_paths=list_of_corr_paths, corr_behavior=corr_behaviors),
        expand("{corr_imaging_paths}/corr/channel_3_corr_{corr_behavior}_SCRAMBLED.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
               corr_imaging_paths=list_of_corr_paths, corr_behavior=corr_behaviors),
"""
# FOR DEBUGGING ONLY!!!
list_of_corr_paths = ['/oak/stanford/groups/trc/data/David/Bruker/preprocessed/FS144_x_FS69/fly_007/func0']
# ******
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
        expand('{corr_imaging_paths}/corr/channel_1_corr_SCRAMBLED.nii' if 'channel_1' in FUNCTIONAL_CHANNELS else[],
            corr_imaging_paths=list_of_corr_paths )


rule scramble_correlation_rule:
    threads: snake_utils.threads_per_memory_less
    resources:
        runtime='60m' # vectorization made this super fast
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
        scramble_correlation.calculate_scrambled_correlation(
            fictrac_path=input.fictrac_path,
            fictrac_fps=fictrac_fps,
            metadata_path=input.metadata_path,
            moco_zscore_highpass_path=[input.corr_path_ch1, input.corr_path_ch2, input.corr_path_ch3],
            save_path=[output.savepath_ch1, output.savepath_ch2, output.savepath_ch3]
        )