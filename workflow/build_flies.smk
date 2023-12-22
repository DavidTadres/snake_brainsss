# with config file type:
# ml python/3.9.0
# source .env_snakemake/bin/activate
# cd snake_brainsss/workflow
# snakemake -s build_flies.smk --profile OLDconfig_sherlock

# config file from: https://github.com/jdblischak/smk-simple-slurm/tree/main

''' this one worked"
snakemake -s build_flies.smk --jobs 1 --cluster 'sbatch --partition trc --cpus-per-task 16 --ntasks 1 --mail-type=ALL'
# This one with config
snakemake -s build_flies.smk --profile
'''
'''
Can also only run a given rule:
snakemake -s build_flies.smk stitch_split_nii --jobs 1 --cluster 'sbatch --partition trc --cpus-per-task 16 --ntasks 1 --mail-type=ALL'
'''
# To be modified by the user
imports_to_process = ['20231207',] # Data deposited by Brukerbridge on oak
# Only needed to run :fly_builder_rule:

#fly_folder_to_process = '' # if already copied to 'fly_00X' folder and only
# do follow up analysis, enter the fly folder to be analyzed here.
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly

# YOUR SUNET ID
current_user = 'dtadres'

# fps of fictrac recording!
#fictrac_fps = 50

import pathlib
from scripts import preprocessing
import shutil
import sys
import json
import datetime
from brainsss import utils


settings = utils.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])
imports_path = pathlib.Path(settings['imports_path'])

all_imports_paths = []
all_fly_dataset_paths = []

# Folders to be created for data to be written to
for current_day in imports_to_process:
    print('Working on folder ' + repr(current_day))
    # Make it pathlib.Path
    current_day_path = pathlib.Path(imports_path, current_day)
    for current_fly in current_day_path.iterdir():
        # There should be 'fly' folders in here, i.e. 'fly1' or similar, just needs to have the keyword fly and be a folder
        if 'fly' in current_fly.name and current_fly.is_dir():
            print('Working on fly: ' + repr(current_fly.name))
            # Check if the fly has already been transferred:
            already_transfered = False
            for current_file in current_fly.iterdir():
                if 'data_transfered_to.txt' in current_file.name:
                    already_transfered = True
                    print('Data already transfered. Delete "data_transfered_to.txt" file to re-transfer')

            if not already_transfered:

                GENOTYPE = None # in case it's missing in the second or third fly
                fly_json_path = pathlib.Path(current_fly, 'fly.json') # if not present will throw error
                with open(fly_json_path,'r') as openfile:
                    fly_json = json.load(openfile)
                    GENOTYPE = fly_json['genotype']  # Must be present, else error

                # Since we have now different genotypes, will have to create new folders. Check if this is the first
                # fly with this genotype during this run.
                first_fly_with_genotype_this_run = True
                # For the list of already created all_fly_dataset_path, check if the genotype is present in any of them
                for current_path in all_fly_dataset_paths:
                    if GENOTYPE in current_path.parts:
                        # If yes, indicate that we can just use previously created folders to create new fly numbers
                        first_fly_with_genotype_this_run = False
                # Get a new fly number.
                new_fly_number = utils.get_new_fly_number(target_path = pathlib.Path(dataset_path, GENOTYPE),
                                                          first_fly_with_genotype_this_run = first_fly_with_genotype_this_run,
                                                          already_created_folders=all_fly_dataset_paths)

                current_fly_dataset_folder = pathlib.Path(dataset_path, GENOTYPE, 'fly_' + new_fly_number)
                # Must create new folder here so that in the next loop we get the correct fly_XXX back
                current_fly_dataset_folder.mkdir(exist_ok=True,parents=True)
                # Create a file named 'incomplete' to clearly indicate when fly building has not finished yet.
                with open(pathlib.Path(current_fly_dataset_folder, 'incomplete'),"w") as f:
                    f.write("")

                if current_fly_dataset_folder.name == 'fly_001':
                    # If this is the first time a fly of a given genotype is used, copy the master_2P.xlsx file from this
                    # repository into the folder
                    shutil.copy(workflow.source_path('master_2P.xlsx'), current_fly_dataset_folder.parent)
                # Collect all paths in list
                all_fly_dataset_paths.append(current_fly_dataset_folder)
                all_imports_paths.append(current_fly)


#print('all_fly_dataset_paths' + repr(all_fly_dataset_paths))
#print('all_imports_paths' + repr(all_imports_paths))
# only run the fly_builder_rule if the folder defined as the target folder
# does not exist yet
rule fly_builder_rule:
    """
    Not parallelized right now.
    """
    threads: 1
    run:

        # Only run the code to copy data from imports to 'fly_00X' if the
        # user defined fly folder does not exist yet.
        # Note: it seems that the fly folder is somehow created
        #if not pathlib.Path(fly_folder_to_process, 'anat').exists() or \
        #        not pathlib.Path(fly_folder_to_process, 'func').exists:

        preprocessing.fly_builder(user=current_user,
                                  import_dirs= all_imports_paths,
                                  dataset_dirs = all_fly_dataset_paths
                                                 )