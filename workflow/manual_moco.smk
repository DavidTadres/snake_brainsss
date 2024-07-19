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

scripts_path = workflow.basedir # Exposes path to this file


current_user = config['user'] # this is whatever is entered when calling snakemake, i.e.
# snakemake --profile profiles/simple_slurm -s snaketest.smk --config user=jcsimon would
# yield 'jcsimon' here
settings = utils.load_user_settings(current_user)
moco_temp_folder = str(settings.get('moco_temp_folder', "/scratch/groups/trc"))

# User provided!
main_channel = config['main_channel']
mirror_channel = config['mirror_channel']

print('main channel: ' + repr(main_channel))

# Define path to imports to find fly.json!
#fly_folder_to_process_oak = pathlib.Path(dataset_path,fly_folder_to_process)
fly_folder_to_process_oak = pathlib.Path(os.getcwd())
print('Analyze data in ' + repr(fly_folder_to_process_oak.as_posix()))