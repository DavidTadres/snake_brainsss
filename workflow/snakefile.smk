# with config file type:
# ml python/3.9.0
# source .env_snakemake/bin/activate
# cd snake_brainsss/workflow
# snakemake -s snakefile.smk --profile config_sherlock

''' this one worked"
snakemake -s snakefile.smk --jobs 1 --cluster 'sbatch --partition trc --cpus-per-task 16 --ntasks 1 --mail-type=ALL'
# This one with config
snakemake -s snakefile.smk --profile 
'''
'''
Can also only run a given rule:
snakemake -s snakefile.smk stitch_split_nii --jobs 1 --cluster 'sbatch --partition trc --cpus-per-task 16 --ntasks 1 --mail-type=ALL'
'''
# To be modified by the user
data_to_process = ['20231207'] # Data deposited by Brukerbridge on oak
# Only needed to run :fly_builder_rule:

fly_folder_to_process = '' # if already copied to 'fly_00X' folder and only
# do follow up analysis, enter the fly folder to be analyzed here.
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly

import pathlib
from scripts import preprocessing
import brainsss
import sys
import pyfiglet

# YOUR SUNET ID
current_user = 'dtadres'

# David's datapaths
#original_data_path = '/Volumes/groups/trc/data/David/Bruker/imports' # Mac
#original_data_path = '/oak/stanford/groups/trc/data/David/Bruker/imports' # Mac
#target_data_path = '/Volumes/groups/trc/data/David/Bruker/preprocessed'
#current_fly = pathlib.Path(original_data_path, '20231207__queue__')

settings = brainsss.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])
imports_path = pathlib.Path(settings['imports_path'])

# Idea - to make the pipeline as traceable as possible, make one logfile
# per fly. When coming back in the future, it's easy to read all the output
# done for each fly
# for logging

if fly_folder_to_process == '':
    # If data has not yet been copied to the target folder,
    new_fly_number = brainsss.get_new_fly_number(dataset_path)
    fly_folder_to_process = pathlib.Path(dataset_path, 'fly_' + new_fly_number)
else:
    fly_folder_to_process = pathlib.Path(dataset_path, fly_folder_to_process)
#print(fly_folder_to_process)

# Problem: Snakemake runs twice. Seems to be a bug:
#https://github.com/snakemake/snakemake/issues/2350
# solution: ignore seconds
#logfile = './logs/' + time.strftime("%Y%m%d-%H%M00") + '.txt'
#####
# LOGGING
#####
pathlib.Path('./logs').mkdir(exist_ok=True)
# Have one log file per fly! This will make everything super traceable!
logfile = './logs/' + fly_folder_to_process.name + '.txt'

#
printlog = getattr(brainsss.Printlog(logfile=logfile),'print_to_log')
sys.stderr = brainsss.LoggerRedirect(logfile)
sys.stdout = brainsss.LoggerRedirect(logfile)
if not pathlib.Path(logfile).is_file():
    width = 120
    brainsss.print_title(logfile, width)
    fly_string = pyfiglet.figlet_format(fly_folder_to_process.name, font="doom" )
    fly_string_shifted = ('\n').join([' ' * 42 + line for line in fly_string.split('\n')][:-2])
    printlog(fly_string, width)
    brainsss.print_datetime(logfile, width)
'''
from scripts import hello_world

rule HelloSnake:
    #shell:
    #    'python3 hello_world.py $args'
    threads: 2
    run:
        brainsss.print_datetime(logfile,width)
        print('\nExecuting HelloSnake rule\n')
        try:
            hello_world.print_hi(logfile=logfile,
                       args='world',
                        arg2='test2'
                        )
        except Exception as error_stack:
            brainsss.write_error(logfile=logfile,
                                 error_stack=error_stack)'''

rule fly_builder_rule:
    threads: 16
    run:
        try:
            preprocessing.fly_builder(logfile=logfile,
                user=current_user,
                dirs_to_build=data_to_process,
                target_folder = fly_folder_to_process
            )
        except Exception as error_stack:
            brainsss.write_error(logfile=logfile,
                error_stack=error_stack)



