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

# YOUR SUNET ID
current_user = 'dtadres'

# fps of fictrac recording!
fictrac_fps = 50

import pathlib
from scripts import preprocessing
import brainsss
import sys
import json
import datetime

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
    print(new_fly_number)
    fly_folder_to_process = pathlib.Path(dataset_path, 'fly_' + new_fly_number)
else:
    fly_folder_to_process = pathlib.Path(dataset_path, fly_folder_to_process)
print(fly_folder_to_process)

#####
# LOGGING
#####
pathlib.Path(fly_folder_to_process, '/logs').mkdir(exist_ok=True)
# Have one log file per fly! This will make everything super traceable!
logfile = str(fly_folder_to_process) + '/logs/' + fly_folder_to_process.name + '.txt'

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

# Files needed for snakemake to create the DAG graph
# This one is created by fly_builder_rule
fly_dirs_dict_path = pathlib.Path(fly_folder_to_process, fly_folder_to_process.name + '_dirs.json')
# Make a folder with todays date in the 'io_files' folder which will be used by
# snakemake to track which rule has been run already
todays_folder = pathlib.Path('./io_files', datetime.datetime.now().strftime("%Y%m%d"))
todays_folder.mkdir(exist_ok=True, parents=True)
io_fictrac_qc_path = pathlib.Path(todays_folder, 'fictrac_qc.txt')
io_bleaching_qc_path = pathlib.Path(todays_folder, 'bleaching_qc.txt')
'''
from scripts import hello_world
rule all:
    """
    This rule is necessary for building the DAG graph (workflow path). 
    To run more than one rule, one needs to define the order using the
    input and output 
    """
    threads: 1
    #input: 'HelloSnake2.txt'
    input:

        # io_fictrac_qc_path
        #io_bleaching_qc_path
    output:
        str(todays_folder) + "fictrac_qc_done_{group}.txt"'''
'''rule HelloSnake:
    #shell:
    #    'python3 hello_world.py $args'
    threads: 2
    output: filename = "HelloSnake.txt"
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
                                 error_stack=error_stack)

        with open(output.filename,"w") as out:
            out.write('HelloSnakes')

rule HelloSnake2:
    threads: 2
    input: "HelloSnake.txt"
    output: filename = 'HelloSnake2.txt'
    run:
        hello_world.print_bye(logfile=logfile, args = 'Stanford')

        with open(output.filename,"w") as out:
            out.write('HelloSnakes')

'''

# only run the fly_builder_rule if the folder defined as the target folder
# does not exist yet
rule fly_builder_rule:
    """
    Threads: Tested 1, 2, 4 and 16 threads for a folder with 1 fly and 3 func folders
    I did not see an improvement in run time increasing the thread number above 2. 
    """
    output: fly_dirs_dict_path
    threads: 1
    run:

        # Only run the code to copy data from imports to 'fly_00X' if the
        # user defined fly folder does not exist yet.
        # Note: it seems that the fly folder is somehow created
        if not pathlib.Path(fly_folder_to_process, 'anat').exists() or \
                not pathlib.Path(fly_folder_to_process, 'func').exists:
            try:
                fly_dirs_dict = preprocessing.fly_builder(logfile=logfile,
                                                         user=current_user,
                                                         dirs_to_build=data_to_process,
                                                         target_folder = fly_folder_to_process
                                                         )
                print(fly_dirs_dict)
                # Json file containing the paths created by fly_builder
                with open(fly_dirs_dict_path,'w') as file:
                    json.dump(fly_dirs_dict,file,sort_keys=True,indent=4)

                brainsss.print_function_done(logfile, width, 'fly_builder')
            except Exception as error_stack:
                brainsss.write_error(logfile=logfile,
                    error_stack=error_stack,
                    width=width)
        else:
            print('Folder ' + fly_folder_to_process.name + ' already exists. Skipping fly builder')
            # Need to load dir_dict to get relative paths of all files of the fly to be processed.
            with open(pathlib.Path(fly_dirs_dict_path),'r') as file:
                fly_dirs_dict = json.load(file)
            # Probably need to overwrite because else snakemake won't be able to continue
            with open(fly_dirs_dict_path,'w') as file:
                json.dump(fly_dirs_dict,file,sort_keys=True,indent=4)
'''
rule fictrac_qc_rule:
    """
    Tested only 1 thread and was super fast. To be tested with more!
    
    This would lend itself to parallelization as the calculation of each folder
    does not what's happening in the other folders. 
    
    """
    input: fly_dirs_dict_path
    output: str(todays_folder) + "fictrac_qc_{group}.txt" # io_fictrac_qc_path
    threads: 1
    run:
        print('fictrac_qc rule')
        try:
            preprocessing.fictrac_qc(logfile=logfile,
                                     directory=fly_folder_to_process,
                                     fictrac_fps=fictrac_fps)
            with open(pathlib.Path(str(todays_folder) + "fictrac_qc_{group}.txt"), 'w') as file:
                file.write('fictrac_qc rule finished successfully')
            brainsss.print_function_done(logfile, width, 'fictrac_qc')
        except Exception as error_stack:
            brainsss.write_error(logfile=logfile,
                                 error_stack=error_stack,
                                 width=width)'''
'''
rule bleaching_qc_rule:
    """
    """
    input: fly_dirs_dict_path
    output: io_bleaching_qc_path
    threads: 16
    run:
        print("bleaching_qc rule")
        try:
            preprocessing.bleaching_qc(logfile=logfile,
                                       directory=fly_folder_to_process,
                                       fly_dirs_dict = fly_dirs_dict)
            with open(io_bleaching_qc_path, 'w') as file:
                file.write('bleaching_qc rule finished successfully')
            brainsss.print_function_done(logfile, width, 'bleaching_qc')
        except Exception as error_stack:
            brainsss.write_error(logfile=logfile,
                                 error_stack=error_stack,
                                 width=width)'''
