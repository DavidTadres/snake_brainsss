## Testing ##
"""
snakemake \
    --s snakefile.smk \
    --jobs 1 \
    --default-resource mem_mb=100\
    --cluster '
        sbatch \
            --partition trc
            --cpus-per-task 16
            --ntasks 1
            --output ./logs/stitchlog3.out
            --open-mode append
            --mail-type=ALL
        ` \
        output.txt
"""

''' this one worked"
snakemake -s snakefile.smk --jobs 1 --cluster 'sbatch --partition trc --cpus-per-task 16 --ntasks 1 --mail-type=ALL'
'''
'''
Can also only run a given rule:
snakemake -s snakefile.smk stitch_split_nii --jobs 1 --cluster 'sbatch --partition trc --cpus-per-task 16 --ntasks 1 --mail-type=ALL'
'''

"""
import hello_world

rule HelloSnake:
    #shell:
    #    "python3 hello_world.py"
    run:
        hello_world.print_hi(args='test', arg2='test2')

"""

import pathlib

# YOUR SUNET ID
current_user = 'dtadres'

# David's datapaths
original_data_path = '/oak/stanford/groups/trc/data/David/Bruker/imports'
#target_data_path = '/Volumes/groups/trc/data/David/Bruker/preprocessed'

current_fly = pathlib.Path(original_data_path, '20231201')
fly_folder_to_process = '' # if already copied to 'fly_00X' folder and only
# do follow up analysis, enter the fly folder to be analyzed here.
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly
print(current_fly)
from workflow.scripts import hello_world

# Idea - to make the pipeline as traceable as possible, make one logfile
# per fly. When coming back in the future, it's easy to read all the output
# done for each fly
# for logging
from workflow import brainsss
import sys

scripts_path = pathlib.Path(__file__).parent.resolve()
print(scripts_path)
settings = brainsss.load_user_settings(current_user,scripts_path)
dataset_path = pathlib.Path(settings['dataset_path'])
if fly_folder_to_process == '':
    # If data has not yet been copied off brukerbridge,
    new_fly_number = brainsss.get_new_fly_number(dataset_path)
    fly_folder_to_process = pathlib.Path(dataset_path, 'fly_' + new_fly_number)
else:
    fly_folder_to_process = pathlib.Path(dataset_path, fly_folder_to_process)

# Problem: Snakemake runs twice. Seems to be a bug:
#https://github.com/snakemake/snakemake/issues/2350
# solution: ignore seconds
#logfile = './logs/' + time.strftime("%Y%m%d-%H%M00") + '.txt'
logfile = './logs' + fly_folder_to_process.name + '.txt'
pathlib.Path('../logs').mkdir(exist_ok=True)

width=120
printlog = getattr(brainsss.Printlog(logfile=logfile),'print_to_log')
sys.stderr = brainsss.LoggerRedirect(logfile)
sys.stdout = brainsss.LoggerRedirect(logfile)
brainsss.print_title(logfile, width)


rule HelloSnake:
    #shell:
    #    'python3 hello_world.py $args'
    run:
        try:
            hello_world.print_hi(logfile=logfile,
                       args='world',
                        arg2='test2'
                        )
        except Exception as error_stack:
            brainsss.write_error(logfile=logfile,
                                 error_stack=error_stack)


        #try:
        #    hello_world.print_hi(
        #        args=logging_file,
        #        arg2='test2',
        #        logfile=current_logger)
        #except Exception as error:
        #    current_logger.logger.error(error,exc_info=True)
        """logging_file = './logs/' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
        pathlib.Path('./logs').mkdir(exist_ok=True)
        logger = logging.getLogger('logging_test')
        fh = logging.FileHandler(str(logging_file))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        try:
            logger.info('Starting script')
            # Do stuff
            hello_world.print_hi(args='test',arg2='test2',logfile=logging_file)

            logger.info('Done')
        except Exception as error:
            logger.error(error,exc_info=True)
        """


"""rule stitch_split_nii:
    input:
        current_fly
    run:
        find_split_files(input)


rule fly_builder_rule:
    run:
        fly_builder(user=current_user,
                    dirs_to_build=['20231201']
                         )

"""

