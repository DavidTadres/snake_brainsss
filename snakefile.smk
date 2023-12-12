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
from stitch_split_nii import find_split_files
from preprocessing import fly_builder

# YOUR SUNET ID
current_user = 'dtadres'

# David's datapaths
original_data_path = '/oak/stanford/groups/trc/data/David/Bruker/imports'
#target_data_path = '/Volumes/groups/trc/data/David/Bruker/preprocessed'

current_fly = pathlib.Path(original_data_path, '20231201')
print(current_fly)
import hello_world

import logging
import time

#
class LogFile():
    """

    """
    logging_file = './logs/' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
    pathlib.Path('./logs').mkdir(exist_ok=True)
    logger = logging.getLogger('logging_test')
    fh = logging.FileHandler(str(logging_file))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

current_log_file = LogFile()

rule HelloSnake:
    #shell:
    #    'python3 hello_world.py $args'
    run:
        try:
            hello_world.print_hi(args='test',arg2='test2',logfile=current_log_file.logging_file)
        except Exception as error:
            current_log_file.logger.error(error,exc_info=True)
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
            logger.error(error,exc_info=True)"""


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

