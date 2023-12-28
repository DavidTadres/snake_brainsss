STITCH_NII_FILES = True
data_to_stitch = ['20231207__queue__']  # Data deposited by Brukerbridge on oak

fly_folder_to_process = ''  # if already copied to 'fly_00X' folder and only
# do follow up analysis, enter the fly folder to be analyzed here.
# ONLY ONE FLY PER RUN. Reason is to cleanly separate log files per fly

import pathlib
from scripts import preprocessing
from scripts import stitch_split_nii
import brainsss
import time
import sys

# YOUR SUNET ID
current_user = 'dtadres'

# David's datapaths
#original_data_path = '/Volumes/groups/trc/data/David/Bruker/imports' # Mac
#target_data_path = '/Volumes/groups/trc/data/David/Bruker/preprocessed'
#current_fly = pathlib.Path(original_data_path, '20231207__queue__')

settings = brainsss.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])
imports_path = pathlib.Path(settings['imports_path'])

"""
Note - There really shouldn't be any errors here.
Error log is in a different folder compared to follow-up processing and analysis as 
fly is not yet defined. 
"""
logfile_stitcher = './logs/stitching/' + time.strftime("%Y%m%d-%H%M00") + '.txt'
pathlib.Path('./logs/stitching').mkdir(exist_ok=True)
sys.stderr = brainsss.LoggerRedirect(logfile_stitcher)
sys.stdout = brainsss.LoggerRedirect(logfile_stitcher)
# This doesn't work as part of the main function as the log is not used
# I think because the code runs twice
width = 120 # can go into a config file as well.

rule stitch_split_nii_rule:
    threads: 16
    run:
        try:
            stitch_split_nii.find_split_files(logfile=logfile_stitcher,
                imports_path=imports_path,
                data_to_stitch=data_to_stitch
            )
        except Exception as error_stack:
            brainsss.write_error(logfile=logfile_stitcher,
                error_stack=error_stack,
                width=width)
