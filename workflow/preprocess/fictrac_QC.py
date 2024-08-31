"""

"""

import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
# In console: sys.path.insert(0, 'C:\\Users\\David\\snake_brainsss\\workflow')

# This just imports '*.py' files from the folder 'brainsss'.
from brainsss import utils
from brainsss import fictrac_utils

def fictrac_qc(fly_directory,
               fictrac_file_path,
               WIDTH):
    """
    Perform fictrac quality control.
    This is based on Bella's fictrac_qc.py  script.
    :param fly_directory: a pathlib.Path object to a 'fly' folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param fictrac_file_paths: a list of paths as pathlib.Path objects
    :param fictrac_fps: frames per second of the videocamera recording the fictrac data, an integer
    """
    ####
    # LOGGING
    ####
    logfile = utils.create_logfile(fly_directory, function_name="fictrac_qc")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")

    #utils.print_function_start(logfile, "fictrac_qc")

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    fictrac_file_path = utils.convert_list_of_string_to_posix_path(fictrac_file_path)
    # Organize path - there is only one, but it comes as a list
    fictrac_file_path = fictrac_file_path[0]

    ####
    # QUALITY CONTROL
    ####
    #for current_file in fictrac_file_paths:

    printlog("Currently looking at: " + repr(fictrac_file_path))
    fictrac_raw = fictrac_utils.load_fictrac(fictrac_file_path)
    # This should yield something like 'fly_001/func0/fictrac
    full_id = ", ".join(fictrac_file_path.parts[-4:-2])

    # Real experiment length as defined by Bruker! This should
    # then plot only the fictrac data where we DO have neuronal data!
    # try to guess path_to_metadata:
    # Either is in func0\stimpack\loco\behavior.dat
    if pathlib.Path(fictrac_file_path.parents[2], 'imaging/recording_metadata.xml').exists():
        path_to_metadata = pathlib.Path(fictrac_file_path.parents[2], 'imaging/recording_metadata.xml')
    # else if func0\fictrac\behavior.dat
    elif pathlib.Path(fictrac_file_path.parents[1], 'imaging/recording_metadata.xml').exists():
        path_to_metadata = pathlib.Path(fictrac_file_path.parents[1], 'imaging/recording_metadata.xml')
    neural_timestamps = utils.load_timestamps(path_to_metadata)
    expt_len = neural_timestamps[-1,-1]
    #fictrac_raw.shape[0] / fictrac_fps * 1000 # in ms

    # Extract fictrac fps from fictrac data
    fictrac_fps=fictrac_utils.get_fictrac_fps(fictrac_raw)

    # Prepare fictrac variables
    plotting_resolution = int((1/fictrac_fps)*1000) #10  # desired resolution in ms # Comes from Bella!
    behaviors = ["dRotLabY", "dRotLabZ"]
    fictrac = {}
    for behavior in behaviors:
        if behavior == "dRotLabY":
            short = "Y"
        elif behavior == "dRotLabZ":
            short = "Z"
        fictrac[short] = fictrac_utils.smooth_and_interp_fictrac(
            fictrac_raw, fictrac_fps, expt_len, behavior,
            plotting_resolution
        )
    time_for_plotting_ms = np.arange(0, expt_len, plotting_resolution) # comes in ms

    figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    fictrac_utils.make_velocity_trace(
        axes[0, 0], fictrac, time_for_plotting_ms,
    )
    fictrac_utils.plot_saccades(
        axes[1, 0], fictrac, fictrac_fps, time_for_plotting_ms
    )
    fictrac_utils.fictrac_timestamps_QC(
        axes[2, 0], fictrac_raw, fictrac_file_path
    )

    fictrac_utils.make_2d_hist(
        axes[0, 1], fictrac, fictrac_file_path, fixed_crop=True
    )

    fictrac_utils.make_2d_hist(
        axes[1, 1], fictrac, fictrac_file_path, fixed_crop=False
    )

    fictrac_utils.inter_turn_interval(
        axes[2, 1], fictrac, fictrac_fps, time_for_plotting_ms
    )

    figure.suptitle(fictrac_file_path.parent)
    figure.tight_layout()

    figure.savefig(pathlib.Path(fictrac_file_path.parent, 'fictracQC.png'))
    ###
    # LOG SUCCESS
    ###
    printlog(f"Prepared fictrac QC plot and saved in: {str(fictrac_file_path.parent):.>{WIDTH - 20}}")
