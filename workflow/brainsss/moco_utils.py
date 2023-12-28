import h5py
import numpy as np
import matplotlib as mpl

mpl.use("agg")  # Agg, is a non-interactive backend that can only write to files.
# Without this I had the following error: Starting a Matplotlib GUI outside of the main thread will likely fail.
from matplotlib import pyplot as plt
import pathlib
import time
import nibabel as nib

# To import brainsss, define path to scripts!
import sys

scripts_path = pathlib.Path(
    __file__
).parent.resolve()  # path of workflow i.e. /Users/dtadres/snake_brainsss/workflow
sys.path.insert(0, pathlib.Path(scripts_path, "workflow"))

from brainsss import utils


def make_empty_h5(directory, file, brain_dims):  # , save_type):
    # if save_type == 'curr_dir':
    # moco_dir = os.path.join(directory,'moco')
    moco_dir = pathlib.Path(directory, "moco")
    # if not os.path.exists(moco_dir):
    # 	os.mkdir(moco_dir)
    moco_dir.mkdir(exist_ok=True, parents=True)
    # elif save_type == 'parent_dir':
    # 		directory = os.path.dirname(directory) # go back one directory
    # 	moco_dir = os.path.join(directory,'moco')
    # 		if not os.path.exists(moco_dir):
    # 		os.mkdir(moco_dir)

    # savefile = os.path.join(moco_dir, file)
    savefile = pathlib.Path(moco_dir, file)
    with h5py.File(savefile, "w") as f:
        dset = f.create_dataset("data", brain_dims, dtype="float32", chunks=True)
    return moco_dir, savefile


# def check_for_file(file, directory):
# 	filepath = os.path.join(directory, file)
# 	if os.path.exists(filepath):
# 		return filepath
# 	else:
# 		return None


def save_moco_figure(transform_matrix, parent_path, moco_dir, printlog):
    """

    :param transform_matrix:
    :param parent_path:
    :param moco_dir:
    :param printlog:
    :return:
    """

    xml_path = None
    # Get voxel resolution for figure
    for current_file in parent_path.iterdir():
        if "recording_metadata.xml" in current_file.name:
            xml_path = current_file

            # if xml_path == None:
            # 	printlog('Could not find xml file for scan dimensions. Skipping plot.')
            # 	return
            # elif not xml_path.is_file():
            # 	printlog('Could not find xml file for scan dimensions. Skipping plot.')
            # 	return

            printlog(f"Found xml file.")
            x_res, y_res, z_res = utils.get_resolution(xml_path)

            # Save figure of motion over time
            # save_file = os.path.join(moco_dir, 'motion_correction.png')
            save_file = pathlib.Path(moco_dir, "motion_correction.png")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.plot(
                transform_matrix[:, 9] * x_res, label="y"
            )  # note, resolutions are switched since axes are switched
            ax.plot(transform_matrix[:, 10] * y_res, label="x")
            ax.plot(transform_matrix[:, 11] * z_res, label="z")
            ax.set_ylabel("Motion Correction, um")
            ax.set_xlabel("Time")
            ax.set_title(moco_dir)
            plt.legend()
            fig.savefig(save_file, bbox_inches="tight", dpi=300)

            return
    printlog("Could not find xml file for scan dimensions. Skipping plot.")


def print_progress_table_moco(total_vol, complete_vol, printlog, start_time, width):
    """
    There's a very similarly named function in utils!
    :param total_vol:
    :param complete_vol:
    :param printlog:
    :param start_time:
    :param width:
    :return:
    """
    fraction_complete = complete_vol / total_vol

    ### Get elapsed time ###
    elapsed = time.time() - start_time
    elapsed_hms = sec_to_hms(elapsed)

    ### Get estimate of remaining time ###
    try:
        remaining = elapsed / fraction_complete - elapsed
    except ZeroDivisionError:
        remaining = 0
    remaining_hms = sec_to_hms(remaining)

    ### Get progress bar ###
    complete_vol_str = f"{complete_vol:04d}"
    total_vol_str = f"{total_vol:04d}"
    length = (
        len(elapsed_hms)
        + len(remaining_hms)
        + len(complete_vol_str)
        + len(total_vol_str)
    )
    bar_string = utils.progress_bar(complete_vol, total_vol, width - length - 10)

    full_line = (
        "| "
        + elapsed_hms
        + "/"
        + remaining_hms
        + " | "
        + complete_vol_str
        + "/"
        + total_vol_str
        + " |"
        + bar_string
        + "|"
    )
    printlog(full_line)


def sec_to_hms(t):
    secs = f"{np.floor(t%60):02.0f}"
    mins = f"{np.floor((t/60)%60):02.0f}"
    hrs = f"{np.floor((t/3600)%60):02.0f}"
    return ":".join([hrs, mins, secs])


def h5_to_nii(h5_path):
    """
    Here we go from float 32 back to uint16 (original files from Bruker seem to be uint16).
    Probably saves a ton of space but what effect does it have on data analysis due to lost precision?
    :param h5_path:
    :return:
    """
    nii_savefile = h5_path.name.split(".")[0] + ".nii"
    with h5py.File(h5_path, "r+") as h5_file:
        image_array = h5_file.get("data")[:].astype("uint16")

    nifti1_limit = 2**16 / 2
    if np.any(np.array(image_array.shape) >= nifti1_limit):  # Need to save as nifti2
        nib.save(nib.Nifti2Image(image_array, np.eye(4)), nii_savefile)
    else:  # Nifti1 is OK
        nib.save(nib.Nifti1Image(image_array, np.eye(4)), nii_savefile)

    return nii_savefile
