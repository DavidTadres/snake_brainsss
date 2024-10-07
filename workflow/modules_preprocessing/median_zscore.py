import pathlib
import sys
import numpy as np
import nibabel as nib


parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
# This just imports '*.py' files from the folder 'brainsss'.
from brainsss import utils
from modules_preprocessing import constants

# Initialize constant class
CONSTANTS = constants.Constants()

def median_zscore(fly_directory, dataset_path, median_zscore_path):
    """
    Calulate modified zscore:
    https://docs.oracle.com/en/cloud/saas/planning-budgeting-cloud/pfusu/insights_metrics_MODIFIED_Z_SCORE.html
    https://www.statology.org/modified-z-score/
    https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=terms-modified-z-score

    :param fly_directory: a pathlib.Path object to a 'fly' folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param dataset_path: Full path as a list of pathlib.Path objects to the nii to be read
    :param zscore_path: Full path as a list of pathlib.Path objects to the nii to be saved
    """
    #####################
    ### MEDIAN ZSCORE ###
    #####################
    logfile = utils.create_logfile(fly_directory, function_name="zscore")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "zscore")

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    zscore_path = utils.convert_list_of_string_to_posix_path(median_zscore_path)

    printlog("Beginning MEDIAN-ZSCORE")

    # we might get a second functional channel in the future!
    for current_dataset_path, current_zscore_path in zip(dataset_path, zscore_path):
        if 'nii' in current_dataset_path.name:
            dataset_proxy = nib.load(current_dataset_path)
            data = np.asarray(dataset_proxy.dataobj, dtype=CONSTANTS.DTYPE)

            printlog("Data shape is {}".format(data.shape))

            # Expect a 4D array, xyz and the fourth dimension is time!
            median_brain = np.nanmedian(data, axis=3)
            printlog('Calculated median of data')
            # Calculate absolute difference between each value and the median (per voxel)
            absolute_delta = np.abs(data - median_brain[:,:,:,np.newaxis])
            # Calculate median absolute deviation per voxel
            median_absolute_deviation = np.nanmedian(absolute_delta, axis=3)

            modified_zscore = (0.6745*(data-median_brain[:,:,:, np.newaxis]))/median_absolute_deviation[:,:,:, np.newaxis]

            aff=np.eye(4)
            zscore_nifty = nib.Nifti1Image(modified_zscore  , aff)
            zscore_nifty.to_filename(current_zscore_path)

            printlog('Saved median z-score image as ' + current_zscore_path.as_posix())
        else:
            printlog('Function currently only works with nii as output')
            printlog('this will probably break')
