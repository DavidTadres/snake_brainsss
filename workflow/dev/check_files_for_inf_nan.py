"""
I have a suspicion that some files contain nan/inf.
Here I'll have a ready made script to allow for checking of files
"""
import pathlib
import nibabel as nib
import numpy as np
import traceback
import time

def check_for_nan_and_inf_func(array):
    """
    Check if there are any nan or inf in the array that is being passed.

    :param array:
    :return:
    """
    no_nan_inf=True
    try:
        if np.isnan(array).any():
            print('!!!!! WARNING - THERE ARE NAN IN THE ARRAY !!!!!')
            print('The position(s) of np.nan is/are: ' + repr(np.where(np.isnan(array))))
            no_nan_inf=False
    except:
        print('Could not check for nan because:\n\n')
        print(traceback.format_exc())
        print('\n')
        no_nan_inf=False

    try:
        if np.isinf(array).any():
            print('!!!!! WARNING - THERE ARE INF IN THE ARRAY !!!!!')
            print('The position(s) of np.inf is/are ' + repr(np.where(np.isnan(array))))
            no_nan_inf=False
    except:
        print('Could not check for inf because:\n\n')
        print(traceback.format_exc())
        print('\n')
        no_nan_inf=False

    if no_nan_inf:
        print('No nan or inf found in this array')
def read_nii_data_and_check(data_path, dtype):
    data_proxy = nib.load(data_path)
    data = np.asarray(data_proxy.dataobj, dtype=dtype)
    check_for_nan_and_inf_func(data)
    del data # Make sure to keep memory as empty as possible
    time.sleep(1)

###
# 'Original' data
data_path = pathlib.Path('/Users/dtadres/Documents/func1/imaging/functional_channel_1.nii')
read_nii_data_and_check(data_path, dtype=np.uint16)
# > No nan or inf found in this array

data_path = pathlib.Path('/Users/dtadres/Documents/func1/imaging/functional_channel_1_mean.nii')
read_nii_data_and_check(data_path, dtype=np.float32) # < make_mean_brain loads like this
# > No nan or inf found in this array

###

data_path = pathlib.Path('/Users/dtadres/Documents/func1/moco/channel_1_moco_mean.nii')
read_nii_data_and_check(data_path)
