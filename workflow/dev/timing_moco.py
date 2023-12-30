"""
Moco is by far the slowest step in the preprocessing pipeline.
It's originally in 2 for loops, an outer one that splits the volume into 'chunks' and an inner loop that feeds
single frames to the ants.registration function.

I want to know how fast one call to ants.registration is.
"""

import h5py
import nibabel as nib
import pathlib
import ants
import matplotlib.pyplot as plt
import numpy as np
import time

type_of_transform = "SyN"
flow_sigma = 3
total_sigma = 0
aff_metric = 'mattes'

imaging_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/SS84990_DNa03_x_UAS-CD8-GFP/fly_002/anat_low_res/imaging/')
fixed_path = pathlib.Path(imaging_path, 'channel_2_mean.nii')
moving_path = pathlib.Path(imaging_path, 'channel_2.nii')

fixed_proxy = nib.load(fixed_path)
fixed = np.asarray(fixed_proxy.dataobj, dtype=np.uint16)

moving_proxy = nib.load(moving_path)
moving_current = np.asarray(moving_proxy.dataobj, np.uint16)

current_moving = moving_current[:,:,:,0]
t0 = time.time()
# Need to have ants images
fixed_ants = ants.from_numpy(np.asarray(fixed, dtype=np.float32))
moving_ants = ants.from_numpy(np.asarray(current_moving, dtype=np.float32))
print('ants conversion took ' + repr(time.time() - t0) + 's')

t0 = time.time()
moco = ants.registration(fixed_ants, moving_ants,
                         type_of_transform=type_of_transform,
                         flow_sigma=flow_sigma,
                         total_sigma=total_sigma,
                         aff_metric=aff_metric)
print('Registration took ' + repr(time.time() - t0) + 's')

moco["warpedmovout"].numpy().shape # > volume of input file!