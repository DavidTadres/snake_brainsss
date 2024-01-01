

import nibabel as nib
import numpy as np
import ants

moving_proxy = nib.load('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0/moco/channel_1_moco_mean.nii')
fixed_proxy = nib.load('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/anat_0/moco/channel_1_moco_mean.nii')
moving_data = np.asarray(moving_proxy.dataobj, dtype=np.float32)
fixed_data = np.asarray(fixed_proxy.dataobj, dtype=np.float32)
grad_step = 0.2
flow_sigma = 3
total_sigma = 0
syn_sampling = 32

resolution_of_fixed=(0.653, 0.653, 1)
resolution_of_moving=(2.611, 2.611, 5)
iso_2um_moving = False
iso_2umfixed = True

moving_data_ants = ants.from_numpy(moving_data)
moving_data_ants.set_spacing(resolution_of_moving)
if iso_2um_moving:
    moving_data_ants = ants.resample_image(moving_data_ants, (2, 2, 2), use_voxels=False)

fixed_data_ants = ants.from_numpy(fixed_data)
fixed_data_ants.set_spacing(resolution_of_fixed)
if iso_2umfixed:
    fixed_data_ants = ants.resample_image(fixed_data_ants, (2,2,2), use_voxels=False)

type_of_transform='SyN'
moco = ants.registration(fixed_data_ants,
                         moving_data_ants,
                         type_of_transform=type_of_transform,
                         grad_step=grad_step,
                         flow_sigma=flow_sigma,
                         total_sigma=total_sigma,
                         syn_sampling=syn_sampling)