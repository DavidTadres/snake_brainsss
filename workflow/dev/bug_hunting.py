# Ilana found a hard to understand piece of code in brainsss:
# temporal_high_pass_filter.py line 60 seems to assigne chunk data to the
# wrong axis.

# Use toy data to figure out if this is really a bug or not

import numpy as np
from scipy.ndimage import gaussian_filter1d
import nibabel as nib
from pathlib import Path
import h5py

example_data_path = Path('C:\\Users\\David\\Desktop\\channel_1.nii')
savepath = Path(example_data_path.parent, 'HP_filtered.h5')
# Everything is only nifty in this pipeline! Define proxy
brain_data_proxy = nib.load(example_data_path)
# Load everything into memory, cast DTYPE
data = np.asarray(brain_data_proxy.dataobj, dtype=np.float32)

hz=2 # DT: roughly
sigma = int(hz / 0.01)  # gets a good sigma of ~1.5min

stepsize = 2
#data = np.zeros((264,128,34,100)) # x,y,z,t
#data = hf['data']  # this doesn't actually LOAD the data - it is just a proxy
dims = np.shape(data)
print("Data shape is {}".format(dims))

steps = list(range(0, dims[-1], stepsize))
steps.append(dims[-1])

with h5py.File(savepath, 'w') as f:
    dset = f.create_dataset('data', dims, dtype='float32', chunks=True)

    for chunk_num in range(len(steps)):
        print('chunk_num: ' + repr(chunk_num))
        #t0 = time()
        if chunk_num + 1 <= len(steps) - 1:
            chunkstart = steps[chunk_num]
            chunkend = steps[chunk_num + 1]
            chunk = data[:, :, chunkstart:chunkend, :]
            chunk_mean = np.mean(chunk, axis=-1)

            ### SMOOTH ###
            smoothed_chunk = gaussian_filter1d(chunk, sigma=sigma, axis=-1, truncate=1)

            ### Apply Smooth Correction ###
            chunk_high_pass = chunk - smoothed_chunk + chunk_mean[:, :, :,
                                                       None]  # need to add back in mean to preserve offset

            ### Save ###
            f['data'][:, :, chunkstart:chunkend, :] = chunk_high_pass
            # Note: filtering & co takes forever when doing it with
            # actual data-once it's finished (i.e. all slices have been done)
            # it's super fast.

### snakebrains
#data = np.asarray(current_dataset_proxy.dataobj, dtype=DTYPE)
brain_data_proxy = nib.load(example_data_path)
# Load everything into memory, cast DTYPE
data = np.asarray(brain_data_proxy.dataobj, dtype=np.float32)
current_temporal_high_pass_filtered_path = Path(example_data_path.parent, 'HP_filtered_snakebrainsss.nii')
print("Data shape is {}".format(data.shape))

# Calculate mean
data_mean = np.mean(data, axis=-1)
# Using filter to smoothen data. This gets rid of
# high frequency noise.
# Note: sigma: standard deviation for Gaussian kernel
# This needs to be made dynamic else we'll filter differently
# at different
smoothed_data = gaussian_filter1d(
    data, sigma=sigma, axis=-1, truncate=1
)  # This for sure makes a copy of
# the array, doubling memory requirements

# To save memory, do in-place operations where possible
# data_high_pass = data - smoothed_data + data_mean[:,:,:,None]
data -= smoothed_data
# to save memory: data-= ndimage.gaussian_filter1d(...output=data) to do everything in-place?
data += data_mean[:, :, :, None]

##########################
### SAVE DATA AS NIFTY ###
##########################
aff = np.eye(4)
temp_highpass_nifty = nib.Nifti1Image(data, aff)
temp_highpass_nifty.to_filename(current_temporal_high_pass_filtered_path)
print('Successfully saved ' + current_temporal_high_pass_filtered_path.as_posix())

# Compare the two:

with h5py.File(savepath, 'r') as hf:
    h5data = hf['data']
    foo = h5data - data # Check if they are the same!