"""
make_mean_brain calls np.mean which returns a float64 object

"""
import nibabel as nib
import pathlib
import numpy as np
import matplotlib.pyplot as plt

data_path = pathlib.Path(
    "/Users/dtadres/Documents/func1/imaging/functional_channel_1_mean.nii"
)

data_proxy = nib.load(data_path)
data64 = np.asarray(data_proxy.dataobj, dtype=np.float64)

data_uint16 = np.asarray(data_proxy.dataobj, dtype=np.uint16)

print(np.max(data64 - data_uint16))
# > ~1, (0.9983388704322351) as we are not rounding, just chopping of the numbers. e.g.
# 1.9 or 1.1 would become 1

counts64, edges64 = np.histogram(data64, bins=1000)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.stairs(counts64, edges64, fill=True)
# Most values are either very small (0-500). There's a second peak at 8K which indicates a saturated PMT

# plt.imshow(data_uint16[:,:,0]-data64[:,:,0])
##########################

data_original_path = pathlib.Path(
    "/Users/dtadres/Documents/func1/imaging/functional_channel_1.nii"
)
data_original_proxy = nib.load(data_original_path)
data_original_uint16 = np.asarray(data_original_proxy.dataobj, dtype=np.uint16)
counts_uint16, edges_uint16 = np.histogram(data_original_uint16, bins=1000)
ax2 = fig.add_subplot(212)
ax2.stairs(counts_uint16, edges_uint16, fill=True)
ax2.set_yscale("log")
ax2.set_xlabel("Pixel intensity [AU]")
