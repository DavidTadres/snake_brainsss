import pathlib
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

parallel_moco_path = pathlib.Path('/Users/dtadres/Documents/test_moco/stitched_ch1.nii')
loop_moco_path = pathlib.Path('/Users/dtadres/Documents/test_moco/channel_1_moco.h5')

# Compare mean brain results

z_slice = 25
t_slice = 100

with h5py.File(loop_moco_path, 'r') as hf:
    loop_moco_brain_proxy = hf['data']
    # print(loop_proxy.shape)
    # loop_one_slice = loop_proxy[:, :, 3, 50]
    loop_moco_brain = loop_moco_brain_proxy[:]

    parallel_moco_brain_proxy = nib.load(parallel_moco_path)
    parallel_moco_brain = np.asarray(parallel_moco_brain_proxy.dataobj, np.float32)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(loop_moco_brain[:, :, z_slice, t_slice].T)
    ax1.set_title(loop_moco_path.name + ', z=' + repr(z_slice) + ', t=' + repr(t_slice))

    ax2 = fig.add_subplot(222)
    ax2.imshow(parallel_moco_brain[:, :, z_slice, t_slice].T)
    ax2.set_title(parallel_moco_path.name + ', z=' + repr(z_slice) + ', t=' + repr(t_slice))

    delta = loop_moco_brain[:, :, z_slice, t_slice] - parallel_moco_brain[:, :, z_slice, t_slice]
    ax3 = fig.add_subplot(223)
    ax3.imshow(delta.T)
    ax3.set_title('Max delta in this slice' + repr(np.max(delta)))

    # Next, plot histogram of both brains using ALL data (not just a single slice
    counts_loop, edges_loop = np.histogram(loop_moco_brain, bins=1000)
    counts_parallel, edges_parallel = np.histogram(parallel_moco_brain, bins=1000)
    ax4 = fig.add_subplot(224)
    ax4.stairs(counts_loop, edges_loop, fill=True, alpha=1, color="k")
    ax4.stairs(counts_parallel, edges_parallel, fill=True, alpha=0.5, color="r")
    ax4.set_yscale("log")
    delta = (
            loop_moco_brain - parallel_moco_brain
    )  # what's the difference in value between the two arrays?
    ax4.set_title(
        "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
    )
    fig.tight_layout()
    fig.savefig(pathlib.Path('/Users/dtadres/Documents/moco_parallel.png'))


