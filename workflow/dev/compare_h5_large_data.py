"""
Files are not identical size but this might be because one is saved as compress and the other
is not: https://stackoverflow.com/questions/61028349/why-are-two-h5py-files-different-in-size-when-content-is-the-same
"""

import h5py
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use("agg") # As this should be run on sherlock, use non-interactive backend!

def run_comparison(path_original, path_my):
    print(path_my)
    path_original = pathlib.Path(path_original)
    path_my = pathlib.Path(path_my)
    print(path_my)

    with h5py.File(path_original, 'r') as hf:
        original_proxy = hf['data']
        #print(loop_proxy.shape)
        #loop_one_slice = loop_proxy[:, :, 3, 50]
        original_data = original_proxy[:]
        print('first loaded')

        with h5py.File(path_my, 'r') as hf:
            my_proxy = hf['data']
            #vec_one_slice = vec_proxy[:, :, 3, 50]
            my_data = my_proxy
            print('second loaded')

            """with h5py.File(path_vec_original, 'r') as hf:
                vec_orig_proxy = hf['data']
                print(vec_orig_proxy.shape)
                vec_orig_one_slice = vec_orig_proxy[:, :, 3, 50]
            print('third loaded')"""
            z_slice = 25
            t_slice = 100

            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax1.imshow(original_data[:,:,z_slice, t_slice].T)
            ax1.set_title(path_original.name + ', z=' + repr(z_slice) + ', t=' + repr(t_slice))

            ax2 = fig.add_subplot(222)
            ax2.imshow(my_data[:,:,z_slice, t_slice])
            ax2.set_title(path_my.name + ', z=' + repr(z_slice) + ', t=' + repr(t_slice))

            delta = original_data[:,:,z_slice, t_slice] - my_data[:,:,z_slice, t_slice]
            ax3 = fig.add_subplot(223)
            ax3.imshow(delta.T)
            ax3.set_title('Max delta in this slice' + repr(np.max(delta)))

            # Next, plot histogram of both brains using ALL data (not just a single slice
            counts_original, edges_original = np.histogram(original_data, bins=1000)
            counts_my, edges_my = np.histogram(my_data, bins=1000)
            ax4 = fig.add_subplot(224)
            ax4.stairs(counts_original, edges_original, fill=True, alpha=1, color="k")
            ax4.stairs(counts_my, edges_my, fill=True, alpha=0.5, color="r")
            ax4.set_yscale("log")
            delta = (
                    original_data - my_data
            )  # what's the difference in value between the two arrays?
            ax4.set_title(
                "Max abs delta between arrays\n" + repr(round(np.max(np.abs(delta)), 10))
            )
            fig.tight_layout()
            fig.savefig(pathlib.Path(path_my.parent, path_my.name + '_delta.png'))