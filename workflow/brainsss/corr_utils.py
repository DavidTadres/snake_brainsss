import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg') # Agg, is a non-interactive backend that can only write to files.
# Without this I had the following error: Starting a Matplotlib GUI outside of the main thread will likely fail.

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pathlib
def save_maxproj_img(image_to_max_project, path):
    """
    Plot max intensity projection (over time) of correlated brain.
    :param image_to_max_project: numpy file with time in last dimension
    :param path: pathlib.Path object of the nii file that is being saved
    :return:
    """
    #brain = np.asarray(nib.load(file).get_fdata().squeeze(), dtype='float32')

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    max_proj_plot = ax.imshow(np.max(image_to_max_project, axis=-1).T, cmap='gray')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(max_proj_plot[3], cax=cax, orientation='vertical')

    # swap the suffix from whatever the input file is to .png
    savepath = pathlib.Path(path.parent, path.name.split(path.suffix)[0] + '.png')

    fig.savefig(savepath, bbox_inches='tight', dpi=300)