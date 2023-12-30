"""
Check if supervoxel values are similar (withing floating point precision)
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np

original_path = pathlib.Path('/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/func_0/clustering')
my_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0/clustering')

original_cluster_labels = np.load(pathlib.Path(original_path, 'cluster_labels.npy'))
my_cluster_labels = np.load(pathlib.Path(my_path, 'channel_2_cluster_labels.npy'))

original_cluster_labels == my_cluster_labels