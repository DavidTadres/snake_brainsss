"""
Check if supervoxel values are similar (withing floating point precision)
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np

original_path = pathlib.Path('/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/func_0/clustering')
my_path1 = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0/clustering1')
my_path2 = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0/clustering')

original_cluster_labels = np.load(pathlib.Path(original_path, 'cluster_labels.npy'))
my_cluster_labels1 = np.load(pathlib.Path(my_path1, 'channel_2_cluster_labels.npy'))
my_cluster_labels2 = np.load(pathlib.Path(my_path2, 'channel_2_cluster_labels.npy'))

# Check if two runs yield the same result
(my_cluster_labels1 == my_cluster_labels2).all() # YES, they do, this is a deteminsitc step

