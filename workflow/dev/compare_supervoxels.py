"""
compare original (Yandan's fly 308) and my supervoxel results
"""
import pathlib
import numpy as np

original_fly_path = pathlib.Path(
    "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308"
)
my_fly_paths = pathlib.Path(
    "/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002"
)

original_labels = np.load(pathlib.Path(original_fly_path, 'func_0/clustering/cluster_labels.npy'))
my_labels = np.load(pathlib.Path(my_fly_paths, 'func_0/clustering/channel_2_cluster_labels.npy'))