import pathlib
import matplotlib.pyplot as plt
import numpy as np


original_fly_path = pathlib.Path(
    "/Volumes/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308"
)
my_fly_paths = pathlib.Path(
    "/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002"
)
my_savepaths = pathlib.Path(my_fly_paths, "testing")

original_motcorr_params = np.load(pathlib.Path(original_fly_path, 'anat_0/moco/motcorr_params.npy'))

my_motcorr_params = np.load(pathlib.Path(my_fly_paths, 'anat_0/moco/motcorr_params.npy'))

diff_motcorr_params = original_motcorr_params - my_motcorr_params