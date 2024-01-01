"""
Moco is by far the slowest step in the preprocessing pipeline.
It's originally in 2 for loops, an outer one that splits the volume into 'chunks' and an inner loop that feeds
single frames to the ants.registration function.

I want to know how fast one call to ants.registration is.

original: ~30 minutes
4 cores:  00:17:29
16 (14) cores: 00:11:43
32 (30) cores: 00:06:18 (91% CPU usage)
64 (63) cores: Maybe not possible on sherlock. Get job submission error!
"""

import h5py
import nibabel as nib
import pathlib
import ants
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import natsort

RUN_LOCAL = False # if not run on sherlock


type_of_transform = "SyN"
flow_sigma = 3
total_sigma = 0
aff_metric = 'mattes'
if RUN_LOCAL:
    imaging_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging')
    experiment_total_frames = 100  # So that I don't wait forever during testing, of course it should just be brain_shape[3]
    cores = 4
else:
    imaging_path = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging')
    cores = 31
fixed_path = pathlib.Path(imaging_path, 'channel_1_mean.nii')
moving_path = pathlib.Path(imaging_path, 'channel_1.nii')
functional_path = pathlib.Path(imaging_path, 'channel_2.nii')
#save_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/moco_parallel')
temp_save_path = pathlib.Path('/scratch/users/dtadres/test_moco2')

fixed_proxy = nib.load(fixed_path)
fixed = np.asarray(fixed_proxy.dataobj, dtype=np.uint16)

moving_proxy = nib.load(moving_path)
#moving_data = np.asarray(moving_proxy.dataobj, np.uint16)

functional_proxy = nib.load(functional_path)
#functional_data = np.asarray(functional_proxy.dataobj, np.uint16)

brain_shape = functional_proxy.header.get_data_shape()
if not RUN_LOCAL:
    experiment_total_frames = brain_shape[-1] # run full experiment

#moco_anatomy = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], brain_shape[3]),dtype=np.float32)
#moco_functional = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], brain_shape[3]),dtype=np.float32)

loop_duration = []
#total_frames_to_process = 10
transform_matrix = np.zeros((12,brain_shape[-1]))
#> FOR LOCAL
# Before multiprocessing I need to split the timepoints that we'll use. In the original for loop we would have something
# like range(brain.shape[3]) which would spit out a list going from 0 to brain.shape[3]. Instead we can prepare
# one list per parallel processing.


def split_input(index, cores):
    """
    :param total_frames_to_process: a list starting at 0, i.e. [0,1,2,...,n]
    :param cores: an integer number
    :return:
    """
    even_split, remainder = divmod(len(index), cores)
    return list((index[i * even_split + min(i, remainder):(i + 1) * even_split + min(i + 1, remainder)] for i in range(cores)))


#def split_array(array, cores):
#    even_split, remainder = divmod(array.shape[-1], cores)


def for_loop_moco(index):
    moco_anatomy = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], int(index[-1]-index[0]+1)), dtype=np.float32)
    moco_functional = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], int(index[-1]-index[0]+1)), dtype=np.float32)

    for counter, current_frame in enumerate(index):

        """
        I usually get less than 10 seconds per loop. Mean is 5.9seconds
        For 600frames that should be ~ 6,000 seconds or 100 minutes
        Interestingly I got about 30 minutes with the original code indicating that there
        is some sort of parallelization going on somehwere. (or the cluster just calculates faster than my mac)
        """
        print(current_frame)

        t_loop_start = time.time()
        current_moving = moving_proxy.dataobj[:,:,:,current_frame]
        #t0 = time.time()
        # Need to have ants images
        fixed_ants = ants.from_numpy(np.asarray(fixed, dtype=np.float32))
        moving_ants = ants.from_numpy(np.asarray(current_moving, dtype=np.float32))
        #print('\nants conversion took ' + repr(time.time() - t0) + 's')

        #t0 = time.time()
        moco = ants.registration(fixed_ants, moving_ants,
                                 type_of_transform=type_of_transform,
                                 flow_sigma=flow_sigma,
                                 total_sigma=total_sigma,
                                 aff_metric=aff_metric)
        #print('Registration took ' + repr(time.time() - t0) + 's')

        #moco_anatomy[:,:,:,current_frame] = moco["warpedmovout"].numpy()
        moco_anatomy[:,:,:,counter] = moco["warpedmovout"].numpy()


        #t0 = time.time()
        # Next, use the transform info fofr the functional image
        transformlist = moco["fwdtransforms"]
        #moving_frame=functional_data[:,:,:,current_frame]

        current_functional = functional_proxy.dataobj[:,:,:,current_frame]
        moving_frame_ants = ants.from_numpy(np.asarray(current_functional, dtype=np.float32))
        moco_ch2 = ants.apply_transforms(fixed_ants, moving_frame_ants, transformlist)
        #moco_functional[:,:,:,current_frame] = moco_ch2.numpy()
        moco_functional[:, :, :, counter] = moco_ch2.numpy()
        #print('apply transforms took ' + repr(time.time() - t0) + 's')

        #t0=time.time()
        # delete writen files:
        # Delete transform info - might be worth keeping instead of huge resulting file? TBD
        for x in transformlist:
            if ".mat" in x:
                temp = ants.read_transform(x)
                transform_matrix[:, current_frame] = temp.parameters
                # temp = ants.read_transform(x)
                # transform_matrix.append(temp.parameters)
            # lets' delete all files created by ants - else we quickly create thousands of files!
            pathlib.Path(x).unlink()
        #print('Delete took: ' + repr(time.time()-t0))
        loop_duration.append(time.time()-t_loop_start)
        print('Loop duration: ' + repr(loop_duration[-1]))
        # LOOP END

    np.save(pathlib.Path(temp_save_path, moving_path.name + 'chunks_'
                         + repr(index[0]) + '-' + repr(index[-1])),
            moco_anatomy)
    np.save(pathlib.Path(temp_save_path, functional_path.name + 'chunks_'
                         + repr(index[0]) + '-' + repr(index[-1])),
            moco_functional)


split_index = split_input(list(np.arange(experiment_total_frames)),cores)
# The code below parallelizes the transform part of moco
# Without parallelization I get ~6 seconds per loop. With 4 cores I get ~12 seconds
# It therefore should take half the time to run motion correction.
# On the server we could get a 8x increase in speed, so instead of ~8 hours it might only
# take 1 hour

def combine_files():
    stitched_anatomy_brain = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], brain_shape[3]), dtype=np.float32)
    for current_file in natsort.natsorted(temp_save_path.iterdir()):
        if 'npy' in current_file.name and 'channel_1.nii' in current_file.name:
            index_start = int(current_file.name.split('chunks_')[-1].split('-')[0])
            index_end = int(current_file.name.split('.npy')[0].split('-')[-1])
            stitched_anatomy_brain[:,:,:,index_start:index_end] = np.load(current_file)[:,:,:,0:int(index_end-index_start)]
    savepath = pathlib.Path(imaging_path.parent() + '/moco')
    savepath.mkdir(exist_ok=True, parents=True)



if __name__ == '__main__':
    with multiprocessing.Pool(cores) as p:
        p.map(for_loop_moco, split_index)
    print('Motion correction done, combining files now.')
    combine_files()
    print('files combined')
# Then put them together to compare