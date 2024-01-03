"""
Moco is by far the slowest step in the preprocessing pipeline.
It's originally in 2 for loops, an outer one that splits the volume into 'chunks' and an inner loop that feeds
single frames to the ants.registration function.

I want to know how fast one call to ants.registration is: ~6 seconds on my mac

original: ~30 minutes
4 cores:  00:17:29
16 (14) cores: 00:11:43
32 (30) cores: 00:06:18 (91% CPU usage)
64 (63) cores: Maybe not possible on sherlock. Get job submission error!
"""

import nibabel as nib
import pathlib
import ants
import numpy as np
import time
import multiprocessing
import natsort
import sys
import shutil
import itertools

###
# Global variable
###
type_of_transform = "SyN"
flow_sigma = 3
total_sigma = 0
aff_metric = 'mattes'

TESTING = True

def prepare_split_index(moving_path, cores):
    """
    :param total_frames_to_process: a list starting at 0, i.e. [0,1,2,...,n]
    :param cores: an integer number
    :return:
    """
    # Put moving anatomy image into a proxy for nibabel
    moving_proxy = nib.load(moving_path)
    # Read the header to get dimensions
    brain_shape = moving_proxy.header.get_data_shape()
    # last dimension is time, indicating the amount of volumes in the dataset
    experiment_total_frames = brain_shape[-1]
    if TESTING:
        experiment_total_frames = 25
    # Make a list with the amount of volumes. For example, if the dataset has
    # 600 volumes (so .shape is something like [256,128,49,600]
    # we want a list like this: [0, 1, 2, ...599]
    list_of_timepoints = list(np.arange(experiment_total_frames))
    # split the index evenly by the number of provided cores.
    even_split, remainder = divmod(len(list_of_timepoints), cores)
    return (list((list_of_timepoints[i * even_split + min(i, remainder):(i + 1) * even_split + min(i + 1, remainder)] for i in range(cores))))

def motion_correction(index,
                      fixed_path,
                      moving_path,
                      functional_path_one,
                      functional_path_two,
                      temp_save_path,
                      ):
    """
    Loop doing the motion correction for a given set of index.
    This is the function that is doing the heavy lifting of the multiprocessing
    Saves the chunks as npy files in the 'temp_save_path' folder with the index contained
    encoded in the filename.
    :param index:
    :return:
    """
    print("moving_path" + repr(moving_path))
    # Put moving anatomy image into a proxy for nibabel
    moving_proxy = nib.load(moving_path)
    # Read the header to get dimensions
    brain_shape = moving_proxy.header.get_data_shape()
    # Preallocate empty array with necessary size
    frames_to_process = len(index)
    moco_anatomy = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], frames_to_process), dtype=np.float32)

    # Load the meanbrain during a given process. Will cost more memory but should avoid having to shuttle memory
    # around between processes.
    fixed_proxy = nib.load(fixed_path)
    fixed = np.asarray(fixed_proxy.dataobj, dtype=np.uint16)
    fixed_ants = ants.from_numpy(np.asarray(fixed, dtype=np.float32))

    # Load moving proxy in this process
    moving_proxy = nib.load(moving_path)

    if functional_path_one is not None:
        # Load functional one proxy in this process
        functional_one_proxy = nib.load(functional_path_one)
        moco_functional_one = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], frames_to_process),
                                   dtype=np.float32)
        if functional_path_two is not None:
            functional_two_proxy = nib.load(functional_path_two)
            moco_functional_two = np.zeros((brain_shape[0], brain_shape[1], brain_shape[2], frames_to_process),
                                           dtype=np.float32)


    # To keep track of parameters, used for plotting 'motion_correction.png'
    transform_matrix = np.zeros((12, brain_shape[-1]))


    for counter, current_frame in enumerate(index):
        # Remove after development
        print(current_frame)
        t_loop_start = time.time()

        # Load data in a given process
        current_moving = moving_proxy.dataobj[:,:,:,current_frame]
        # Convert to ants images
        moving_ants = ants.from_numpy(np.asarray(current_moving, dtype=np.float32))
        #print('\nants conversion took ' + repr(time.time() - t0) + 's')

        #t0 = time.time()
        # Perform the registration
        moco = ants.registration(fixed_ants, moving_ants,
                                 type_of_transform=type_of_transform,
                                 flow_sigma=flow_sigma,
                                 total_sigma=total_sigma,
                                 aff_metric=aff_metric)
        #print('Registration took ' + repr(time.time() - t0) + 's')

        # put registered anatomy image in correct position in preallocated array
        moco_anatomy[:,:,:,counter] = moco["warpedmovout"].numpy()

        #t0 = time.time()
        # Next, use the transform info for the functional image
        transformlist = moco["fwdtransforms"]

        if functional_path_one is not None:
            current_functional_one = functional_one_proxy.dataobj[:,:,:,current_frame]
            moving_frame_one_ants = ants.from_numpy(np.asarray(current_functional_one, dtype=np.float32))
            # to motion correction for functional image
            moving_frame_one_ants = ants.apply_transforms(fixed_ants, moving_frame_one_ants, transformlist)
            # put moco functional image into preallocated array
            moco_functional_one[:, :, :, counter] = moving_frame_one_ants.numpy()
            #print('apply transforms took ' + repr(time.time() - t0) + 's')

            if functional_path_two is not None:
                current_functional_two = functional_two_proxy.dataobj[:,:,:, current_frame]
                moving_frame_two_ants = ants.from_numpy(np.asarray(current_functional_two, dtype=np.float32))
                moco_functional_two = ants.apply_transforms(fixed_ants, moving_frame_two_ants, transformlist)
                moco_functional_two[:,:,:, counter] = moco_functional_two.numpy()


        #t0=time.time()
        # delete writen files:
        # Delete transform info - might be worth keeping instead of huge resulting file? TBD
        for x in transformlist:
            if ".mat" in x:
                # Keep transform_matrix, I think this is used to make the plot
                # called 'motion_correction.png'
                temp = ants.read_transform(x)
                transform_matrix[:, current_frame] = temp.parameters
                # temp = ants.read_transform(x)
                # transform_matrix.append(temp.parameters)
            # lets' delete all files created by ants - else we quickly create thousands of files!
            pathlib.Path(x).unlink()
        print('Loop duration: ' + repr(time.time() - t_loop_start))
        # LOOP END
    np.save(pathlib.Path(temp_save_path, moving_path.name + 'chunks_'
                         + repr(index[0]) + '-' + repr(index[-1])),
            moco_anatomy)

    if functional_path_one is not None:
        np.save(pathlib.Path(temp_save_path, functional_path_one.name + 'chunks_'
                             + repr(index[0]) + '-' + repr(index[-1])),
                moco_functional_one)
        if functional_path_two is not None:
            np.save(pathlib.Path(temp_save_path, functional_path_two.name + 'chunks_'
                                 +repr(index[0] + '_' + repr(index[-1]))))

    # SAVE TRANSFORM MATRIX AND PUT BACK TOGETHER!!!!
def index_from_filename(filename):
    index_start = int(filename.name.split('chunks_')[-1].split('-')[0])
    index_end = int(filename.name.split('.npy')[0].split('-')[-1])
    total_frames_this_array = index_end - index_start + 1

    return(index_start, index_end, total_frames_this_array)

def combine_temp_files(moving_path,
                       functional_path_one,
                       functional_path_two,
                       temp_save_path):

    # Put moving anatomy image into a proxy for nibabel
    moving_proxy = nib.load(moving_path)
    # Read the header to get dimensions
    brain_shape = moving_proxy.header.get_data_shape()

    stitched_anatomy_brain = np.zeros((brain_shape[0],brain_shape[1],
                                       brain_shape[2], brain_shape[3]),
                                      dtype=np.float32)
    if functional_path_one is not None:
        stitched_functional_one = np.zeros((brain_shape[0],brain_shape[1],
                                       brain_shape[2], brain_shape[3]),
                                      dtype=np.float32)
    if functional_path_two is not None:
        stitched_functional_two = np.zeros((brain_shape[0],brain_shape[1],
                                       brain_shape[2], brain_shape[3]),
                                      dtype=np.float32)

    for current_file in natsort.natsorted(temp_save_path.iterdir()):
        if '.npy' in current_file.name and moving_path.name in current_file.name:
            #index_start = int(current_file.name.split('chunks_')[-1].split('-')[0])
            #index_end = int(current_file.name.split('.npy')[0].split('-')[-1])
            #total_frames_this_array = index_end-index_start+1
            index_start, index_end, total_frames_this_array = index_from_filename(current_file)
            stitched_anatomy_brain[:,:,:,index_start:index_start+total_frames_this_array] = np.load(current_file)
        elif functional_path_one is not None:
            if '.npy' in current_file.name and functional_path_one.name in current_file.name:
                index_start, index_end, total_frames_this_array = index_from_filename(current_file)
                stitched_functional_one[:,:,:,index_start:index_start+total_frames_this_array] = np.load(current_file)

            elif functional_path_two is not None:
                if '.npy' in current_file.name and functional_path_two.name in current_file.name:
                    index_start, index_end, total_frames_this_array = index_from_filename(current_file)
                stitched_functional_two[:,:,:,index_start:index_start+total_frames_this_array] = np.load(current_file)
    # Saving
    # we create a new subfolder called 'moco' where the file is saved
    # Not a great solution - the definition of the output file should happen in the snakefile, not hidden
    # in here!
    savepath_root = pathlib.Path(moving_path.parents[1], 'moco')
    savepath_root.mkdir(parents=True, exist_ok=True)
    # Prepare nifti file
    aff = np.eye(4)
    # Save anatomy channel:
    savepath_anatomy = pathlib.Path(savepath_root, moving_path.stem + '_moco.nii')
    stitched_anatomy_brain_nifty = nib.Nifti1Image(stitched_anatomy_brain, aff)
    stitched_anatomy_brain_nifty.to_filename(savepath_anatomy)

    if functional_path_one is not None:
        savepath_func_one = pathlib.Path(savepath_root, functional_path_one.stem + '_moco.nii')
        stitched_functional_one_nifty = nib.Nifti1Image(stitched_functional_one, aff)
        stitched_functional_one_nifty.to_filename(savepath_func_one)

        if functional_path_two is not None:
            savepath_func_two = pathlib.Path(savepath_root, functional_path_two.stem + '_moco.nii')
            stitched_functional_two_nifty = nib.Nifti1Image(stitched_functional_two, aff)
            stitched_functional_two_nifty.to_filename(savepath_func_two)

    # After saving the stitched file, delete the temporary files
    shutil.rmtree(temp_save_path)

if __name__ == '__main__':
    # Organize input
    # sys.argv[0] is the name of the script
    #type_of_transform = sys.argv[1]
    #flow_sigma = sys.argv[2]
    #total_sigma = sys.argv[3]
    #aff_metric = sys.argv[4]

    # Convert str paths to pathlib.Path objects as everywhere else in snake-brainsss
    fixed_path = pathlib.Path(sys.argv[1])
    moving_path = pathlib.Path(sys.argv[2])
    try:
        functional_path_one = pathlib.Path(sys.argv[3])
    except IndexError:
        functional_path_one = None

    try:
        functional_path_two = pathlib.Path(sys.argv[4])
    except IndexError:
        functional_path_two = None #
        pass

    # Path were the intermediate npy files are saved.
    # It is important that it's different for each run.
    # We can just put in on scratch
    # This will only work if we have a folder called trc and data is in /data, of course
    relevant_temp_save_path_part = moving_path.as_posix().split('trc/data/')[-1]
    temp_save_path = pathlib.Path('/scratch/groups/trc', relevant_temp_save_path_part).parent
    if TESTING:
       temp_save_path = pathlib.Path('/Users/dtadres/Documents/test_folder')
    if temp_save_path.is_dir():
        # Check if temp_save is on scratch - Only delete folder if yes to avoid deleting source data (for now at least)
        if temp_save_path.parents[-2].as_posix()  == '/scratch':
            # Remove dir with all files if it exists!!!
            shutil.rmtree(temp_save_path)
        else:
            print('WARNING: Did not remove files in ' + str(temp_save_path))
            print('Only remove folders that are on scratch to avoid accidentally deleting source data')

    # create dir
    # No need for exist_ok=True because the dir should have been deleted just now
    temp_save_path.mkdir(parents=True)

    # always use one core less than max to make sure nothing gets clogged
    cores = 31 # Sherlock should always use 32 cores so we can use 31 for parallelization
    #cores = multiprocessing.cpu_count() - 1
    if TESTING:
        cores = 4
    
    # create split index
    split_index = prepare_split_index(moving_path, cores)

    with multiprocessing.Pool(cores) as p:
        p.starmap(motion_correction, zip(split_index,
                                         itertools.repeat(fixed_path),
                                         itertools.repeat(moving_path),
                                         itertools.repeat(functional_path_one),
                                         itertools.repeat(functional_path_two),
                                         itertools.repeat(temp_save_path)
                                         ))
    print('Motion correction done, combining files now.')
    combine_temp_files(moving_path, functional_path_one, functional_path_two, temp_save_path)
    print('files combined')
# Then put them together to compare