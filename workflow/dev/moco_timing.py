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
    # Make a list with the amount of volumes. For example, if the dataset has
    # 600 volumes (so .shape is something like [256,128,49,600]
    # we want a list like this: [0, 1, 2, ...599]
    list_of_timepoints = list(np.arange(experiment_total_frames))
    # split the index evenly by the number of provided cores.
    even_split, remainder = divmod(len(list_of_timepoints), cores)
    return (list((list_of_timepoints[i * even_split + min(i, remainder):(i + 1) * even_split + min(i + 1, remainder)] for i in range(cores))))

def motion_correction(index,
                      moving_path,
                      fixed_path,
                      functional_path_one,
                      functional_path_two,
                      temp_save_path,
                      #type_of_transform,
                      #flow_sigma,
                      #total_sigma,
                      #aff_metric
                      ):
    """
    Loop doing the motion correction for a given set of index.
    This is the function that is doing the heavy lifting of the multiprocessing
    Saves the chunks as npy files in the 'temp_save_path' folder with the index contained
    encoded in the filename.
    :param index:
    :return:
    """
    print(index)
    print(moving_path)
    print(fixed_path)
    print(functional_path_one)
    print(functional_path_two)

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
            moco_functional_one = ants.apply_transforms(fixed_ants, moving_frame_one_ants, transformlist)
            # put moco functional image into preallocated array
            moco_functional_one[:, :, :, counter] = moco_functional_one.numpy()
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
    savepath_anatomy = pathlib.Path(savepath_root, moving_path.name.stem + '_moco.nii')
    stitched_anatomy_brain_nifty = nib.Nifti1Image(stitched_anatomy_brain, aff)
    stitched_anatomy_brain_nifty.to_filename(savepath_anatomy)

    if functional_path_one is not None:
        savepath_func_one = pathlib.Path(savepath_root, functional_path_one.stemp + '_moco.nii')
        stitched_functional_one_nifty = nib.Nifti1Image(stitched_functional_one, aff)
        stitched_functional_one_nifty.to_filename(savepath_func_one)

        if functional_path_two is not None:
            savepath_func_two = pathlib.Path(savepath_root, functional_path_two.stemp + '_moco.nii')
            stitched_functional_two_nifty = nib.Nifti1Image(stitched_functional_two, aff)
            stitched_functional_two_nifty.to_filename(savepath_func_two)

class MotionCorrect():
    def __init__(self, fixed_path, moving_path, functional_paths, cores, temp_save_path=None):
        self.type_of_transform = "SyN"
        self.flow_sigma = 3
        self.total_sigma = 0
        self.aff_metric = 'mattes'
        self.cores = cores

        RUN_LOCAL = True  # True if run on my Mac, False if run on sherlock
        # Interesting: When I set this to True, it took LONGER:00:07:10
        # instead of 00:06:41 - not a huge difference but definitely not better..
        # With preload false, got  00:07:06 - seems to just make no difference!
        #PRELOAD_DATA = False


        if RUN_LOCAL:
            imaging_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging')
            self.temp_save_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/temp_moco2')
            self.experiment_total_frames = 25  # So that I don't wait forever during testing, of course it should just be brain_shape[3]
            #self.cores = 4
        else:
            imaging_path = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging')
            self.temp_save_path = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/temp_moco2')
            #self.cores = 31
        self.fixed_path = pathlib.Path(imaging_path, 'channel_1_mean.nii')
        self.moving_path = pathlib.Path(imaging_path, 'channel_1.nii')
        self.functional_path = pathlib.Path(imaging_path, 'channel_2.nii')
        #save_path = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/moco_parallel')
        #temp_save_path = pathlib.Path('/scratch/users/dtadres/test_moco2')

        functional_proxy = nib.load(self.functional_path)
        brain_shape = functional_proxy.header.get_data_shape()
        if not RUN_LOCAL:
            self.experiment_total_frames = brain_shape[-1] # run full experiment

        #loop_duration = []
        #total_frames_to_process = 10
        self.transform_matrix = np.zeros((12,brain_shape[-1]))
        #> FOR LOCAL
        # Before multiprocessing I need to split the timepoints that we'll use. In the original for loop we would have something
        # like range(brain.shape[3]) which would spit out a list going from 0 to brain.shape[3]. Instead we can prepare
        # one list per parallel processing.

        self.split_index = self.split_input(list(np.arange(self.experiment_total_frames)), self.cores)
        # The code below parallelizes the transform part of moco
        # Without parallelization I get ~6 seconds per loop. With 4 cores I get ~12 seconds
        # It therefore should take half the time to run motion correction.
        # On the server we could get a 8x increase in speed, so instead of ~8 hours it might only
        # take 1 hour


    def split_input(self, index, cores):
        """
        :param total_frames_to_process: a list starting at 0, i.e. [0,1,2,...,n]
        :param cores: an integer number
        :return:
        """
        even_split, remainder = divmod(len(index), cores)
        return(list((index[i * even_split + min(i, remainder):(i + 1) * even_split + min(i + 1, remainder)] for i in range(cores))))


    #def split_array(array, cores):
    #    even_split, remainder = divmod(array.shape[-1], cores)


    def for_loop_moco(self, index):
        """
        Loop doing the motion correction for a given set of index.
        This is the function that is doing the heavy lifting of the multiprocessing
        Saves the chunks as npy files in the 'temp_save_path' folder with the index contained
        encoded in the filename.
        :param index:
        :return:
        """

        # Preallocate empty array with necessary size
        frames_to_process = len(index)
        moco_anatomy = np.zeros((self.brain_shape[0], self.brain_shape[1], self.brain_shape[2], frames_to_process), dtype=np.float32)
        moco_functional = np.zeros((self.brain_shape[0], self.brain_shape[1], self.brain_shape[2], frames_to_process), dtype=np.float32)

        # Load the meanbrain during a given process. Will cost more memory but should avoid having to shuttle memory
        # around between processes.
        fixed_proxy = nib.load(self.fixed_path)
        fixed = np.asarray(fixed_proxy.dataobj, dtype=np.uint16)
        fixed_ants = ants.from_numpy(np.asarray(fixed, dtype=np.float32))

        # Load moving proxy in this process
        moving_proxy = nib.load(self.moving_path)

        # Load function proxy in this process
        functional_proxy = nib.load(self.functional_path)

        for counter, current_frame in enumerate(index):

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
                                     type_of_transform=self.type_of_transform,
                                     flow_sigma=self.flow_sigma,
                                     total_sigma=self.total_sigma,
                                     aff_metric=self.aff_metric)
            #print('Registration took ' + repr(time.time() - t0) + 's')

            # put registered anatomy image in correct position in preallocated array
            moco_anatomy[:,:,:,counter] = moco["warpedmovout"].numpy()

            #t0 = time.time()
            # Next, use the transform info for the functional image
            transformlist = moco["fwdtransforms"]

            current_functional = functional_proxy.dataobj[:,:,:,current_frame]
            moving_frame_ants = ants.from_numpy(np.asarray(current_functional, dtype=np.float32))
            # to motion correction for functional image
            moco_ch2 = ants.apply_transforms(fixed_ants, moving_frame_ants, transformlist)
            # put moco functional image into preallocated array
            moco_functional[:, :, :, counter] = moco_ch2.numpy()
            #print('apply transforms took ' + repr(time.time() - t0) + 's')

            #t0=time.time()
            # delete writen files:
            # Delete transform info - might be worth keeping instead of huge resulting file? TBD
            for x in transformlist:
                if ".mat" in x:
                    # Keep transform_matrix, I think this is used to make the plot
                    # called 'motion_correction.png'
                    temp = ants.read_transform(x)
                    self.transform_matrix[:, current_frame] = temp.parameters
                    # temp = ants.read_transform(x)
                    # transform_matrix.append(temp.parameters)
                # lets' delete all files created by ants - else we quickly create thousands of files!
                pathlib.Path(x).unlink()
            print('Loop duration: ' + repr(time.time() - t_loop_start))
            # LOOP END
        np.save(pathlib.Path(self.temp_save_path, self.moving_path.name + 'chunks_'
                             + repr(index[0]) + '-' + repr(index[-1])),
                moco_anatomy)
        np.save(pathlib.Path(self.temp_save_path, self.functional_path.name + 'chunks_'
                             + repr(index[0]) + '-' + repr(index[-1])),
                moco_functional)




    def combine_files(self,):
        stitched_anatomy_brain = np.zeros((self.brain_shape[0], self.brain_shape[1], self.brain_shape[2], self.brain_shape[3]), dtype=np.float32)
        for current_file in natsort.natsorted(self.temp_save_path.iterdir()):
            if 'npy' in current_file.name and 'channel_1.nii' in current_file.name:
                index_start = int(current_file.name.split('chunks_')[-1].split('-')[0])
                index_end = int(current_file.name.split('.npy')[0].split('-')[-1])
                total_frames_this_array = index_end-index_start+1
                stitched_anatomy_brain[:,:,:,index_start:index_start+total_frames_this_array] = np.load(current_file)
            # Saving
            #savepath = pathlib.Path(imaging_path.parent, '/moco')
            #savepath.mkdir(exist_ok=True, parents=True)
            aff = np.eye(4)
            stitched_anatomy_brain_nifty = nib.Nifti1Image(
                stitched_anatomy_brain, aff
            )
            stitched_anatomy_brain_nifty.to_filename(pathlib.Path(self.temp_save_path, 'stitched_ch1.nii'))

if __name__ == '__main__':
    # Organize input
    # sys.argv[0] is the name of the script
    #type_of_transform = sys.argv[1]
    #flow_sigma = sys.argv[2]
    #total_sigma = sys.argv[3]
    #aff_metric = sys.argv[4]
    fixed_path = sys.argv[1]
    moving_path = sys.argv[2]
    try:
        functional_path_one = sys.argv[3]
    except IndexError:
        functional_path_one = None

    try:
        functional_path_two = sys.argv[4]
    except IndexError:
        functional_path_two = None #
        pass

    # Path were the intermediate npy files are saved.
    # It is important that it's different for each run.
    # We can just put in on scratch
    # This will only work if we have a folder called trc and data is in /data, of course
    relevant_temp_save_path_part = moving_path.as_posix().split('trc/data/')[-1]
    temp_save_path = pathlib.Path('/scratch/groups/trc', relevant_temp_save_path_part).parent
    if temp_save_path.is_dir():
        # Remove dir with all files if it exists!!!
        shutil.rmtree(temp_save_path)
    # create dir
    # No need for exist_ok=True because the dir should have been deleted just now
    temp_save_path.mkdir(parents=True)

    # always use one core less than max to make sure nothing gets clogged
    cores = multiprocessing.cpu_count() - 1
    
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