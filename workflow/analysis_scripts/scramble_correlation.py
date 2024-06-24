import numpy as np
import nibabel as nib
import pathlib
import sys

parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)

from brainsss import fictrac_utils
from brainsss import utils

# The original brainsss usually saved everything in float32 but some
# calculations were done with float64 (e.g. np.mean w/o defining dtype).
# To be more explicit, I define here two global variables.
# Dtype is what was mostly called when saving and loading data
DTYPE = np.float32

def calculate_scrambled_correlation(fly_log_folder,
                                    fictrac_path,
                                    fictrac_fps,
                                    metadata_path,
                                    moco_zscore_highpass_path,
                                    save_path
                                    ):
    """
    Scrambles each velocity timepoint before calculating the behavior correlation
    with neuronal data.

    # Shuffle principle:
    foo = np.zeros((10,3))
    foo[:,0] = np.arange(0,10)
    foo[:,1] = np.arange(10,20)
    foo[:,2] = np.arange(20,30)
    print(foo[:,0])
    # >[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    np.random.shuffle(foo[:,0])
    print(foo[:,0])
    # >[9. 4. 6. 3. 8. 0. 7. 2. 1. 5.]


    :param fictrac_path:
    :param fictrac_fps:
    :param metadata_path:
    :param moco_zscore_highpass_path:
    :param save_path:
    :return:
    """
    #####################
    ### SETUP LOGGING ###
    #####################
    logfile = utils.create_logfile(fly_log_folder, function_name="scramble_correlation")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    utils.print_function_start(logfile, "correlation")

    #


    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    moco_zscore_highpass_path = utils.convert_list_of_string_to_posix_path(moco_zscore_highpass_path)
    save_path = utils.convert_list_of_string_to_posix_path(save_path)

    printlog('Will use ' + repr(moco_zscore_highpass_path) + ' for input.')
    printlog('Will save as ' + repr(save_path))
    # Then extract the desired behavior. Since there's only one output per run, the list
    # has length 1.
    behavior = save_path[0].name.split("corr_")[-1].split("_SCRAMBLED.nii")[0]
    # Load fictrac data
    fictrac_raw = fictrac_utils.load_fictrac(fictrac_path)
    # Calculate length of experiment.
    expt_len = (fictrac_raw.shape[0] / fictrac_fps) * 1000
    # and the resolution
    fictrac_resolution = (1/fictrac_fps) * 1000 # how many ms between frames
    # Load timestamps from imaging data (used for fictrac data)
    neural_timestamps = utils.load_timestamps(metadata_path)
    # interpolate fictrac to match the timestamps from the microscope!
    fictrac_interp = fictrac_utils.smooth_and_interp_fictrac(
        fictrac_raw, fictrac_fps, fictrac_resolution, expt_len, behavior, timestamps=neural_timestamps
    )
    printlog('Loaded fictrac data')
    # It seems easier to scramble the behavior timestamps instead of the brain timestamps!
    # Use np.shuffle to scramble the timepoints!
    fictrac_interp_shuffled = fictrac_interp.copy()  # Ensure we copy and not just view the array!
    for current_z in range(fictrac_interp_shuffled.shape[1]):
        np.random.shuffle(fictrac_interp_shuffled[:, current_z])
    printlog('Shuffled fictrac data')
    for current_moco_zscore_highpass_path, current_savepath in zip(moco_zscore_highpass_path,save_path):
        # Next, load brain!
        #brain_proxy = nib.load(moco_zscore_highpass_path)
        brain_proxy = nib.load(current_moco_zscore_highpass_path)
        brain = np.asarray(brain_proxy.dataobj, dtype=DTYPE)
        printlog('loaded brain file, starting correlation now')
        printlog('will do ' + repr(brain.shape[2]) + ' z-slices')

        # Correlation - copy paste from snake-brainsss (vectorized correlation, fast!)
        # Preallocate an array filled with zeros
        scrambled_corr_brain = np.zeros(
            (brain.shape[0], brain.shape[1], brain.shape[2]), dtype=DTYPE
        )
        # Fill array with NaN to rule out that we interpret a missing value as a real value when it should be NaN
        scrambled_corr_brain.fill(np.nan)
        # Loop through each slice:
        for z in range(brain.shape[2]):
            print('current z-slice: ' + repr(z))
            # Vectorized correlation - see 'dev/pearson_correlation.py' for development and timing info
            # The formula is:
            # r = (sum(x-m_x)*(y-m_y) / sqrt(sum((x-m_x)^2)*sum((y-m_y)^2)
            brain_mean = brain[:, :, z, :].mean(axis=-1)  # , dtype=np.float64)

            # When I plot the data plt.plot(fictrac_interp) and plt.plot(fictrac_interp.astype(np.float32) I
            # can't see a difference. Should be ok to do this as float.
            # This is advantagous because we have to do the dot product with the brain. np.dot will
            # default to higher precision leading to making a copy of brain data which costs a lot of memory
            # Unfortunately fictrac_interp comes in [time, z] as opposed to brain that comes in [x,y,z,t]
            fictrac_shuffled_mean = fictrac_interp_shuffled[:, z].mean(dtype=DTYPE)

            # >> Typical values for z scored brain seem to be between -25 and + 25.
            # It shouldn't be necessary to cast as float64. This then allows us
            # to do in-place operation!
            brain_mean_m = brain[:, :, z, :] - brain_mean[:, :, None]
            # fictrac data is small, so working with float64 shouldn't cost much memory!
            # Correction - if we cast as float64 and do dot product with brain, we'll copy the
            # brain to float64 array, balloning the memory requirement
            fictrac_shuffled_mean_m = fictrac_interp_shuffled[:, z].astype(DTYPE) - fictrac_shuffled_mean

            # Note from scipy pearson docs: This can overflow if brain_mean_m is, for example [-5e210, 5e210, 3e200, -3e200]
            # I doubt we'll ever get close to numbers like this.
            normbrain = np.linalg.norm(
                brain_mean_m, axis=-1
            )  # Make a copy, but since there's no time dimension it's quite small
            normfictrac_shuffled = np.linalg.norm(fictrac_shuffled_mean_m)

            # Calculate correlation
            scrambled_corr_brain[:, :, z] = np.dot(
                brain_mean_m / normbrain[:, :, None], fictrac_shuffled_mean_m / normfictrac_shuffled
            )

        # Prepare Nifti file for saving
        aff = np.eye(4)
        object_to_save = nib.Nifti1Image(scrambled_corr_brain, aff)
        # nib.Nifti1Image(corr_brain, np.eye(4)).to_filename(save_file)
        printlog('Saving file to ' + repr(current_savepath))
        object_to_save.to_filename(current_savepath)