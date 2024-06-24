import numpy as np
import nibabel as nib
import pandas as pd
from xml.etree import ElementTree as ET
from scipy.interpolate import interp1d
import scipy
import pathlib
import sys

parent_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, parent_path)
from brainsss import utils

def calculate_scrambled_correlation(fictrac_path,
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

    DTYPE = np.float32

    ##########
    ### Convert list of (sometimes empty) strings to pathlib.Path objects
    ##########
    moco_zscore_highpass_path = utils.convert_list_of_string_to_posix_path(moco_zscore_highpass_path)
    save_path = utils.convert_list_of_string_to_posix_path(save_path)

    print('Will use ' + repr(moco_zscore_highpass_path) + ' for input.')
    print('Will save as ' + repr(save_path))
    # Then extract the desired behavior. Since there's only one output per run, the list
    # has length 1.
    behavior = save_path[0].name.split("corr_")[-1].split("_SCRAMBLED.nii")[0]
    # Load fictrac data
    fictrac_raw = load_fictrac(fictrac_path)
    # Calculate length of experiment.
    expt_len = (fictrac_raw.shape[0] / fictrac_fps) * 1000
    # and the resolution
    fictrac_resolution = (1/fictrac_fps) * 1000 # how many ms between frames
    # Load timestamps from imaging data (used for fictrac data)
    neural_timestamps = load_timestamps(metadata_path)
    # interpolate fictrac to match the timestamps from the microscope!
    fictrac_interp = smooth_and_interp_fictrac(
        fictrac_raw, fictrac_fps, fictrac_resolution, expt_len, behavior, timestamps=neural_timestamps
    )
    print('Loaded fictrac data')
    # It seems easier to scramble the behavior timestamps instead of the brain timestamps!
    # Use np.shuffle to scramble the timepoints!
    fictrac_interp_shuffled = fictrac_interp.copy()  # Ensure we copy and not just view the array!
    for current_z in range(fictrac_interp_shuffled.shape[1]):
        np.random.shuffle(fictrac_interp_shuffled[:, current_z])
    print('Shuffled fictrac data')
    for current_moco_zscore_highpass_path, current_savepath in zip(moco_zscore_highpass_path,save_path):
        # Next, load brain!
        #brain_proxy = nib.load(moco_zscore_highpass_path)
        brain_proxy = nib.load(current_moco_zscore_highpass_path)
        brain = np.asarray(brain_proxy.dataobj, dtype=DTYPE)
        print('loaded brain file, starting correlation now')
        print('will do ' + repr(brain.shape[2]) + ' z-slices')

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
        print('Saving file to ' + repr(current_savepath))
        object_to_save.to_filename(current_savepath)


def load_fictrac(fictrac_file_path):
    """
    Loads fictrac.dat file, adds columns names and performs a speed sanity check.
    To-do:
        1) change units based on diameter of ball etc.
        2) For speed sanity check, instead remove bad frames so we don't have to throw out whole trial.

    :param fictrac_file_path: string with full path to fictrac data file
    :return pandas dataframe of all parameters saved by fictrac
    """
    with open(fictrac_file_path, "r") as f:
        df = pd.DataFrame(l.rstrip().split() for l in f)

        # Name columns
        df = df.rename(
            index=str,
            columns={
                0: "frameCounter",
                1: "dRotCamX",
                2: "dRotCamY",
                3: "dRotCamZ",
                4: "dRotScore",
                5: "dRotLabX",
                6: "dRotLabY",
                7: "dRotLabZ",
                8: "AbsRotCamX",
                9: "AbsRotCamY",
                10: "AbsRotCamZ",
                11: "AbsRotLabX",
                12: "AbsRotLabY",
                13: "AbsRotLabZ",
                14: "positionX",
                15: "positionY",
                16: "heading",
                17: "runningDir",
                18: "speed",
                19: "integratedX",
                20: "integratedY",
                21: "timeStamp",
                22: "sequence",
            },
        )

        # Remove commas
        for column in df.columns.values[:-1]:
            df[column] = [float(x[:-1]) for x in df[column]]

        fictrac_data = df

    # sanity check for extremely high speed (fictrac failure)
    speed = np.asarray(fictrac_data["speed"])
    max_speed = np.max(speed)
    if max_speed > 10:
        raise Exception(
            "Fictrac ball tracking failed (reporting impossibly high speed)."
        )
    return fictrac_data
def load_timestamps(path_to_metadata):
    """
    NEW:
    Parses a Bruker xml file to get the times of each frame
    :param path_to_metadata: a full filepath to a recording_metadata.xml file
    return: timestamps: [t,z] numpy array of times (in ms) of Bruker imaging frames.

    ORIGINAL:

    Parses a Bruker xml file to get the times of each frame, or loads h5py file if it exists.

    First tries to load from 'timestamps.h5' (h5py file). If this file doesn't exist
    it will load and parse the Bruker xml file, and save the h5py file for quick loading in the future.

    Parameters
    ----------
    directory: full directory that contains xml file (str).
    file: Defaults to 'functional.xml'

    Returns
    -------
    timestamps: [t,z] numpy array of times (in ms) of Bruker imaging frames.

    """
    # Not sure if h5py really needs to be made...We only need the timestamps once per call
    # and it's different for each recording...
    # if '.h5' in path_to_metadata:
    # try:
    #    print('Trying to load timestamp data from hdf5 file.')
    #    #with h5py.File(os.path.join(directory, 'timestamps.h5'), 'r') as hf:
    #    with h5py.File(path_to_metadata, 'r') as hf:
    #        timestamps = hf['timestamps'][:]

    # except:
    # else:
    # print('Failed. Extracting frame timestamps from bruker xml file.')
    # xml_file = os.path.join(directory, file)
    # xml_file = pathlib.Path(directory, file)

    tree = ET.parse(path_to_metadata)
    root = tree.getroot()
    timestamps = []

    sequences = root.findall("Sequence")
    for sequence in sequences:
        frames = sequence.findall("Frame")
        for frame in frames:
            # filename = frame.findall('File')[0].get('filename')
            time = float(frame.get("relativeTime"))
            timestamps.append(time)
    timestamps = np.multiply(timestamps, 1000)

    if len(sequences) > 1:
        timestamps = np.reshape(timestamps, (len(sequences), len(frames)))
    else:
        timestamps = np.reshape(timestamps, (len(frames), len(sequences)))

    ### Save h5py file ###
    # with h5py.File(os.path.join(directory, 'timestamps.h5'), 'w') as hf:
    #    hf.create_dataset("timestamps", data=timestamps)

    print("Success.")
    return timestamps

def smooth_and_interp_fictrac(
    fictrac, fps, resolution, expt_len, behavior, timestamps=None, z=None
):  # , smoothing=25, z=None):
    if behavior == "dRotLabZpos":
        behavior = "dRotLabZ"
        clip = "pos"
    elif behavior == "dRotLabZneg":
        behavior = "dRotLabZ"
        clip = "neg"
    else:
        clip = None

    ### get orginal timestamps ###
    camera_rate = 1 / fps * 1000  # camera frame rate in ms
    x_original = np.arange(
        0, expt_len, camera_rate
    )  # same shape as fictrac (e.g. 20980)

    ### smooth ###
    # >>> DANGEROUS - the filter length of the following function is not normalized by the fps
    # e.g. Bella recorded at 100 fps and if we do fictrac_smooth with window length 25 we get
    # filtered data over 10ms * 25 = 250ms.
    # If I record at 50fps each frame is only 20ms. We still filter over 25 points so now we
    # filter over 25*20 = 500ms
    # <<<<
    # I remove the smoothing input from this function and make it dependent on the fps
    smoothing = int(
        np.ceil(0.25 / (1 / fps))
    )  # This will always yield 250 ms (or the next closest
    # possible number, e.g. if we have 50fps we would get a smotthing window of 12.5 which we can't
    # index of course. We always round up so with 50 fps we'd get 13 = 260 ms
    fictrac_smoothed = scipy.signal.savgol_filter(
        np.asarray(fictrac[behavior]), smoothing, 3
    )
    # Identical shape in output as input, e.g. 20980

    ### clip if desired ###
    if clip == "pos":
        fictrac_smoothed = np.clip(fictrac_smoothed, a_min=0, a_max=None
        )  # Unsure what this does
    elif clip == "neg":
        fictrac_smoothed = (np.clip(fictrac_smoothed, a_min=None, a_max=0) * -1
        )  # Unsure what this does

    ### interpolate ###
    # This function probably just returns everything from an input array
    fictrac_interp_temp = interp1d(
        x_original, fictrac_smoothed, bounds_error=False
    )  # yields a function
    xnew = np.arange(0, expt_len, resolution)  # 0 to last time at subsample res
    # ## different number, e.g. 41960, or just 2x shape before.
    # This is probably because resolution is set to 10. If framerate is 50 we have a frame every 20 ms.
    if timestamps is None:
        fictrac_interp = fictrac_interp_temp(xnew)
    elif z is not None:  # For testing only!
        fictrac_interp = fictrac_interp_temp(timestamps[:, z])
    else:
        # So we only select which timestamps here.
        # fictrac_interp = fictrac_interp_temp(timestamps[:,z]) # This would return ALL timestamps per z slice
        fictrac_interp = fictrac_interp_temp(timestamps)

    ### convert units for common cases ###
    sphere_radius = 4.5e-3  # in m
    if behavior in ["dRotLabY"]:
        """starts with units of rad/frame
        * sphere_radius(m); now in m/frame
        * fps; now in m/sec
        * 1000; now in mm/sec"""

        fictrac_interp = fictrac_interp * sphere_radius * fps * 1000  # now in mm/sec

    if behavior in ["dRotLabZ"]:
        """starts with units of rad/frame
        * 180 / np.pi; now in deg/frame
        * fps; now in deg/sec"""

        fictrac_interp = fictrac_interp * 180 / np.pi * fps

    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False)

    return fictrac_interp