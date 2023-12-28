"""
This is some stuff from Yandan: https://github.com/yandanw/Analysis_Collection/blob/main/make_explosion-function.ipynb
I just want to quickly look at a clustered brain
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
import scipy
import pandas as pd
from scipy.interpolate import interp1d
from xml.etree import ElementTree as ET

def load_fictrac(fictrac_file_path):
    # def load_fictrac(directory, file='fictrac.dat'):
    """
    NEW:


    ORIGINAL:
    Loads fictrac data from .dat file that fictrac outputs.

    To-do: change units based on diameter of ball etc.
    For speed sanity check, instead remove bad frames so we don't have to throw out whole trial.

    Parameters
    ----------
    directory: string of full path to file
    file: string of file name

    Returns
    -------
    fictrac_data: pandas dataframe of all parameters saved by fictrac """

    # for item in os.listdir(directory):
    #  if '.dat' in item:
    #    file = item

    # with open(os.path.join(directory, file),'r') as f:
    with open(fictrac_file_path, 'r') as f:
        df = pd.DataFrame(l.rstrip().split() for l in f)

        # Name columns
        df = df.rename(index=str, columns={0: 'frameCounter',
                                           1: 'dRotCamX',
                                           2: 'dRotCamY',
                                           3: 'dRotCamZ',
                                           4: 'dRotScore',
                                           5: 'dRotLabX',
                                           6: 'dRotLabY',
                                           7: 'dRotLabZ',
                                           8: 'AbsRotCamX',
                                           9: 'AbsRotCamY',
                                           10: 'AbsRotCamZ',
                                           11: 'AbsRotLabX',
                                           12: 'AbsRotLabY',
                                           13: 'AbsRotLabZ',
                                           14: 'positionX',
                                           15: 'positionY',
                                           16: 'heading',
                                           17: 'runningDir',
                                           18: 'speed',
                                           19: 'integratedX',
                                           20: 'integratedY',
                                           21: 'timeStamp',
                                           22: 'sequence'})

        # Remove commas
        for column in df.columns.values[:-1]:
            df[column] = [float(x[:-1]) for x in df[column]]

        fictrac_data = df

    # sanity check for extremely high speed (fictrac failure)
    speed = np.asarray(fictrac_data['speed'])
    max_speed = np.max(speed)
    if max_speed > 10:
        raise Exception('Fictrac ball tracking failed (reporting impossibly high speed).')
    return fictrac_data

def smooth_and_interp_fictrac(fictrac, fps, resolution, expt_len, behavior, timestamps=None,
                              z=None):  # , smoothing=25, z=None):

    if behavior == 'dRotLabZpos':
        behavior = 'dRotLabZ'
        clip = 'pos'
    elif behavior == 'dRotLabZneg':
        behavior = 'dRotLabZ'
        clip = 'neg'
    else:
        clip = None

    ### get orginal timestamps ###
    camera_rate = 1 / fps * 1000  # camera frame rate in ms
    x_original = np.arange(0, expt_len, camera_rate)  # same shape as fictrac (e.g. 20980)

    ### smooth ###
    # >>> DANGEROUS - the filter length of the following function is not normalized by the fps
    # e.g. Bella recorded at 100 fps and if we do fictrac_smooth with window length 25 we get
    # filtered data over 10ms * 25 = 250ms.
    # If I record at 50fps each frame is only 20ms. We still filter over 25 points so now we
    # filter over 25*20 = 500ms
    # <<<<
    # I remove the smoothing input from this function and make it dependent on the fps
    smoothing = int(np.ceil(0.25 / (1 / 50)))  # This will always yield 250 ms (or the next closest
    # possible number, e.g. if we have 50fps we would get a smotthing window of 12.5 which we can't
    # index of course. We always round up so with 50 fps we'd get 13 = 260 ms
    fictrac_smoothed = scipy.signal.savgol_filter(np.asarray(fictrac[behavior]), smoothing, 3)
    # Identical shape in output as input, e.g. 20980

    ### clip if desired ###
    if clip == 'pos':
        fictrac_smoothed = np.clip(fictrac_smoothed, a_min=0, a_max=None)  # Unsure what this does
    elif clip == 'neg':
        fictrac_smoothed = np.clip(fictrac_smoothed, a_min=None, a_max=0) * -1  # Unsure what this does

    ### interpolate ###
    # This function probably just returns everything from an input array
    fictrac_interp_temp = interp1d(x_original, fictrac_smoothed, bounds_error=False)  # yields a function
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
    if behavior in ['dRotLabY']:
        ''' starts with units of rad/frame
        * sphere_radius(m); now in m/frame
        * fps; now in m/sec
        * 1000; now in mm/sec '''

        fictrac_interp = fictrac_interp * sphere_radius * fps * 1000  # now in mm/sec

    if behavior in ['dRotLabZ']:
        ''' starts with units of rad/frame
        * 180 / np.pi; now in deg/frame
        * fps; now in deg/sec '''

        fictrac_interp = fictrac_interp * 180 / np.pi * fps

    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False);

    return (fictrac_interp)

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
    #if '.h5' in path_to_metadata:
    #try:
    #    print('Trying to load timestamp data from hdf5 file.')
    #    #with h5py.File(os.path.join(directory, 'timestamps.h5'), 'r') as hf:
    #    with h5py.File(path_to_metadata, 'r') as hf:
    #        timestamps = hf['timestamps'][:]

    #except:
    #else:
    #print('Failed. Extracting frame timestamps from bruker xml file.')
    #xml_file = os.path.join(directory, file)
    #xml_file = pathlib.Path(directory, file)

    tree = ET.parse(path_to_metadata)
    root = tree.getroot()
    timestamps = []

    sequences = root.findall('Sequence')
    for sequence in sequences:
        frames = sequence.findall('Frame')
        for frame in frames:
            #filename = frame.findall('File')[0].get('filename')
            time = float(frame.get('relativeTime'))
            timestamps.append(time)
    timestamps = np.multiply(timestamps, 1000)

    if len(sequences) > 1:
        timestamps = np.reshape(timestamps, (len(sequences), len(frames)))
    else:
        timestamps = np.reshape(timestamps, (len(frames), len(sequences)))

    ### Save h5py file ###
    #with h5py.File(os.path.join(directory, 'timestamps.h5'), 'w') as hf:
    #    hf.create_dataset("timestamps", data=timestamps)

    print('Success.')
    return timestamps

path_fly = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/')
cluster_labels = np.load(pathlib.Path(path_fly, 'func0/clustering/channel_2_cluster_labels.npy'))
cluster_signals = np.load(pathlib.Path(path_fly, 'func0/clustering/channel_2_cluster_signals.npy'))

fictrac_path = pathlib.Path(path_fly, 'func0/fictrac/fictrac_behavior_data.dat')
fictrac_raw = load_fictrac(fictrac_path)

fps = 100
resolution = 10 #desired resolution in ms
expt_len = fictrac_raw.shape[0]/fps*1000
timestamps = load_timestamps(pathlib.Path(path_fly, 'func0/imaging/recording_metadata.xml'))
'''## to vectorize
corrs = []
behavior = 'dRotLabY'
for z in range(49):
    fictrac_trace = smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, behavior, timestamps[:,z])
    fictrac_trace_L = np.clip(fictrac_trace.flatten(),None,0)*-1 #only needed for Z that has +-.
    # shifted_beh = np.roll(fictrac['Y'][:,0],8) # mismatch of the time
    for voxel in range(2000): #<Optimize! Else it'll take forever.
        corrs.append(scipy.stats.pearsonr(cluster_signals[z,voxel,:], fictrac_trace.flatten())[0])
##'''
z_slices = 49
n_clusters = 2000
behavior = 'dRotLabY'
fictrac_interp = smooth_and_interp_fictrac(fictrac_raw, fps, resolution, expt_len, behavior,
                                                        timestamps=timestamps)

corr_brain = np.zeros((z_slices, n_clusters))
for z in range(corr_brain.shape[0]):
    signal_mean = cluster_signals[z,:,:].mean(axis=-1, dtype=np.float32)
    fictrac_mean = fictrac_interp[:,z].mean(dtype=np.float32)

    signal_mean_m = cluster_signals[z,:,:] - signal_mean[:,None]
    fictrac_mean_m = fictrac_interp[:,z].astype(np.float32) - fictrac_mean

    normsignal = np.linalg.norm(signal_mean_m, axis=-1)
    normfictrac = np.linalg.norm(fictrac_mean_m)

    corr_brain[z,:] = np.dot(signal_mean_m/normsignal[:,None], fictrac_mean_m/normfictrac)

whole_corr = np.reshape(corr_brain,(z_slices,n_clusters))



reformed_brain=[]
for z in range(z_slices):
    colored_by_betas = np.zeros((256 * 128))
    for cluster_num in range(n_clusters):
        cluster_indicies = np.where(cluster_labels[z, :] == cluster_num)[0]
        colored_by_betas[cluster_indicies] = whole_corr[z, cluster_num]
    colored_by_betas = colored_by_betas.reshape(256, 128)
    reformed_brain.append(colored_by_betas)

def warp_STA_brain(STA_brain, fly, fixed, anat_to_mean_type):
    n_tp = STA_brain.shape[1]
    dataset_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset'
    moving_resolution = (2.611, 2.611, 5)
    ###########################
    ### Organize Transforms ###
    ###########################
    warp_directory = os.path.join(dataset_path, fly, 'warp')
    warp_sub_dir = 'func-to-anat_fwdtransforms_2umiso'
    affine_file = os.listdir(os.path.join(warp_directory, warp_sub_dir))[0]
    affine_path = os.path.join(warp_directory, warp_sub_dir, affine_file)
    if anat_to_mean_type == 'myr':
        warp_sub_dir = 'anat-to-meanbrain_fwdtransforms_2umiso'
    elif anat_to_mean_type == 'non_myr':
        warp_sub_dir = 'anat-to-non_myr_mean_fwdtransforms_2umiso'
    else:
        print('invalid anat_to_mean_type')
        return
    syn_files = os.listdir(os.path.join(warp_directory, warp_sub_dir))
    syn_linear_path = os.path.join(warp_directory, warp_sub_dir, [x for x in syn_files if '.mat' in x][0])
    syn_nonlinear_path = os.path.join(warp_directory, warp_sub_dir, [x for x in syn_files if '.nii.gz' in x][0])
    ####transforms = [affine_path, syn_linear_path, syn_nonlinear_path]
    transforms = [syn_nonlinear_path, syn_linear_path, affine_path] ### INVERTED ORDER ON 20220503!!!!
    #ANTS DOCS ARE SHIT. THIS IS PROBABLY CORRECT, AT LEAST IT NOW WORKS FOR THE FLY(134) THAT WAS FAILING


    ### Warp timeponts
    warps = []
    for tp in range(n_tp):
        to_warp = np.rollaxis(STA_brain[:,tp,:,:],0,3)
        moving = ants.from_numpy(to_warp)
        moving.set_spacing(moving_resolution)
        ########################
        ### Apply Transforms ###
        ########################
        moco = ants.apply_transforms(fixed, moving, transforms)
        warped = moco.numpy()
        warps.append(warped)

    return warps