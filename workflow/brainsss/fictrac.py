import numpy as np
import sys
import os
import scipy
import scipy.signal
import pandas as pd
from scipy.interpolate import interp1d
import pathlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg') # Agg, is a non-interactive backend that can only write to files.
# Without this I had the following error: Starting a Matplotlib GUI outside of the main thread will likely fail.

def load_fictrac(directory, file='fictrac.dat'):
    """ Loads fictrac data from .dat file that fictrac outputs.

    To-do: change units based on diameter of ball etc.
    For speed sanity check, instead remove bad frames so we don't have to throw out whole trial.

    Parameters
    ----------
    directory: string of full path to file
    file: string of file name

    Returns
    -------
    fictrac_data: pandas dataframe of all parameters saved by fictrac """

    for item in os.listdir(directory):
      if '.dat' in item:
        file = item

    with open(os.path.join(directory, file),'r') as f:
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

def interpolate_fictrac(fictrac, timestamps, fps, dur, behavior='speed',sigma=3,sign=None):
    """ Interpolate fictrac.

    Parameters
    ----------
    fictrac: fictrac pandas dataframe.
    timestamps: [t,z] numpy array of imaging timestamps (to interpolate to).
    fps: camera frame rate (Hz)
    dur: int, duration of fictrac recording (in ms)
    behavior: column of dataframe to use
    sigma: for smoothing

    Returns
    -------
    fictrac_interp: [t,z] numpy array of fictrac interpolated to timestamps

    """
    camera_rate = 1/fps * 1000 # camera frame rate in ms
    raw_fictrac_times = np.arange(0,dur,camera_rate)
    
    # Cut off any extra frames (only happened with brain 4)
    fictrac = fictrac[:90000]
 
    if behavior == 'my_speed':
      dx = np.asarray(fictrac['dRotLabX'])
      dy = np.asarray(fictrac['dRotLabY'])
      dx = scipy.ndimage.filters.gaussian_filter(dx,sigma=3)
      dy = scipy.ndimage.filters.gaussian_filter(dy,sigma=3)
      fictrac_smoothed = np.sqrt(dx*dx + dy*dy)
    elif behavior == 'speed_all_3':
      dx = np.asarray(fictrac['dRotLabX'])
      dy = np.asarray(fictrac['dRotLabY'])
      dz = np.asarray(fictrac['dRotLabZ'])
      dx = scipy.ndimage.filters.gaussian_filter(dx,sigma=3)
      dy = scipy.ndimage.filters.gaussian_filter(dy,sigma=3)
      dz = scipy.ndimage.filters.gaussian_filter(dz,sigma=3)
      fictrac_smoothed = np.sqrt(dx*dx + dy*dy + dz*dz)
    else:
      fictrac_smoothed = scipy.ndimage.filters.gaussian_filter(np.asarray(fictrac[behavior]),sigma=sigma)

    if sign is not None and sign == 'abs':
      fictrac_smoothed = np.abs(fictrac_smoothed)
    elif sign is not None and sign == 'plus':
      fictrac_smoothed = np.clip(fictrac_smoothed,a_min=0,a_max=None)
    elif sign is not None and sign == 'minus':
      fictrac_smoothed = np.clip(fictrac_smoothed,a_min=None,a_max=0)
    elif sign is not None and sign == 'df':
      fictrac_smoothed = np.append(np.diff(fictrac_smoothed),0)
    elif sign is not None and sign == 'df_abs':
      fictrac_smoothed = np.abs(np.append(np.diff(fictrac_smoothed),0))

    # Interpolate
    # Warning: interp1d set to fill in out of bounds times
    fictrac_interp_temp = interp1d(raw_fictrac_times, fictrac_smoothed, bounds_error = False)
    fictrac_interp = fictrac_interp_temp(timestamps)
    
    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False);
    
    return fictrac_interp

def smooth_and_interp_fictrac(fictrac, fps, resolution, expt_len, behavior, timestamps=None, smoothing=25, z=None):

    if behavior == 'dRotLabZpos':
      behavior = 'dRotLabZ'
      clip = 'pos'
    elif behavior == 'dRotLabZneg':
      behavior = 'dRotLabZ'
      clip = 'neg'
    else:
      clip = None

    ### get orginal timestamps ###
    camera_rate = 1/fps * 1000 # camera frame rate in ms
    x_original = np.arange(0,expt_len,camera_rate)

    ### smooth ###
    fictrac_smoothed = scipy.signal.savgol_filter(np.asarray(fictrac[behavior]),smoothing,3)

    ### clip if desired ###
    if clip == 'pos':
      fictrac_smoothed = np.clip(fictrac_smoothed, a_min=0, a_max=None)
    elif clip == 'neg':
      fictrac_smoothed = np.clip(fictrac_smoothed, a_min=None, a_max=0)*-1

    ### interpolate ###
    fictrac_interp_temp = interp1d(x_original, fictrac_smoothed, bounds_error = False)
    xnew = np.arange(0,expt_len,resolution) #0 to last time at subsample res
    if timestamps is None:
      fictrac_interp = fictrac_interp_temp(xnew)
    else:
      fictrac_interp = fictrac_interp_temp(timestamps[:,z])

    ### convert units for common cases ###
    sphere_radius = 4.5e-3 # in m
    if behavior in ['dRotLabY']:
        ''' starts with units of rad/frame
        * sphere_radius(m); now in m/frame
        * fps; now in m/sec
        * 1000; now in mm/sec '''
        
        fictrac_interp = fictrac_interp * sphere_radius * fps * 1000 # now in mm/sec
        
    if behavior in ['dRotLabZ']:
        ''' starts with units of rad/frame
        * 180 / np.pi; now in deg/frame
        * fps; now in deg/sec '''
        
        fictrac_interp = fictrac_interp * 180 / np.pi * fps
    
    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False);
    
    return fictrac_interp

def make_2d_hist(fictrac, fictrac_folder, full_id, save=True, fixed_crop=True):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    norm = mpl.colors.LogNorm()
    hist = ax.hist2d(fictrac['Y'], fictrac['Z'], bins=100, cmap='Blues', norm=norm);
    ax.set_ylabel('Rotation, deg/sec')
    ax.set_xlabel('Forward, mm/sec')
    ax.set_title('Behavior 2D hist {}'.format(full_id))
    # For colorbar - get coordinates of ax to set it to the right
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(hist, cax=cax, orientation='vertical')
    name = 'fictrac_2d_hist.png'
    if fixed_crop:
        ax.set_ylim(-400, 400)
        ax.set_xlim(-10, 15)
        name = 'fictrac_2d_hist_fixed.png'
    if save:
        #fname = os.path.join(fictrac_folder, name)
        fname = pathlib.Path(fictrac_folder, name)
        fig.savefig(fname, dpi=100, bbox_inches='tight')

def make_velocity_trace(fictrac, fictrac_folder, full_id, xnew, save=True):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(xnew/1000,fictrac['Y'],color='xkcd:dusk')
    ax.set_ylabel('forward velocity mm/sec')
    ax.set_xlabel('time, sec')
    ax.set_title(full_id)
    if save:
        #fname = os.path.join(fictrac_folder, 'velocity_trace.png')
        fname = pathlib.Path(fictrac_folder, 'velocity_trace.png')
        fig.savefig(fname,dpi=100,bbox_inches='tight')