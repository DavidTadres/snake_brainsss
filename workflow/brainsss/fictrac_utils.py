import numpy as np
import scipy
import scipy.signal
import pandas as pd
from scipy.interpolate import interp1d
import pathlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
import matplotlib as mpl

mpl.use(
    "agg"
)  # Agg, is a non-interactive backend that can only write to files. Necessary to avoid error: Starting a Matplotlib GUI outside of the main thread will likely fail.


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
            # Unclear why Bella decided to explicitly define this as str.
            # index=str,
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

'''
def interpolate_fictrac(
    fictrac, timestamps, fps, dur, behavior="speed", sigma=3, sign=None
):
    """
    Interpolate fictrac data.

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
    camera_rate = 1 / fps * 1000  # camera frame rate in ms
    raw_fictrac_times = np.arange(0, dur, camera_rate)

    # Cut off any extra frames (only happened with brain 4)
    fictrac = fictrac[:90000]

    if behavior == "my_speed":
        dx = np.asarray(fictrac["dRotLabX"])
        dy = np.asarray(fictrac["dRotLabY"])
        dx = scipy.ndimage.filters.gaussian_filter(dx, sigma=3)
        dy = scipy.ndimage.filters.gaussian_filter(dy, sigma=3)
        fictrac_smoothed = np.sqrt(dx * dx + dy * dy)
    elif behavior == "speed_all_3":
        dx = np.asarray(fictrac["dRotLabX"])
        dy = np.asarray(fictrac["dRotLabY"])
        dz = np.asarray(fictrac["dRotLabZ"])
        dx = scipy.ndimage.filters.gaussian_filter(dx, sigma=3)
        dy = scipy.ndimage.filters.gaussian_filter(dy, sigma=3)
        dz = scipy.ndimage.filters.gaussian_filter(dz, sigma=3)
        fictrac_smoothed = np.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        fictrac_smoothed = scipy.ndimage.filters.gaussian_filter(
            np.asarray(fictrac[behavior]), sigma=sigma
        )

    if sign is not None and sign == "abs":
        fictrac_smoothed = np.abs(fictrac_smoothed)
    elif sign is not None and sign == "plus":
        fictrac_smoothed = np.clip(fictrac_smoothed, a_min=0, a_max=None)
    elif sign is not None and sign == "minus":
        fictrac_smoothed = np.clip(fictrac_smoothed, a_min=None, a_max=0)
    elif sign is not None and sign == "df":
        fictrac_smoothed = np.append(np.diff(fictrac_smoothed), 0)
    elif sign is not None and sign == "df_abs":
        fictrac_smoothed = np.abs(np.append(np.diff(fictrac_smoothed), 0))

    # Interpolate
    # Warning: interp1d set to fill in out of bounds times
    fictrac_interp_temp = interp1d(
        raw_fictrac_times, fictrac_smoothed, bounds_error=False
    )
    fictrac_interp = fictrac_interp_temp(timestamps)

    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False)

    return fictrac_interp'''


def smooth_and_interp_fictrac(
        fictrac_data,
        fictrac_fps,
        expt_len,
        behavior,
        resolution=None,
        neural_timestamps=None,
        z_slice=None
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
    #camera_rate = 1 / fictrac_fps * 1000  # camera frame rate in ms
    #original_fictrac_timestamps = np.arange(
    #    0, expt_len, camera_rate
    #)  # same shape as fictrac (e.g. 20980)
    # Better way to get timestamps:
    # MIGHT ONLY WORK WITH REAL-TIME FICTRAC TRACKING!!!!!
    # Comes in nanoseconds. Want milliseconds!
    original_fictrac_timestamps=(fictrac_data['timeStamp'] - fictrac_data['timeStamp'].iloc[0])/1e6

    ### smooth ###
    # I remove the smoothing input from this function and make it dependent on the fps
    smoothing = int(
        np.ceil(0.25 / (1 / fictrac_fps))
    )  # This will always yield 250 ms (or the next closest
    # possible number, e.g. if we have 50fps we would get a smotthing window of 12.5 which we can't
    # index of course. We always round up so with 50 fps we'd get 13 = 260 ms
    fictrac_smoothed = scipy.signal.savgol_filter(
        np.asarray(fictrac_data[behavior]), smoothing, 3
    )
    # Identical shape in output as input, e.g. 20980

    ### clip if desired ###
    if clip == "pos":
        fictrac_smoothed = np.clip(
            fictrac_smoothed, a_min=0, a_max=None
        )  # Unsure what this does
    elif clip == "neg":
        fictrac_smoothed = (
            np.clip(fictrac_smoothed, a_min=None, a_max=0) * -1
        )  # Unsure what this does

    ### interpolate ###
    # This function probably just returns everything from an input array
    # Here we do use the original fictrac timestamps and assign them to the
    fictrac_interp_temp = interp1d(
        original_fictrac_timestamps, fictrac_smoothed, bounds_error=False
    )  # yields a function
    # ## different number, e.g. 41960, or just 2x shape before.
    # This is probably because resolution is set to 10. If framerate is 50 we have a frame every 20 ms.
    if neural_timestamps is None:
        new_timestamps = np.arange(0, expt_len, resolution)  # 0 to last time at subsample res
        fictrac_interp = fictrac_interp_temp(new_timestamps)
    elif z_slice is not None:  # For testing only! Used for correlation I think
        fictrac_interp = fictrac_interp_temp(neural_timestamps[:, z_slice])
    else:
        # This is how we map fictrac timestamps on neural data
        # fictrac_interp = fictrac_interp_temp(timestamps[:,z]) # This would return ALL timestamps per z slice
        fictrac_interp = fictrac_interp_temp(neural_timestamps)

    ### convert units for common cases ###
    sphere_radius = 4.5e-3  # in m
    if behavior in ["dRotLabY"]:
        """starts with units of rad/frame
        * sphere_radius(m); now in m/frame
        * fps; now in m/sec
        * 1000; now in mm/sec"""

        fictrac_interp = fictrac_interp * sphere_radius * fictrac_fps * 1000  # now in mm/sec

    if behavior in ["dRotLabZ"]:
        """starts with units of rad/frame
        * 180 / np.pi; now in deg/frame
        * fps; now in deg/sec"""

        fictrac_interp = fictrac_interp * 180 / np.pi * fictrac_fps

    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False)

    return(fictrac_interp)

def get_fictrac_fps(fictrac_data):
    """
    This might only work on life tracked fictrac data! TEST
    Failure mode: If more than 50% of timestamps is incorrect.
    """
    fictrac_fps=int(np.nanmedian(round(1 / ((fictrac_data['timeStamp'].diff()) / 1e9))))
    print('Fictrac fps identifed as: ' + repr(fictrac_fps))
    return(fictrac_fps)


def make_2d_hist(ax,fictrac,
                 fictrac_path,
                 #full_id,
                 fixed_crop=True):
    """
    Plot a 2D histogram of rotational and forward speed as reported by fictrac.
    :param fictrac: A dictionary with two keys, "Y" and "Z" with values produced by :func:`fictrac_utils.smooth_and_interp_fictrac`
    :param fictrac_path: a path as a pathlib.Path object
    :param full_id: a string such as 'fly_001/func0", printed on resulting figure
    :param fixed_crop: Boolean, zoom into relevant behavioral space if True
    """
    # Prepare figure
    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111)
    # Prepare normalization for hist2d
    norm = mpl.colors.LogNorm()
    # Plot rotational and forward velocity for the whole experiment.
    hist_plot = ax.hist2d(fictrac["Y"], fictrac["Z"], bins=100, cmap="Blues", norm=norm)
    ax.set_ylabel("Rotation, [deg/sec]")
    ax.set_xlabel("Forward, [mm/sec]")
    #ax.set_title("Behavior 2D hist {}".format(full_id))
    # For colorbar - get coordinates of ax to set it to the right
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(hist_plot[3], cax=cax, orientation="vertical")
    name = "fictrac_2d_hist.png"
    # Zoom into relevant behavioral space
    if fixed_crop:
        ax.set_ylim(-400, 400)
        ax.set_xlim(-10, 15)
        name = "fictrac_2d_hist_fixed.png"
    # Savename
    fname = pathlib.Path(pathlib.Path(fictrac_path).parent, name)

def fictrac_timestamps_QC(ax,
                          fictrac,
                          fictrac_path
                          ):
    # New QC: Make sure timestamps make sense for the duration of the experiment!
    # Difference between timestamps - this is QC for how variable the timestamps are
    timestamp_delta = (fictrac['timeStamp'].diff()) / 1e9
    timestamps_from_start = (fictrac['timeStamp'] - fictrac['timeStamp'].iloc[0]) / 1e9
    # fictrac continues to run even if no frames are coming in. However, timestamps don't
    # change at that point! Hence, as soon as delta between timestamps is 0, we are in a regime
    # where fictrac isn't working on new frames anymore!
    relevant_indeces = np.where(timestamp_delta > 0)[0]

    # Plot delta timestamps - this should help us to quickly spot skipped frames!
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax.plot(timestamps_from_start.iloc[relevant_indeces], timestamp_delta[relevant_indeces])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - ymin * .001, ymax + ymax * .001)
    ax.set_ylabel('Delta between\ntimestamps [s]')
    ax.set_xlabel('Time [s]')
    #ax.set_title("Behavior 2D hist {}".format(full_id))
    ax.set_title('Timestamps QC')
    #fig.tight_layout()

    name = "fictrac_timestamp_QC.png"
    fname = pathlib.Path(pathlib.Path(fictrac_path).parent, name)
    print(fname)
    #fig.savefig(fname, dpi=100, bbox_inches="tight")


def make_velocity_trace(ax,
                        fictrac,
                        time_for_plotting_ms):
    """
    Velocity trace with the duration of the experiment on the x-axis
    :param fictrac: A dictionary with two keys, "Y" and "Z" with values produced by :func:`fictrac_utils.smooth_and_interp_fictrac`
    :param fictrac_path: a path as a pathlib.Path object
    :param full_id:  a string such as 'fly_001/func0", printed on resulting figure
    :param time_for_plotting: numpy array with one timepoint per index, in ms
    :param save:
    :return:
    """
    # Prepare figure
    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111)
    # Plot speed (fictrac["Y"]) for the duration of the experiment
    ax.plot(time_for_plotting_ms/1e3, fictrac["Y"], color="xkcd:dusk")
    ax.set_ylabel("forward velocity\n[mm/sec]")
    ax.set_xlabel("Time [s]")
    #ax.set_title(full_id)
    #savename = pathlib.Path(pathlib.Path(fictrac_path).parent, "velocity_trace.png")
    #fig.savefig(savename, dpi=100, bbox_inches="tight")

def plot_saccades(ax,
                  fictrac,
                  fictrac_fps,
                  fictrac_timestamps_ms):
    """
    As a QC, plot saccades!
    This might have different
    """
    # Show turns on dRotZ
    #fictrac_smoothed = signal.savgol_filter(np.asarray(fictrac_data['dRotLabZ']), 25, 3)
    # We don't always plot the whole series, this is define with min_time and max_time
    turn_indeces = extract_turn_bouts(fictrac_z=fictrac['Z'], fictrac_fps=fictrac_fps,
                                      fictrac_timestamps=fictrac_timestamps_ms/1e3,
                                      minimal_time_between_turns=0.25, turn_thresh=200)

    #fictrac_smoothed = fictrac_smoothed * 180 / np.pi * fictrac_fps  # now in deg/sec
    #fictrac_timestamps = np.arange(0, fictrac_smoothed.shape[0] / fictrac_fps, 1 / fictrac_fps)
    ax.plot(fictrac_timestamps_ms/1e3, fictrac['Z'], alpha=0.5)

    for counter, L in enumerate(turn_indeces['L']):
        ax.scatter([L, L],
                    [fictrac['Z'][int(round(L * fictrac_fps))], fictrac['Z'][int(round(L * fictrac_fps))]],
                    c='g',  # label='Left turns',
                    alpha=1)
    for counter, R in enumerate(turn_indeces['R']):
        ax.scatter([R, R],
                    [fictrac['Z'][int(round(R * fictrac_fps))], fictrac['Z'][int(round(R * fictrac_fps))]],
                    c='r',  # label='Right turns',
                    alpha=1)

    ax.set_ylabel('rot velocity [deg/s]')
    ax.set_xlabel("Time [s]")
    #ax.set_title(full_id)


def extract_turn_bouts(fictrac_z,
                       fictrac_fps,
                       fictrac_timestamps,
                       minimal_time_between_turns,
                       turn_thresh=200):
    """
    From https://github.com/ClandininLab/brezovec_volition_2023/blob/main/predict_turn_direction.py
    """

    minimal_time_between_turns = minimal_time_between_turns * fictrac_fps # convert from seconds to fps!

    ###########################
    ### Identify turn bouts ###
    ###########################

    peaks = {'L': [], 'R': []}
    heights = {'L': [], 'R': []}

    for turn, scalar in zip(['L', 'R'], [1, -1]):
        # identify positive peaks (with 1) or negative peaks (with -1)!
        found_peaks = signal.find_peaks(fictrac_z * scalar, height=turn_thresh,
                                        distance=minimal_time_between_turns)
        pks = found_peaks[0]
        pk_height = found_peaks[1]['peak_heights']

        # convert index to time in seconds!
        peaks[turn] = fictrac_timestamps[pks]
        heights[turn] = pk_height

    return(peaks)

def inter_turn_interval(ax,
                        fictrac,
                        fictrac_fps,
                        fictrac_timestamps_ms):
    """

    """
    # Identify index where turn happens
    turn_indeces = extract_turn_bouts(fictrac_z=fictrac['Z'], fictrac_fps=fictrac_fps,
                                      fictrac_timestamps=fictrac_timestamps_ms / 1e3,
                                      minimal_time_between_turns=0.25, turn_thresh=200)

    # Combine all turns in one array
    all_turns = []
    for current_turn_indeces in turn_indeces:
        for current_turn in turn_indeces[current_turn_indeces]:
            all_turns.append(current_turn)
    all_turns = np.array(sorted(all_turns))

    # Plotting
    counts, edges = np.histogram(np.diff(all_turns), bins=50)
    ax.stairs(counts, edges, fill=True)
    ax.set_yscale('log')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Log scale')
    ax.set_title('Inter turn interval')