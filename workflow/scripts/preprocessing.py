import json
import numpy as np
from shutil import copyfile
from xml.etree import ElementTree as ET
from lxml import etree, objectify
from openpyxl import load_workbook
import pathlib
import sys
import time
import traceback
import natsort
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg') # Agg, is a non-interactive backend that can only write to files.
# Without this I had the following error: Starting a Matplotlib GUI outside of the main thread will likely fail.
import nibabel as nib
import shutil

####################
# GLOBAL VARIABLES #
####################
WIDTH = 120 # This is used in all logging files

# To import brainsss, define path to scripts!
scripts_path = pathlib.Path(__file__).parent.resolve()  # path of brainsss
sys.path.insert(0, pathlib.Path(scripts_path, 'workflow'))
# print(pathlib.Path(scripts_path, 'workflow'))
import brainsss
def copy_to_scratch(fly_directory, paths_on_oak, paths_on_scratch):
    """
    For faster reading and writing, it might be worth putting the nii
    files (and other large files) on $SCRATCH and only save the result
    on oak:
    https://www.sherlock.stanford.edu/docs/storage/filesystems/#scratch
        Each compute node has a low latency, high-bandwidth Infiniband
        link to $SCRATCH. The aggregate bandwidth of the filesystem is
        about 75GB/s. So any job with high data performance requirements
         will take advantage from using $SCRATCH for I/O.
    :return:
    """
    #printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    logfile = brainsss.create_logfile(fly_directory, function_name='copy_to_scratch')
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    brainsss.print_function_start(logfile, WIDTH, 'copy_to_scratch')

    #width = 120
    # For log file readability clearly indicate when function was called

    for current_file_src, current_file_dst in zip(paths_on_oak, paths_on_scratch):

        # make folder if not exist
        pathlib.Path(current_file_dst).parent.mkdir(exist_ok=True, parents=True)
        # copy file
        shutil.copy(current_file_src, current_file_dst)
        printlog('Copied: ' + repr(current_file_dst))

def make_mean_brain(fly_directory,
                    meanbrain_n_frames,
                    path_to_read,
                    path_to_save):
    """
    Function to calculate
    :param meanbrain_n_frames:
    :return:
    """
    logfile = brainsss.create_logfile(fly_directory, function_name='make_mean_brain')
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    brainsss.print_function_start(logfile, WIDTH, 'make_mean_brain')

    # standalone = True  # I'll add if statements to be able to go back to Bella's script easliy
    # args = {'logfile': logfile, 'directory': directory, 'files': files}

    """#if standalone:
        #new logfile
        #import time
        #width = 120  # width of print log
        #logfile = './logs/' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
        #printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
        #sys.stderr = brainsss.Logger_stderr_sherlock(logfile)
        #brainsss.print_title(logfile, width)

        # get path!
        #scripts_path = args['PWD']
        #com_path = os.path.join(scripts_path, 'com')
        #user = scripts_path.split('/')[3]
        #settings = brainsss.load_user_settings(user, scripts_path)

        ### Parse user settings
        #imports_path = settings['imports_path']
        #dataset_path = settings['dataset_path']

        #directory = '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001/func1/imaging'

        #files = []
        #for current_file in pathlib.Path(directory).iterdir():
        #    if 'anatomy_channel' in current_file.name or 'functional_channel' in current_file.name \
        #            and not 'mean' in current_file.name: # this last one is necessary so that if script is run twice, mean is still calculated only once
        #        files.append(current_file.name)

        
        # Copy from preprocess.py
        for funcanat, dirtype in zip(funcanats, dirtypes):
            directory = os.path.join(funcanat, 'imaging')

            if dirtype == 'func':
                files = ['functional_channel_1.nii', 'functional_channel_2.nii']
            if dirtype == 'anat':
                files = ['anatomy_channel_1.nii', 'anatomy_channel_2.nii']
            """
    #else:
        #logfile = args['logfile']
        #directory = args['directory'] # directory will be a full path to either an anat/imaging folder or a func/imaging folder
        #files = args['files']


    #meanbrain_n_frames = args.get('meanbrain_n_frames', None)  # First n frames to average over when computing mean/fixed brain | Default None (average over all frames)
    #width = 120
    #printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')

    # Check if files is just a single file path string
    #if type(files) is str:
    #    files = [files]

    brain_data = np.asarray(nib.load(path_to_read).get_fdata(), dtype='uint16')
    if meanbrain_n_frames is not None:
        # average over first meanbrain_n_frames frames
        meanbrain = np.mean(brain_data[..., :int(meanbrain_n_frames)], axis=-1)
    else:  # average over all frames
        meanbrain = np.mean(brain_data, axis=-1)

    aff = np.eye(4)
    object_to_save = nib.Nifti1Image(meanbrain, aff)
    object_to_save.to_filename(path_to_save)
    fly_print = fly_directory.name
    func_print = path_to_read.split('/')[-2]
    printlog(F"meanbrn | COMPLETED | {fly_print} | {func_print} | {file} | {brain_data.shape} ===> {meanbrain.shape}")

    for file in files:

        try:
            ### make mean ###
            #full_path = os.path.join(directory, file)
            #if full_path.endswith('.nii'):
            #    brain = np.asarray(nib.load(full_path).get_fdata(), dtype='uint16')
            #elif full_path.endswith('.h5'):
            #    with h5py.File(full_path, 'r') as hf:
            #        brain = np.asarray(hf['data'][:], dtype='uint16')

            #if meanbrain_n_frames is not None:
            #    # average over first meanbrain_n_frames frames
            #    meanbrain = np.mean(brain[...,:int(meanbrain_n_frames)], axis=-1)
            #else: # average over all frames
            #    meanbrain = np.mean(brain, axis=-1)

            ### Save ###
            #save_file = os.path.join(directory, file[:-4] + '_mean.nii')
            #aff = np.eye(4)
            #img = nib.Nifti1Image(meanbrain, aff)
            #img.to_filename(save_file)

            #fly_func_str = ('|').join(path_to_read.split('/')[-3:-1])
            fly_print = fly_directory.name
            func_print = path_to_read.split('/')[-2]
            #printlog(f"COMPLETE | {fly_func_str} | {file} | {brain.shape} --> {meanbrain.shape}")
            printlog(F"meanbrn | COMPLETED | {fly_print} | {func_print} | {file} | {brain.shape} ===> {meanbrain.shape}")
            #if not standalone:
            #    print(brain.shape[-1]) ### IMPORTANT: for communication to main
            brain = None
        except FileNotFoundError:
            printlog(F"Not found (skipping){file:.>{WIDTH-20}}")
            #printlog(f'{file} not found.')

def bleaching_qc(fly_directory,
                 imaging_data_path_read_from,
                 imaging_data_path_save_to
                 ):
    """
    Perform bleaching qc.
    This is based on Bella's 'bleaching_qc.py' script

    :param: logfile: logfile to be used for all errors (stderr) and console outputs (stdout)
    :param imaging_data_path_read_from: a nested(!) list to a 'fly' folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
                                        This is a list containing pathlibt.Path objects like this:
                                        [[PosixPath('../fly_001/func0/imaging/functional_channel_1.nii'),
                                          PosixPath('../fly_001/func0/imaging/functional_channel_2.nii')],
                                         [PosixPath('../fly_001/func1/imaging/functional_channel_1.nii'),
                                          PosixPath('../fly_001/func1/imaging/functional_channel_2.nii')]
                                        ]
    :param imaging_data_path_save_to: a list to the 'bleaching' target file as pathlib.Path objects like this:
                                      [PosixPath('../fly_001/func0/imaging/bleaching.png'),
                                       PosixPath('../fly_001/func1/imaging/bleaching.png)]

    :return:
    """
    # Logging
    logfile = brainsss.create_logfile(fly_directory, function_name='bleaching_qc')
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    brainsss.print_function_start(logfile, WIDTH, 'bleaching_qc')
    #

    # For each experiment,
    for current_folder_read, current_file_path_save in zip(imaging_data_path_read_from, imaging_data_path_save_to):
        # yields e.g. current_folder_read = PosixPath('../fly_001/func0/imaging/functional_channel_1.nii'),
        #                                           PosixPath('../fly_001/func0/imaging/functional_channel_2.nii')]
        # and current_file_path_save = PosixPath('../fly_001/func0/imaging/bleaching.png')
        data_mean = {}
        for current_file_path_read in current_folder_read:
            # yields e.g. '../fly_001/func0/imaging/functional_channel_1.nii')
            printlog(F"Currently reading: {current_file_path_read.name:.>{WIDTH - 20}}")
            # Read data
            brain = np.asarray(nib.load(current_file_path_read).get_fdata(), dtype=np.uint16)
            # take the mean of ALL values
            data_mean[pathlib.Path(current_file_path_read).name] = np.mean(brain, axis=(0,1,2))
        ##############################
        ### Output Bleaching Curve ###
        ##############################
        # plotting params
        plt.rcParams.update({'font.size': 24})
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        signal_loss = {}

        for filename in data_mean:
            xs = np.arange(len(data_mean[filename]))
            color='k'
            if 'functional_channel_1.nii' in filename:
                color='red'
            if 'functional_channel_2.nii' in filename:
                color='green'
            ax.plot(data_mean[filename],color=color,label=filename)
            # Fit polynomial to mean fluorescence.
            linear_fit = np.polyfit(xs, data_mean[filename], 1)
            # and plot it
            ax.plot(np.poly1d(linear_fit)(xs),color='k',linewidth=3,linestyle='--')
            # take the linear fit to calculate how much signal is lost and report it as the title
            signal_loss[filename] = linear_fit[0]*len(data_mean[filename])/linear_fit[1]*-100
        ax.set_xlabel('Frame Num')
        ax.set_ylabel('Avg signal')
        loss_string = ''
        for filename in data_mean:
            loss_string = loss_string + filename + ' lost' + F'{int(signal_loss[filename])}' +'%\n'
        ax.set_title(loss_string, ha='center', va='bottom')

        # Save plot
        save_file = pathlib.Path(current_file_path_save)
        fig.savefig(save_file,dpi=300,bbox_inches='tight')
        printlog(F"Prepared plot and saved as: {str(save_file):.>{WIDTH - 20}}")

        # release memory
        del data_mean

def fictrac_qc(fly_directory, fictrac_file_paths, fictrac_fps):
    """
    Perform fictrac quality control.
    This is based on Bella's fictrac_qc.py  script.
    #:param logfile: logfile to be used for all errors (stderr) and console outputs (stdout)
    :param fly_directory: a pathlib.Path object to a 'fly' folder such as '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/fly_001'
    :param fictrac_file_paths: a list of paths as strings
    :return:
    """
    #printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    logfile = brainsss.create_logfile(fly_directory, function_name='fictrac_qc')
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    brainsss.print_function_start(logfile, WIDTH, 'fictrac_qc')

    #width = 120
    # For log file readability clearly indicate when function was called

    for current_file in fictrac_file_paths:
        printlog('Currently looking at: ' + repr(current_file))
        fictrac_raw = brainsss.load_fictrac(current_file)
        # I expect this to yield something like 'fly_001/func0/fictrac
        full_id = ', '.join(str(current_file).split('/')[-3:1])

        resolution = 10  # desired resolution in ms # Comes from Bella!
        expt_len = fictrac_raw.shape[0] / fictrac_fps * 1000
        behaviors = ['dRotLabY', 'dRotLabZ']
        fictrac = {}
        for behavior in behaviors:
            if behavior == 'dRotLabY':
                short = 'Y'
            elif behavior == 'dRotLabZ':
                short = 'Z'
            fictrac[short] = brainsss.smooth_and_interp_fictrac(fictrac_raw, fictrac_fps, resolution, expt_len, behavior)
        xnew = np.arange(0, expt_len, resolution)

        brainsss.make_2d_hist(fictrac, current_file, full_id, save=True, fixed_crop=True)
        brainsss.make_2d_hist(fictrac, current_file, full_id, save=True, fixed_crop=False)
        brainsss.make_velocity_trace(fictrac, current_file, full_id, xnew, save=True)

def fly_builder(logfile, user, dirs_to_build, target_folder):
    """
    Move folders from imports to fly dataset - need to restructure folders.
    This is based on Bella's 'fly_builder.py' script

    :param logfile: logfile to be used for all errors (stderr) and console outputs (stdout)
    :param user: your SUnet ID as a string
    :param dirs_to_build: a list of folders to build. e.g. dir_to_build = ['20231301'] or  dir_to_build = ['20232101', '20231202']
    :param target_folder:
    :return:
    """
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    # printlog('\nBuilding flies from directory {}'.format(flagged_dir))
    # For log file readability clearly indicate when function was called
    brainsss.print_function_start(logfile, WIDTH, 'fly_builder')

    # To be consistent with Bella's script, might be removed later
    destination_fly = target_folder
    destination_fly.mkdir(parents=True, exist_ok=True)  # Don't use 'exist_ok=True' to make sure we get an error if folder exists!
    printlog(F'Created fly directory:{str(destination_fly.name):.>{WIDTH - 22}}')

    # Create a dict that will save all paths were data is saved save it in the folder
    # to streamline downstream analysis and to make explicit where a given folder (e.g.
    # func0 imaging data) can be found.
    fly_dirs_dict = {}
    fly_dirs_dict['fly ID'] = destination_fly.name

    ### Parse user settings
    settings = brainsss.load_user_settings(user)
    imports_path = pathlib.Path(settings['imports_path'])
    #dataset_path = pathlib.Path(settings['dataset_path'])

    paths_to_build = []
    for current_dir in dirs_to_build:
        paths_to_build.append(pathlib.Path(imports_path, current_dir))

    # get fly folders in flagged directory and sort to ensure correct fly order
    printlog(F'Building flies from: {str(paths_to_build):.>{WIDTH - 22}}')
    #likely_fly_folders = os.listdir(flagged_dir)
    #brainsss.sort_nicely(likely_fly_folders)

    # loop through each of the provided dirs_to_build
    for current_path_to_build in paths_to_build:
        # Make sure that each fly folder is actually containing the keyword 'fly'
        likely_fly_folders = [i for i in current_path_to_build.iterdir() if 'fly' in i.name]
        printlog(F"Found fly folders{str(likely_fly_folders):.>{WIDTH - 17}}")

        for current_fly_folder in likely_fly_folders:

            printlog(f"\n{'   Building ' + current_fly_folder.name + ' as ' + str(target_folder.name) + '   ':-^{WIDTH}}")

            # Copy fly data
            fly_dirs_dict = copy_fly(current_fly_folder, destination_fly, printlog, user, fly_dirs_dict)

            # Add date to fly.json file
            try:
                add_date_to_fly(destination_fly)
            except Exception as e:
                printlog(str(e))
                printlog(str(e))
                printlog(traceback.format_exc())


            # Add json metadata to master dataset
            try:
                add_fly_to_xlsx(destination_fly, printlog)
            except Exception as e:
                printlog('Could not add xls data because of error:')
                printlog(str(e))
                printlog(traceback.format_exc())

    """# How many anat folder?
    no_of_anat_folders = 0
    no_of_func_folders = 0
    for current_path in fly_dirs_dict:
        if 'anat' in current_path:
            no_of_anat_folders += 1
        elif 'func' in current_path:
            no_of_func_folders +=1
    fly_dirs_dict['# of anatomy folders'] = no_of_anat_folders
    fly_dirs_dict['# of functional folders'] = no_of_func_folders"""

    return(fly_dirs_dict)

def add_date_to_fly(destination_fly):
    ''' get date from xml file and add to fly.json'''

    ### Get date
    try:  # Check if there are func folders
        # Get func folders
        #func_folders = [os.path.join(destination_fly, x) for x in os.listdir(destination_fly) if 'func' in x]
        func_folders = [pathlib.Path(destination_fly, x) for x in destination_fly.iterdir() if 'func' in x.name]
        #brainsss.sort_nicely(func_folders)
        func_folders = natsort.natsorted(func_folders)
        func_folder = func_folders[0]  # This throws an error if no func folder, hence try..except
        # Get full xml file path
        xml_file = pathlib.Path(func_folder, 'imaging', 'functional.xml')
        #xml_file = os.path.join(func_folder, 'imaging',
        #                        'functional.xml')  # Unsure how this leads to correct filename!
    except:  # Use anatomy folder
        # Get anat folders
        #anat_folders = [os.path.join(destination_fly, x) for x in os.listdir(destination_fly) if 'anat' in x]
        anat_folders = [pathlib.Path(destination_fly, x) for x in destination_fly.iterdir() if 'anat' in x]
        #brainsss.sort_nicely(anat_folders)
        anat_folders = natsort.natsorted((anat_folders))
        anat_folder = anat_folders[0]
        # Get full xml file path
        xml_file = pathlib.Path(anat_folder, 'imaging', 'anatomy.xml')
        #xml_file = os.path.join(anat_folder, 'imaging', 'anatomy.xml')  # Unsure how this leads to correct filename!
    # Extract datetime
    datetime_str, _, _ = get_datetime_from_xml(xml_file)
    # Get just date
    date = datetime_str.split('-')[0]
    time = datetime_str.split('-')[1]

    ### Add to fly.json
    json_file = pathlib.Path(destination_fly, 'fly.json')
    #json_file = os.path.join(destination_fly, 'fly.json')
    with open(json_file, 'r+') as f:
        metadata = json.load(f)
        metadata['date'] = str(date)
        metadata['time'] = str(time)
        f.seek(0)
        json.dump(metadata, f, indent=4)
        f.truncate()

def copy_fly(current_fly_folder, destination_fly, printlog, user, fly_dirs_dict):
    """
    There will be two types of folders in a fly folder.
    1) func_x folder
    2) anat_x folder
    For functional folders, need to copy fictrac and visual as well
    For anatomy folders, only copy folder. There will also be
    3) fly json data
    """

    # look at every item in source fly folder
    for current_file_or_folder in current_fly_folder.iterdir():
        # This should be e.g. directory such as anat1 or func0 or fly.json
        print('Currently looking at source_item: {}'.format(current_file_or_folder.name))
        # Handle folders

        if current_file_or_folder.is_dir():
            # Call this folder source expt folder
            current_imaging_folder = current_file_or_folder
            # Make the same folder in destination fly folder
            current_target_folder = pathlib.Path(destination_fly, current_imaging_folder.name)
            current_target_folder.mkdir(parents=True)

            # Is this folder an anatomy or functional folder?
            if 'anat' in current_imaging_folder.name:
                # If anatomy folder, just copy everything
                # Make imaging folder and copy
                #imaging_destination = os.path.join(expt_folder, 'imaging')
                imaging_destination = pathlib.Path(current_target_folder, 'imaging')
                #os.mkdir(imaging_destination)
                imaging_destination.mkdir(parents=True)
                copy_bruker_data(current_imaging_folder, imaging_destination, 'anat', printlog)
                current_fly_dir_dict = str(imaging_destination).split(imaging_destination.parents[1].name)[-1]
                fly_dirs_dict[current_imaging_folder.name + ' Imaging'] = current_fly_dir_dict
                ######################################################################
                print(f"anat:{current_target_folder}")  # IMPORTANT - FOR COMMUNICATING WITH MAIN
                ######################################################################
            elif 'func' in current_imaging_folder.name:
                # Make imaging folder and copy
                #imaging_destination = os.path.join(expt_folder, 'imaging')
                imaging_destination = pathlib.Path(current_target_folder, 'imaging')
                #os.mkdir(imaging_destination)
                imaging_destination.mkdir(parents=True)
                copy_bruker_data(current_imaging_folder, imaging_destination, 'func', printlog)
                # Update fly_dirs_dict
                current_fly_dir_dict = str(imaging_destination).split(imaging_destination.parents[1].name)[-1]
                fly_dirs_dict[current_imaging_folder.name + ' Imaging'] = current_fly_dir_dict
                # Copy fictrac data based on timestamps
                try:
                    copy_fictrac(current_target_folder, printlog, user, current_imaging_folder, fly_dirs_dict)
                    # printlog('Fictrac data copied')
                except Exception as e:
                    printlog('Could not copy fictrac data because of error:')
                    printlog(str(e))
                    printlog(traceback.format_exc())
                # Copy visual data based on timestamps, and create visual.json
                try:
                    copy_visual(current_target_folder, printlog)
                except Exception as e:
                    printlog('Could not copy visual data because of error:')
                    printlog(str(e))

                ######################################################################
                #print(f"func:{expt_folder}")  # IMPORTANT - FOR COMMUNICATING WITH MAIN
                ######################################################################
                # REMOVED TRIGGERING

            else:
                printlog('Invalid directory in fly folder (skipping): {}'.format(current_imaging_folder.name))

        # Copy fly.json file
        else:
            current_file = current_file_or_folder
            if current_file_or_folder.name == 'fly.json':
                ##print('found fly json file')
                ##sys.stdout.flush()
                #source_path = os.path.join(source_fly, item)
                #target_path = os.path.join(destination_fly, item)
                target_path = pathlib.Path(destination_fly, current_file.name)
                ##print('Will copy from {} to {}'.format(source_path, target_path))
                ##sys.stdout.flush()
                copyfile(current_file, target_path)
            else:
                printlog('Invalid file in fly folder (skipping): {}'.format(current_file.name))
                ##sys.stdout.flush()

    return(fly_dirs_dict)

def copy_bruker_data(source, destination, folder_type, printlog):
    # Do not update destination - download all files into that destination
    #for item in os.listdir(source):
    for source_path in source.iterdir():
        # Check if item is a directory
        if source_path.is_dir():
            # Do not update destination - download all files into that destination
            copy_bruker_data(source_path, destination, folder_type, printlog)
            # In my case this leads to /oak/stanford/groups/trc/data/David/Bruker/imports/20231201/fly2/func1
            # The code then calls itself with this path which then goes one folder deeper

        # If the item is a file
        else:
            ### Change file names and filter various files
            # Don't copy these files
            if 'SingleImage' in source_path.name:
                continue
            # Rename functional file to functional_channel_x.nii
            if 'concat.nii' in source_path.name and folder_type == 'func':
                target_name = 'functional_channel_' + source_path.name.split('ch')[1].split('_')[0] + '.nii'
                target_path = pathlib.Path(destination, target_name)
            elif '.nii' in source_path.name and folder_type == 'func':
                continue  # do not copy!! Else we'll copy all the split nii files as well.
                # This is an artifact of splitting the nii file on Brukerbridge and might not be
                # relevant in the future/for other users!
            # Rename anatomy file to anatomy_channel_x.nii
            if '.nii' in source_path.name and folder_type == 'anat':
                target_name = 'anatomy_' + source_path.name.split('_')[1] + '_' + source_path.name.split('_')[2] + '.nii'
                target_path = pathlib.Path(destination, target_name)
            # Special copy for photodiode since it goes in visual folder

            # To be tested once I have such data!!
            if '.csv' in source_path.name:
                source_name = 'photodiode.csv'
                visual_folder_path= pathlib.Path(destination.name, 'visual')
                visual_folder_path.mkdir(exist_ok=True)
                target_path = pathlib.Path(visual_folder_path, source_name)
            # Special copy for visprotocol metadata since it goes in visual folder
            # To be tested once I have such data!!
            if '.hdf5' in source_path.name:
                # Create folder 'visual'
                visual_folder_path = pathlib.Path(destination.name, 'visual')
                visual_folder_path.mkdir(exist_ok=True)
                target_path = pathlib.Path(visual_folder_path, source_path.name)

            # Rename to anatomy.xml if appropriate
            if '.xml' in source_path.name and folder_type == 'anat' and \
                    'Voltage' not in source_path.name:
                target_name = 'anatomy.xml'
                target_path = pathlib.Path(destination, target_name)

            # Rename to functional.xml if appropriate, copy immediately, then make scan.json
            if ('.xml' in source_path.name and folder_type == 'func' and
                    'Voltage' not in source_path.name):
                target_path = pathlib.Path(destination, 'functional.xml')
                #target_item = os.path.join(destination, item)
                copy_file(source_path, target_path, printlog)
                # Create json file
                create_imaging_json(target_path, printlog)
                continue
            if '.xml' in source_path.name and 'VoltageOutput' in source_path.name:
                target_path = pathlib.Path(destination, 'voltage_output.xml')

            # Actually copy the file
            #target_item = os.path.join(destination, item)
            copy_file(source_path, target_path, printlog)

def copy_file(source, target, printlog):
    # printlog('Transfering file {}'.format(target))
    #to_print = ('/').join(target.split('/')[-4:])
    #print('source: ' + str(source))
    #print('target: ' + str(target))
    to_print=str(source.name +' to ' + target.name)
    #width = 120
    printlog(f'Transfering file{to_print:.>{WIDTH - 16}}')
    ##sys.stdout.flush()
    copyfile(source, target)

def copy_visual(destination_region, printlog):
    print('copy_visual NOT IMPLEMENTED YET')
    """width = 120
    printlog(F"Copying visual stimulus data{'':.^{width - 28}}")
    visual_folder = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/imports/visual'
    visual_destination = os.path.join(destination_region, 'visual')

    # Find time of experiment based on functional.xml
    true_ymd, true_total_seconds = get_expt_time(os.path.join(destination_region, 'imaging'))

    # Find visual folder that has the closest datetime
    # First find all folders with correct date, and about the correct time
    folders = []
    for folder in os.listdir(visual_folder):
        test_ymd = folder.split('-')[1]
        test_time = folder.split('-')[2]
        test_hour = test_time[0:2]
        test_minute = test_time[2:4]
        test_second = test_time[4:6]
        test_total_seconds = int(test_hour) * 60 * 60 + \
                             int(test_minute) * 60 + \
                             int(test_second)

        if test_ymd == true_ymd:
            time_difference = np.abs(true_total_seconds - test_total_seconds)
            if time_difference < 3 * 60:
                folders.append([folder, test_total_seconds])
                printlog('Found reasonable visual folder: {}'.format(folder))

    # if more than 1 folder, use the oldest folder
    if len(folders) == 1:
        correct_folder = folders[0]
    # if no matching folder,
    elif len(folders) == 0:
        printlog(F"{'No matching visual folders found; continuing without visual data':.<{width}}")
        return
    else:
        printlog('Found more than 1 visual stimulus folder within 3min of expt. Picking oldest.')
        correct_folder = folders[0]  # set default to first folder
        for folder in folders:
            # look at test_total_seconds entry. If larger, call this the correct folder.
            if folder[-1] > correct_folder[-1]:
                correct_folder = folder

    # now that we have the correct folder, copy it's contents
    printlog('Found correct visual stimulus folder: {}'.format(correct_folder[0]))
    try:
        os.mkdir(visual_destination)
    except:
        pass
        ##print('{} already exists'.format(visual_destination))
    source_folder = os.path.join(visual_folder, correct_folder[0])
    printlog('Copying from: {}'.format(source_folder))
    for file in os.listdir(source_folder):
        target_path = os.path.join(visual_destination, file)
        source_path = os.path.join(source_folder, file)
        ##print('Transfering from {} to {}'.format(source_path, target_path))
        ##sys.stdout.flush()
        copyfile(source_path, target_path)

    # Create visual.json metadata
    # Try block to prevent quiting if visual stimuli timing is wonky (likely went too long)
    try:
        unique_stimuli = brainsss.get_stimuli(visual_destination)
    except:
        unique_stimuli = 'brainsss.get_stimuli failed'
    with open(os.path.join(visual_destination, 'visual.json'), 'w') as f:
        json.dump(unique_stimuli, f, indent=4)"""

def copy_fictrac(destination_region, printlog, user, source_fly, fly_dirs_dict):
    # Make fictrac folder
    fictrac_destination = pathlib.Path(destination_region, 'fictrac')
    fictrac_destination.mkdir(exist_ok=True)
    # Different users have different rule on what to do with the data
    if user == 'brezovec':
        user = 'luke'
    if user == 'yandanw':
        user = 'luke'
    if user == 'ilanazs':
        user = 'luke'
    if user == 'dtadres':
        fictrac_folder = pathlib.Path("/oak/stanford/groups/trc/data/David/Bruker/Fictrac")
        # when doing post-hoc fictrac, Bella's code where one compare the recording
        # timestamps of imaging and fictrac doesn't work anymore.
        # I instead use a deterministic file structure:
        # for example for fly 20231201\fly2\func1 imaging data, fictrac data must
        # be in the folder 20231201_fly2_func1. There must only be a single dat file in that folder.
        source_path = pathlib.Path(fictrac_folder,
                                   source_fly.parts[-3] + '_' + \
                                   source_fly.parts[-2] + '_' + \
                                   source_fly.parts[-1])
        for current_file in source_path.iterdir():
            if 'dat' in current_file.name:
                #width = 120
                #source_path = os.path.join(source_path, file)
                dat_path = current_file
                target_path = pathlib.Path(fictrac_destination, current_file.name)
                to_print = str(target_path)
                printlog(f'Transfering file{to_print:.>{WIDTH - 16}}')

                # put fictrac file path in into fly_dirs_dict
                current_fly_dir_dict = str(target_path).split(fictrac_destination.parents[1].name)[-1]
                fly_dirs_dict[destination_region.name + 'Fictrac '] = current_fly_dir_dict
    else:
        #fictrac_folder = os.path.join("/oak/stanford/groups/trc/data/fictrac", user)
        fictrac_folder = pathlib.Path("/oak/stanford/groups/trc/data/fictrac", user)

        # Find time of experiment based on functional.xml
        #true_ymd, true_total_seconds = get_expt_time(os.path.join(destination_region, 'imaging'))
        true_ymd, true_total_seconds = get_expt_time(pathlib.Path(destination_region, 'imaging'))

        # printlog(f'true_ymd: {true_ymd}; true_total_seconds: {true_total_seconds}')

        # Find .dat file of 1) correct-ish time, 2) correct-ish size
        correct_date_and_size = []
        time_differences = []
        #for file in os.listdir(fictrac_folder):
        for file in fictrac_folder.iterdir():

            file = str(file) # To be changed in the future
            # but I'm currently to lazy to change everything to
            # pathlib object below.

            # must be .dat file
            if '.dat' not in file:
                continue

            # Get datetime from file name
            datetime = file.split('-')[1][:-4]
            test_ymd = datetime.split('_')[0]
            test_time = datetime.split('_')[1]
            test_hour = test_time[0:2]
            test_minute = test_time[2:4]
            test_second = test_time[4:6]
            test_total_seconds = int(test_hour) * 60 * 60 + \
                                 int(test_minute) * 60 + \
                                 int(test_second)

            # Year/month/day must be exact
            if true_ymd != test_ymd:
                continue
            # printlog('Found file from same day: {}'.format(file))

            # Must be correct size
            #fp = os.path.join(fictrac_folder, file)
            fp = pathlib.Path(fictrac_folder, file)
            file_size = fp.stat().st_size
            #file_size = os.path.getsize(fp)
            if file_size < 1000000:  # changed to 1MB to accomidate 1 min long recordings. #30000000: #30MB
                # width = 120
                # printlog(F"Found correct .dat file{file:.>{width-23}}")
                # datetime_correct = datetime
                # break
                continue

            # get time difference from expt
            time_difference = np.abs(true_total_seconds - test_total_seconds)
            # Time must be within 10min
            if time_difference > 10 * 60:
                continue

            # if correct date and size append to list of potential file
            correct_date_and_size.append(file)
            time_differences.append(time_difference)

        # now that we have all potential files, pick the one with closest time
        # except clause will happen if empty list
        try:
            datetime_correct = correct_date_and_size[np.argmin(time_differences)]
        except:
            #width = 120
            printlog(F"{'   No fictrac data found --- continuing without fictrac data   ':*^{WIDTH}}")
            return

        # Collect all fictrac files with correct datetime
        correct_time_files = [file for file in fictrac_folder.iterdir() if datetime_correct in file.name]
        #correct_time_files = [file for file in os.listdir(fictrac_folder) if datetime_correct in file]

        # correct_time_files = []
        # for file in os.listdir(fictrac_folder):
        #     if datetime_correct in file:
        #         correct_time_files.append(file)

        # printlog('Found these files with correct times: {}'.format(correct_time_files))
        ##sys.stdout.flush()

        # Now transfer these 4 files to the fly
        fictrac_folder.mkdir()
        #os.mkdir(fictrac_destination)
        for file in correct_time_files:
            #width = 120
            target_path = pathlib.Path(fictrac_folder, file)
            source_path = pathlib.Path(fictrac_folder, file)
            #target_path = os.path.join(fictrac_destination, file)
            #source_path = os.path.join(fictrac_folder, file)
            #to_print = ('/').join(target_path.split('/')[-4:])
            to_print = str(target_path)
            printlog(f'Transfering file{to_print:.>{WIDTH - 16}}')
            # printlog('Transfering {}'.format(target_path))
            ##sys.stdout.flush()

    copyfile(dat_path, target_path)

    ### Create empty xml file.
    # Update this later
    root = etree.Element('root')
    fictrac = objectify.Element('fictrac')
    root.append(fictrac)
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)
    tree = etree.ElementTree(fictrac)
    #with open(os.path.join(fictrac_destination, 'fictrac.xml'), 'wb') as file:
    with open(pathlib.Path(fictrac_destination, 'fictrac.xml'), 'wb') as file:
        tree.write(file, pretty_print=True)

def create_imaging_json(xml_source_file, printlog):

    # Make empty dict
    source_data = {}

    # Get datetime
    try:
        datetime_str, _, _ = get_datetime_from_xml(xml_source_file)
    except:
        printlog('No xml or cannot read.')
        ##sys.stdout.flush()
        return
    date = datetime_str.split('-')[0]
    time = datetime_str.split('-')[1]
    source_data['date'] = str(date)
    source_data['time'] = str(time)

    # Get rest of data
    tree = objectify.parse(xml_source_file)
    source = tree.getroot()
    statevalues = source.findall('PVStateShard')[0].findall('PVStateValue')
    for statevalue in statevalues:
        key = statevalue.get('key')
        if key == 'micronsPerPixel':
            indices = statevalue.findall('IndexedValue')
            for index in indices:
                axis = index.get('index')
                if axis == 'XAxis':
                    source_data['x_voxel_size'] = float(index.get('value'))
                elif axis == 'YAxis':
                    source_data['y_voxel_size'] = float(index.get('value'))
                elif axis == 'ZAxis':
                    source_data['z_voxel_size'] = float(index.get('value'))
        if key == 'laserPower':
            # I think this is the maximum power if set to vary by z depth - WRONG
            indices = statevalue.findall('IndexedValue')
            laser_power_overall = int(float(indices[0].get('value')))
            source_data['laser_power'] = laser_power_overall
        if key == 'pmtGain':
            indices = statevalue.findall('IndexedValue')
            for index in indices:
                index_num = index.get('index')
                if index_num == '0':
                    source_data['PMT_red'] = int(float(index.get('value')))
                if index_num == '1':
                    source_data['PMT_green'] = int(float(index.get('value')))
        if key == 'pixelsPerLine':
            source_data['x_dim'] = int(float(statevalue.get('value')))
        if key == 'linesPerFrame':
            source_data['y_dim'] = int(float(statevalue.get('value')))
    sequence = source.findall('Sequence')[0]
    last_frame = sequence.findall('Frame')[-1]
    source_data['z_dim'] = int(last_frame.get('index'))

    # Need this try block since sometimes first 1 or 2 frames don't have laser info...
    # try:
    #     # Get laser power of first and last frames
    #     last_frame = sequence.findall('Frame')[-1]
    #     source_data['laser_power'] = int(last_frame.findall('PVStateShard')[0].findall('PVStateValue')[1].findall('IndexedValue')[0].get('value'))
    #     #first_frame = sequence.findall('Frame')[0]
    #     #source_data['laser_power_min'] = int(first_frame.findall('PVStateShard')[0].findall('PVStateValue')[1].findall('IndexedValue')[0].get('value'))
    # except:
    #     source_data['laser_power_min'] = laser_power_overall
    #     source_data['laser_power_max'] = laser_power_overall
    #     #printlog('Used overall laser power.')
    #     # try:
    #     #     first_frame = sequence.findall('Frame')[2]
    #     #     source_data['laser_power_min'] = int(first_frame.findall('PVStateShard')[0].findall('PVStateValue')[1].findall('IndexedValue')[0].get('value'))
    #     #     printlog('Took min laser data from frame 3, not frame 1, due to bruker metadata error.')
    #     # # Apparently sometimes the metadata will only include the
    #     # # laser value at the very beginning
    #     # except:
    #     #     source_data['laser_power_min'] = laser_power_overall
    #     #     source_data['laser_power_max'] = laser_power_overall
    #     #     printlog('Used overall laser power.')

    # Save data
    #with open(os.path.join(os.path.split(xml_source_file)[0], 'scan.json'), 'w') as f:
    with open(pathlib.Path(xml_source_file.parent, 'scan.json'), 'w') as f:
        json.dump(source_data, f, indent=4)

def get_expt_time(directory):
    ''' Finds time of experiment based on functional.xml '''
    xml_file = pathlib.Path(directory, 'functional.xml')
    #xml_file = os.path.join(directory, 'functional.xml')
    _, _, datetime_dict = get_datetime_from_xml(xml_file)
    true_ymd = datetime_dict['year'] + datetime_dict['month'] + datetime_dict['day']
    true_total_seconds = int(datetime_dict['hour']) * 60 * 60 + \
                         int(datetime_dict['minute']) * 60 + \
                         int(datetime_dict['second'])

    ##print('dict: {}'.format(datetime_dict))
    ##print('true_ymd: {}'.format(true_ymd))
    ##print('true_total_seconds: {}'.format(true_total_seconds))
    ##sys.stdout.flush()
    return true_ymd, true_total_seconds

"""def get_fly_time(fly_folder):
    # need to read all xml files and pick oldest time
    # find all xml files
    xml_files = []
    xml_files = get_xml_files(fly_folder, xml_files)

    ##print('found xml files: {}'.format(xml_files))
    ##sys.stdout.flush()
    datetimes_str = []
    datetimes_int = []
    for xml_file in xml_files:
        datetime_str, datetime_int, _ = get_datetime_from_xml(xml_file)
        datetimes_str.append(datetime_str)
        datetimes_int.append(datetime_int)

    # Now pick the oldest datetime
    datetimes_int = np.asarray(datetimes_int)
    ##print('Found datetimes: {}'.format(datetimes_str))
    ##sys.stdout.flush()
    index_min = np.argmin(datetimes_int)
    datetime = datetimes_str[index_min]
    ##print('Found oldest datetime: {}'.format(datetime))
    ##sys.stdout.flush()
    return datetime"""

"""def get_xml_files(fly_folder, xml_files):
    # Look at items in fly folder
    for item in os.listdir(fly_folder):
        full_path = os.path.join(fly_folder, item)
        if os.path.isdir(full_path):
            xml_files = get_xml_files(full_path, xml_files)
        else:
            if '.xml' in item and \
            '_Cycle' not in item and \
            'fly.xml' not in item and \
            'scan.xml' not in item and \
            'expt.xml' not in item:
                xml_files.append(full_path)
                ##print('Found xml file: {}'.format(full_path))
                ##sys.stdout.flush()
    return xml_files"""

def get_datetime_from_xml(xml_file):
    ##print('Getting datetime from {}'.format(xml_file))
    ##sys.stdout.flush()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    datetime = root.get('date')
    # will look like "4/2/2019 4:16:03 PM" to start

    # Get dates
    date = datetime.split(' ')[0]
    month = date.split('/')[0]
    day = date.split('/')[1]
    year = date.split('/')[2]

    # Get times
    time = datetime.split(' ')[1]
    hour = time.split(':')[0]
    minute = time.split(':')[1]
    second = time.split(':')[2]

    # Convert from 12 to 24 hour time
    am_pm = datetime.split(' ')[-1]
    if am_pm == 'AM' and hour == '12':
        hour = str(00)
    elif am_pm == 'AM':
        pass
    elif am_pm == 'PM' and hour == '12':
        pass
    else:
        hour = str(int(hour) + 12)

    # Add zeros if needed
    if len(month) == 1:
        month = '0' + month
    if len(day) == 1:
        day = '0' + day
    if len(hour) == 1:
        hour = '0' + hour

    # Combine
    datetime_str = year + month + day + '-' + hour + minute + second
    datetime_int = int(year + month + day + hour + minute + second)
    datetime_dict = {'year': year,
                     'month': month,
                     'day': day,
                     'hour': hour,
                     'minute': minute,
                     'second': second}

    return datetime_str, datetime_int, datetime_dict

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def load_xml(file):
    tree = objectify.parse(file)
    root = tree.getroot()
    return root


def add_fly_to_xlsx(fly_folder, printlog):

    printlog("Adding fly to master_2P excel log")

    ### TRY TO LOAD ELSX ###
    try:
        xlsx_path = pathlib.Path(fly_folder.parent, 'master_2P.xlsx')
        #xlsx_path = '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/master_2P.xlsx'
        wb = load_workbook(filename=xlsx_path, read_only=False)
        ws = wb.active
        printlog("Sucessfully opened master_2P log")
    except Exception as e:
        printlog("FYI you have no excel metadata sheet found, so unable to append metadata for this fly.")
        printlog(traceback.format_exc())
        return

    ### TRY TO LOAD FLY METADATA ###
    try:
        fly_file = pathlib.Path(fly_folder, 'fly.json')
        #fly_file = os.path.join(fly_folder, 'fly.json')
        fly_data = load_json(fly_file)
        printlog("Sucessfully loaded fly.json")
    except:
        printlog("FYI no *fly.json* found; this will not be logged in your excel sheet.")
        fly_data = {}
        fly_data['circadian_on'] = None
        fly_data['circadian_off'] = None
        fly_data['gender'] = None
        fly_data['age'] = None
        fly_data['temp'] = None
        fly_data['notes'] = None
        fly_data['date'] = None
        fly_data['genotype'] = None

    #expt_folders = []
    #expt_folders = [os.path.join(fly_folder, x) for x in os.listdir(fly_folder) if 'func' in x]
    expt_folders = [pathlib.Path(fly_folder, x) for x in fly_folder.iterdir() if 'func' in x.name]
    #brainsss.sort_nicely(expt_folders)
    expt_folders = natsort.natsorted(expt_folders)
    for expt_folder in expt_folders:

        ### TRY TO LOAD EXPT METADATA ###
        try:
            #expt_file = os.path.join(expt_folder, 'expt.json')
            expt_file = pathlib.Path(expt_folders, 'expt.json')
            expt_data = load_json(expt_file)
            printlog("Sucessfully loaded expt.json")
        except:
            printlog("FYI no *expt.json* found; this will not be logged in your excel sheet.")
            expt_data = {}
            expt_data['brain_area'] = None
            expt_data['notes'] = None
            expt_data['time'] = None

        ### TRY TO LOAD SCAN DATA ###
        try:
            scan_file = pathlib.Path(expt_folder, 'imaging' , 'scan.json')
            #scan_file = os.path.join(expt_folder, 'imaging', 'scan.json')
            scan_data = load_json(scan_file)
            scan_data['x_voxel_size'] = '{:.1f}'.format(scan_data['x_voxel_size'])
            scan_data['y_voxel_size'] = '{:.1f}'.format(scan_data['y_voxel_size'])
            scan_data['z_voxel_size'] = '{:.1f}'.format(scan_data['z_voxel_size'])
            printlog("Sucessfully loaded scan.json")
        except:
            printlog("FYI no *scan.json* found; this will not be logged in your excel sheet.")
            scan_data = {}
            scan_data['laser_power'] = None
            scan_data['PMT_green'] = None
            scan_data['PMT_red'] = None
            scan_data['x_dim'] = None
            scan_data['y_dim'] = None
            scan_data['z_dim'] = None
            scan_data['x_voxel_size'] = None
            scan_data['y_voxel_size'] = None
            scan_data['z_voxel_size'] = None

        visual_file = pathlib.Path(expt_folder, 'visual', 'visual.json')
        #visual_file = os.path.join(expt_folder, 'visual', 'visual.json')
        try:
            visual_data = load_json(visual_file)
            visual_input = visual_data[0]['name'] + ' ({})'.format(len(visual_data))
        except:
            visual_input = None

        # Get fly_id
        fly_folder = expt_folder.parent
        #fly_folder = os.path.split(os.path.split(expt_folder)[0])[-1]
        fly_id = fly_folder.name.split('_')[-1]
        printlog(F"Got fly ID as {fly_id}")

        # Get expt_id
        expt_id = expt_folder.name # probably 'func1' etc.
        printlog(F"Got expt ID as {expt_id}")
        #expt_id = 'NA' # Not sure what this is, NA for now

        # Append the new row
        new_row = []
        new_row = [int(fly_id),
                   str(expt_id),
                   fly_data['date'],
                   expt_data['brain_area'],
                   fly_data['genotype'],
                   visual_input,
                   None,
                   fly_data['notes'],
                   expt_data['notes'],
                   expt_data['time'],
                   fly_data['circadian_on'],
                   fly_data['circadian_off'],
                   fly_data['gender'],
                   fly_data['age'],
                   fly_data['temp'],
                   scan_data['laser_power'],
                   scan_data['PMT_green'],
                   scan_data['PMT_red'],
                   scan_data['x_dim'],
                   scan_data['y_dim'],
                   scan_data['z_dim'],
                   scan_data['x_voxel_size'],
                   scan_data['y_voxel_size'],
                   scan_data['z_voxel_size']]

        ws.append(new_row)
        printlog(F"Appended {fly_id} {expt_id}")

    # Save the file
    wb.save(xlsx_path)
    printlog("master_2P successfully updated")
