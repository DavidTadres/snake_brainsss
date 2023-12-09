# Original file from Ashley
# Modified by David to account for Brezovec style folder structure:
#/20220307
#    /fly_0
#         /func_0
#               /T-Series
#               /SingleImage
#         /anat_0
#    /fly_1
#        /func_0
#        /anat_0
#
# I also optimized memory allocation which might speed up the process.

import time
import numpy as np
import nibabel as nib
import gc
import os
from pathlib import Path
import natsort
import xml.etree.ElementTree as ET
import traceback


def nii_stitcher(x_resolution, y_resolution, frames_per_stack, no_of_stacks,
                 sorted_channel_list, current_folder, savename):
    # Preallocate empty numpy array
    full_brain = np.zeros((x_resolution, y_resolution, frames_per_stack, no_of_stacks),
                          dtype=np.uint16)
    counter = 1
    for current_file in sorted_channel_list:
        current_start_index = int(current_file.split('_s')[-1].split('.nii')[0])
        # For most instances, the split function made nii files with 500 frames per file
        if counter < len(sorted_channel_list):
            if buggy_brukerbridge:
                full_brain[:, :, :, current_start_index:current_start_index + 499] = \
                    nib.load(Path(current_folder, current_file)).get_fdata().astype(np.uint16)
            else:
                full_brain[:, :, :, current_start_index:current_start_index + 500] = \
                    nib.load(Path(current_folder, current_file)).get_fdata().astype(np.uint16)
        # Except in the last one - Since we know how many frames we expect (see above), it's just
        # the rest of the preallocated numpy array.
        else:
            full_brain[:, :, :, current_start_index::] = \
                nib.load(Path(current_folder, current_file)).get_fdata().astype(np.uint16)
        counter += 1
    # save stiched brain
    aff = np.eye(4)  # https://nipy.org/nibabel/coordinate_systems.html
    img = nib.Nifti1Image(full_brain, aff)
    img.to_filename(Path(current_folder, savename))
    del full_brain  # to delete from memory
    del img  # to delete from memory
    gc.collect()  # extra delete from memory
    time.sleep(30)  ##to give to time to delete


# get to files
#dates = ['20231201']  # must be a string

folder_name_to_target = 'func' # All my folders with functional imaging are called func, e.g. 'func1', 'func2' etc.

buggy_brukerbridge = False # early version of brukerbridge omitted one sequence (z-stack) per split file. Account for this

def find_split_files(dataset_path):
    #for current_date in dates:
    #     print('STARTING DATE:', str(current_date))
        #dataset_path = Path("/oak/stanford/groups/trc/data/David/Bruker/imports/", current_date)
        #dataset_path = Path("/Volumes/groups/trc/data/David/Bruker/imports", current_date)

    for current_fly_folder in Path(str(dataset_path)).iterdir():
        print(current_fly_folder.name)
        #directory = os.path.join(dataset_path, fly)
        # Find folders that are called 'func1', 'func2' etc.
        for current_func_dir in current_fly_folder.iterdir():
            if folder_name_to_target in current_func_dir.name:
                #print(current_func_dir)
                for tseries_folder in Path(current_func_dir).iterdir():
                    if 'TSeries' in tseries_folder.name:

                        files = os.listdir(tseries_folder)
                        channel_1_list = []
                        channel_2_list = []
                        ch1_already_stitched = False
                        ch2_already_stitched = False

                        for file in files:
                            ## stitch brain ##
                            # append all appropriate nii files together
                            # these need to be appended in order so first make a list of ch specific files then sort later

                            if "channel_1" in file and "nii" in file:
                                channel_1_list.append(file)
                            elif "channel_2" in file and "nii" in file:
                                channel_2_list.append(file)
                            elif "xml" in file and not "Voltage" in file:
                                xml_file_name = file
                            # Check if the stitched files already exist
                            if 'ch1_stitched.nii' in file:
                                ch1_already_stitched = True
                            if 'ch2_stitched.nii' in file:
                                ch2_already_stitched = True


                        # How many sequences does the microscope report on having have recorded
                        xml_file = ET.parse(Path(tseries_folder, xml_file_name)).getroot()
                        no_of_stacks = 0
                        for current_xml_entry in xml_file:
                            if "Sequence" in str(current_xml_entry):
                                no_of_stacks += 1

                        # How many frames in z does the microscope report on having recorded?
                        frames_per_stack = len(xml_file[3].findall('Frame'))

                        # X/Y resolution
                        for current_setting in xml_file[1].findall('PVStateValue'):
                            # How many lines per frame (resolution in y)
                            if "linesPerFrame" in current_setting.attrib['key']:
                                y_resolution = int(current_setting.attrib['value'])
                            # How many pixel per line (resolution in x)
                            if "pixelsPerLine" in current_setting.attrib['key']:
                                x_resolution = int(current_setting.attrib['value'])

                        print('Expected final shape of each file: ' + \
                              repr(x_resolution) + ', ' + repr(y_resolution) + ',' + \
                              repr(frames_per_stack) + ', ' + repr(no_of_stacks))

                        # natsorted should sort print as expected.
                        sorted_channel_1_list = natsort.natsorted(channel_1_list)
                        sorted_channel_2_list = natsort.natsorted(channel_2_list)

                        print('sorted_channel_1_list', sorted_channel_1_list)
                        print('sorted_channel_2_list', sorted_channel_2_list)

                        ########
                        # Channel 1
                        ########
                        if len(sorted_channel_1_list) > 0 and not ch1_already_stitched:
                            print('loading split files for Ch1 in ' + str(tseries_folder) )
                            try:
                                nii_stitcher(x_resolution=x_resolution,
                                             y_resolution=y_resolution,
                                             frames_per_stack=frames_per_stack,
                                             no_of_stacks=no_of_stacks,
                                             sorted_channel_list=sorted_channel_1_list,
                                             current_folder=tseries_folder,
                                             savename='ch1_concat.nii',# it is important this is saved as ch1 rather than channel so
                                                # it doesn't try to get restitched if the code runs twice
                                             )

                                print('CH1 COMPLETE for: ', str(tseries_folder))
                            except Exception:
                                traceback.print_exc()
                        ########
                        # Channel 2
                        ########
                        if len(sorted_channel_2_list)> 0 and not ch2_already_stitched:
                            print('loading split files for Ch2 in ' + str(tseries_folder) )
                            try:
                                nii_stitcher(x_resolution=x_resolution,
                                             y_resolution=y_resolution,
                                             frames_per_stack=frames_per_stack,
                                             no_of_stacks=no_of_stacks,
                                             sorted_channel_list=sorted_channel_2_list,
                                             current_folder=tseries_folder,
                                             savename='ch2_concat.nii',# it is important this is saved as ch1 rather than channel so
                                                # it doesn't try to get restitched if the code runs twice
                                             )

                                print('CH2 COMPLETE for: ', str(tseries_folder))
                            except Exception:
                                traceback.print_exc()
