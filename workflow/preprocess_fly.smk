"""
Here the idea is to do preprocessing a little bit differently:

We assume that the fly_builder already ran and that the fly_dir exists.

Note:
    Although jobs can directly read and write to $OAK during execution,
    it is recommended to first stage files from $OAK to $SCRATCH at the
    beginning of a series of jobs, and save the desired results back from
    $SCRATCH to $OAK at the end of the job campaign.

    We strongly recommend using $OAK to reference your group home directory in
    scripts, rather than its explicit path.
AND:
    Each compute node has a low latency, high-bandwidth Infiniband link to $SCRATCH.
    The aggregate bandwidth of the filesystem is about 75GB/s. So any job with high
    data performance requirements will take advantage from using $SCRATCH for I/O.
"""

# On sherlock using config file type:
# ml python/3.9.0
# source .env_snakemake/bin/activate
# cd snake_brainsss/workflow
# snakemake -s preprocess_fly.smk --profile profiles/sherlock
# OR
# snakemake -s preprocess_fly.smk --profile profiles/simple_slurm

######
fly_folder_to_process = 'fly_002' # folder to be processed
# ONLY ONE FLY PER RUN for now. The path must be relative to
# what you set in your 'user/username.json' file under 'dataset_path'
# in my case, it's 'user/dtadres.json and it says "/oak/stanford/groups/trc/data/David/Bruker/preprocessed"
#####

# the name of the file in 'user' that you want to use. Ideally it's your SUNet ID
current_user = 'dtadres'

#>>>>
fictrac_fps = 50 # AUTOMATE THIS!!!! ELSE BUG PRONE!!!!
#<<<<

# First n frames to average over when computing mean/fixed brain | Default None
# (average over all frames).
meanbrain_n_frames =  None

##############################################
import pathlib
import json
import datetime

scripts_path = pathlib.Path(__file__).resolve()  # path of workflow i.e. /Users/dtadres/snake_brainsss/workflow
#print(scripts_path)
from brainsss import utils
from scripts import preprocessing
from scripts import snake_utils

#### KEEP for future
# SCRATCH_DIR
#SCRATCH_DIR = '/scratch/users/' + current_user
#print(SCRATCH_DIR)
####

settings = utils.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])
imports_path = pathlib.Path(settings['imports_path'])

# Define path to imports to find fly.json!
fly_folder_to_process_oak = pathlib.Path(dataset_path,fly_folder_to_process)
print('Analyze data in ' + repr(fly_folder_to_process_oak.as_posix()))

# Read channel information from fly.json file
with open(pathlib.Path(fly_folder_to_process_oak, 'fly.json'), 'r') as file:
    fly_json = json.load(file)

ANATOMY_CHANNEL = fly_json['anatomy_channel'] # < This needs to come from some sort of json file the experimenter
# creates while running the experiment. Same as genotype.
FUNCTIONAL_CHANNELS = fly_json['functional_channel']

def ch_exists_func(channel):
    """
    Check if a given channel exists in global variables ANATOMY_CHANNEL and FUNCTIONAL_CHANNELS
    :param channel:
    :return:
    """
    if 'channel_' + str(channel) in ANATOMY_CHANNEL or 'channel_' + str(channel) in FUNCTIONAL_CHANNELS:
        ch_exists = True
    else:
        ch_exists = False
    return(ch_exists)

CH1_EXISTS = ch_exists_func("1")
CH2_EXISTS = ch_exists_func("2")
CH3_EXISTS = ch_exists_func("3")

####
# Load fly_dir.json
####
width = 120 # can go into a config file as well.
fly_dirs_dict_path = pathlib.Path(fly_folder_to_process_oak, fly_folder_to_process_oak.name + '_dirs.json')
with open(pathlib.Path(fly_dirs_dict_path),'r') as file:
    fly_dirs_dict = json.load(file)

#####
# Prepare filepaths to be used
#####
# Snakemake runs acyclical, meaning it checks which input depends on the output of which rule
# in order to parallelize a given snakefile.
# I'll therefore keep variables with lists of paths that will be feed into a given rule.
# These lists of paths are created here.

# Imaging data paths
imaging_file_paths = []
fictrac_file_paths = []
for key in fly_dirs_dict:
    if 'Imaging' in key:
        imaging_file_paths.append(fly_dirs_dict[key][1::])
    elif 'Fictrac' in key:
        fictrac_file_paths.append(fly_dirs_dict[key][1::])
def create_path_func(fly_folder_to_process, list_of_paths, filename='', func_only=False):
    """
    Creates lists of path that can be feed as input/output to snakemake rules
    :param fly_folder_to_process: a folder pointing to a fly, i.e. /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_001
    :param list_of_paths: a list of path created, usually created from fly_dirs_dict (e.g. fly_004_dirs.json)
    :param filename: filename to append at the end. Can be nothing (i.e. for fictrac data).
    :return: list of full paths as pathlib.Path objects to a file based on the list_of_path provided
    """
    final_path = []
    for current_path in list_of_paths:
        # Sometimes we need only paths from the functional channel, for example for z-scoring
        if func_only:
            if 'func' in current_path:
                final_path.append(pathlib.Path(fly_folder_to_process,current_path,filename))
            else:
                pass # if no func, don't use the path!
        else:
            final_path.append(pathlib.Path(fly_folder_to_process, current_path, filename))

    return(final_path)

def create_output_path_func(list_of_paths, filename):
    """
    :param list_of_paths: expects a list of paths pointing to a file, for example from variable full_fictrac_file_paths
    :param filename: filename
    returns a list paths as pathlib.Path objects pointing to where the output file is going to be expecte4d
    """
    final_path = []
    for current_path in list_of_paths:
        #print("current_path" + repr(current_path))
        if isinstance(current_path, list):
            # This is for lists of lists, for example created by :func: create_paths_each_experiment
            # If there's another list assume that we only want one output file!
            # For example, in the bleaching_qc the reason we have a list of lists is because each experiment
            # is plotted in one file. Hence, the output should be only one file
            #print(pathlib.Path(pathlib.Path(current_path[0]).parent, filename))
            try:
                final_path.append(pathlib.Path(pathlib.Path(current_path[0]).parent, filename))
            except IndexError:
                pass # Happens because we don't always hvae two functional channels
        else:
            final_path.append(pathlib.Path(pathlib.Path(current_path).parent, filename))

    return(final_path)

#######
# Data path on OAK
#######
def create_file_paths(path_to_fly_folder, imaging_file_paths, filename, func_only=False):
    """
    Creates lists of path that can be feed as input/output to snakemake rules taking into account that
    different fly_00X folder might have different channels!
    :param path_to_fly_folder: a folder pointing to a fly, i.e. /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_001
    :param list_of_paths: a list of path created, usually created from fly_dirs_dict (e.g. fly_004_dirs.json)
    :param filename: filename to append at the end. Can be nothing (i.e. for fictrac data).
    :param func_only: Sometimes we need only paths from the functional channel, for example for z-scoring
    :return: list of filepaths
    """
    list_of_filepaths = []
    for current_path in imaging_file_paths:
        if func_only:
            if 'func' in current_path:
                if CH1_EXISTS:
                    list_of_filepaths.append(pathlib.Path(path_to_fly_folder, current_path, 'channel_1' + filename))
                if CH2_EXISTS:
                    list_of_filepaths.append(pathlib.Path(path_to_fly_folder, current_path, 'channel_2' + filename))
                if CH3_EXISTS:
                    list_of_filepaths.append(pathlib.Path(path_to_fly_folder, current_path, 'channel_3' + filename))
        else:
            if CH1_EXISTS:
                list_of_filepaths.append(pathlib.Path(path_to_fly_folder,current_path,'channel_1' + filename))
            if CH2_EXISTS:
                list_of_filepaths.append(pathlib.Path(path_to_fly_folder,current_path,'channel_2' + filename))
            if CH3_EXISTS:
                list_of_filepaths.append(pathlib.Path(path_to_fly_folder,current_path,'channel_3' + filename))
    return(list_of_filepaths)

# This will create path to the imaging files that exists so we'll get a list like this:
# ['/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_001/anat1/imaging/channel_1.nii',
#  '/Volumes/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_001/anat1/imaging/channel_2.nii', ...]
all_imaging_oak_paths = create_file_paths(path_to_fly_folder=fly_folder_to_process_oak,
                                        imaging_file_paths=imaging_file_paths,
                                         filename='.nii')

# >>>> This will hopefully change
# Fictrac files are named non-deterministically (could be changed of course) but for now
# the full filename is in the fly_dirs_dict
full_fictrac_file_oak_paths = create_path_func(fly_folder_to_process_oak, fictrac_file_paths)
# <<<<

# Path for make_mean_brain_rule
# will look like this: [PosixPath('../data/../imaging/channel_1'),PosixPath('../data/../imaging/channel_2')]
paths_for_make_mean_brain_oak = create_file_paths(path_to_fly_folder=fly_folder_to_process_oak,
                                                imaging_file_paths=imaging_file_paths,
                                                filename='')

# List of paths for moco
list_of_imaging_paths_moco = []
for current_path in imaging_file_paths:
    list_of_imaging_paths_moco.append(current_path.split('/imaging')[0])
# List of paths for zscore
imaging_paths_zscore = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_zscore.append(current_path.split('/imaging')[0])
# list of paths for temporal highpass filter
# identical to zscore imaging paths but for ease of readibility, explicitly create a new one
imaging_paths_temp_HP_filter = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_temp_HP_filter.append(current_path.split('/imaging')[0])
####

'''
#######
# Data path on SCRATCH < Not working yet, might not be necessary!!!
#######
def convert_oak_path_to_scratch(oak_path):
    """

    :param oak_path: expects a list of path, i.e. ch1_func_file_oak_paths OR a single pathlib.Path object
    :return: list of paths as pathlib.objects OR a single pathlib.Path object
    """
    #print("OAK PATH" + repr(oak_path))
    if isinstance(oak_path, list):
        all_scratch_paths = []
        for current_path in oak_path:
            # The [1::] is again necessary because split leads to something like /David/Bruker etc. which ignores preceeding parts
            relevant_path_part = current_path.as_posix().split('data')[-1][1::] # data seems to be the root folder everyone is using
            all_scratch_paths.append(pathlib.Path(SCRATCH_DIR, 'data', relevant_path_part))
        return(all_scratch_paths)

    elif oak_path.is_dir():
        relevant_path_part = oak_path.as_posix().split('data')[-1][1::]
        return(pathlib.Path(SCRATCH_DIR, 'data', relevant_path_part))

    else:
        print('oak_path needs to be a list of pathlib.Path objects or a single pathlib.Path object')
        print('You provided: ' + repr(oak_path) +'. This might lead to a bug.')

#all_imaging_scratch_paths = convert_oak_path_to_scratch(all_imaging_oak_paths)
#print("all_imaging_scratch_paths" + repr(all_imaging_scratch_paths))

#paths_for_make_mean_brain_rule_scratch = convert_oak_path_to_scratch(paths_for_make_mean_brain_rule_oak)
#print("paths_for_make_mean_brain_rule_scratch" + repr(paths_for_make_mean_brain_rule_scratch))

#fly_folder_to_process_scratch = convert_oak_path_to_scratch(fly_folder_to_process_oak) # Must be provided as a list
'''

####
# Path per folder
# This is a bit different to path above as function (in this case bleaching) requires 2 or three input files
####
# Some scripts require to have path organized in lists per experiment. See docstring in
# :create_paths_each_experiment: function for example
def create_paths_each_experiment(imaging_file_paths):
    """
    get paths to imaging data as list of lists, i.e.
    [[func0/channel1.nii, func0/channel2.nii], [func1/channel1.nii, func1/channel2.nii]]
    :param imaging_file_paths: a list with all func and anat path as defined in 'fly_00X_dirs.json'
    """
    imaging_path_by_folder_oak = []
    #imaging_path_by_folder_scratch = []
    for current_path in imaging_file_paths:
        #if 'func' in current_path:
        temp = []
        if CH1_EXISTS:
            temp.append(pathlib.Path(fly_folder_to_process_oak, current_path, 'channel_1.nii'))
        if CH2_EXISTS:
            temp.append(pathlib.Path(fly_folder_to_process_oak, current_path, 'channel_2.nii'))
        if CH3_EXISTS:
            temp.append(pathlib.Path(fly_folder_to_process_oak, current_path, 'channel_3.nii'))
        imaging_path_by_folder_oak.append(temp)
        #imaging_path_by_folder_scratch.append([
        #    pathlib.Path(SCRATCH_DIR, 'data' + imaging_path_by_folder_oak[-1][0].as_posix().split('data')[-1]),
        #    pathlib.Path(SCRATCH_DIR, 'data' + imaging_path_by_folder_oak[-1][1].as_posix().split('data')[-1])])
    #return(imaging_path_by_folder_oak, imaging_path_by_folder_scratch)
    return (imaging_path_by_folder_oak)

#imaging_paths_by_folder_oak, imaging_paths_by_folder_scratch = create_paths_each_experiment(imaging_file_paths)
imaging_paths_by_folder_oak = create_paths_each_experiment(imaging_file_paths)

#####
# Output data path
#####
# Output files for fictrac_qc rule
print("full_fictrac_file_oak_paths" + repr(full_fictrac_file_oak_paths))
fictrac_output_files_2d_hist_fixed = create_output_path_func(list_of_paths=full_fictrac_file_oak_paths,
                                                             filename='fictrac_2d_hist_fixed.png')
bleaching_qc_output_files = create_output_path_func(list_of_paths=imaging_paths_by_folder_oak,
                                                           filename='bleaching.png')

rule all:
    """
    See: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html
        By default snakemake executes the first rule in the snakefile. This gives rise to pseudo-rules at the beginning 
        of the file that can be used to define build-targets similar to GNU Make
    Or in other words: Here we define which file we want at the end. Snakemake checks which one is there and which 
    one is missing. It then uses the other rules to see how it can produce the missing files.
    """
    threads: 1 # should be sufficent
    resources: mem_mb=1000 # should be sufficient
    input:
        ###
        # Fictrac QC
        ###
        expand("{fictrac_output}", fictrac_output=fictrac_output_files_2d_hist_fixed),
        ###
        # Bleaching QC
        ###
        bleaching_qc_output_files,
        ###
        # Meanbrain
        ###
        expand("{mean_brains_output}_mean.nii", mean_brains_output=paths_for_make_mean_brain_oak),
        # Motion correction output
        # While we don't really need this image, it's a good idea to have it here because the empty h5 file
        # we actually want is created very early during the rule call and will be present even if the program
        # crashed.
        expand(str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/motcorr_params.npy", moco_imaging_paths=list_of_imaging_paths_moco),
        # depending on which channels are present,
        expand(str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_1_moco.h5" if CH1_EXISTS else[], moco_imaging_paths=list_of_imaging_paths_moco),
        expand(str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_2_moco.h5" if CH2_EXISTS else[], moco_imaging_paths=list_of_imaging_paths_moco),
        expand(str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_3_moco.h5" if CH3_EXISTS else[],moco_imaging_paths=list_of_imaging_paths_moco),
        ####
        # Z-score
        ####
        #>expand(str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_1_moco_zscore.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[], zscore_imaging_paths=imaging_paths_zscore),
        #>expand(str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_2_moco_zscore.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[], zscore_imaging_paths=imaging_paths_zscore),
        #>expand(str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_3_moco_zscore.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[], zscore_imaging_paths=imaging_paths_zscore),
        ###
        # temporal high-pass filter
        ###
        #>expand(str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore_highpass.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[], temp_HP_filter_imaging_paths=imaging_paths_temp_HP_filter),
        #>expand(str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore_highpass.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[], temp_HP_filter_imaging_paths=imaging_paths_temp_HP_filter),
        #>expand(str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore_highpass.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[], temp_HP_filter_imaging_paths=imaging_paths_temp_HP_filter)

rule fictrac_qc_rule:
    threads: 1
    input:
        full_fictrac_file_oak_paths
        #fictrac_file_paths = expand("{fictrac}", fictrac=full_fictrac_file_scratch_paths)
    output:
        expand("{fictrac_output}", fictrac_output=fictrac_output_files_2d_hist_fixed)
    run:
        try:
            preprocessing.fictrac_qc(fly_folder_to_process_oak,
                                    fictrac_file_paths= full_fictrac_file_oak_paths,
                                    fictrac_fps=fictrac_fps # AUTOMATE THIS!!!! ELSE BUG PRONE!!!!
                                    )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_fictrac_qc_rule')
            utils.write_error(logfile=logfile,
                                 error_stack=error_stack,
                                 width=width)

rule bleaching_qc_rule:
    """
    Out of memory with 1 & 4 threads on sherlock.
    With 8 I had a ~45% memory utiliziation which seems ok as in the test dataset I had a 
    30 minute volumetric recording.
    
    # some timing testing (3 func folders, 2 small ones with 4Gb imaging data) and 1 large (20Gb)  
    using oak as data source: 
        16 threads: runtime 5 min, 34 seconds
        8  threads: runtime (1) 2 min, 48 seconds (2) 3 min, 05 seconds
    using scratch as data source:
        8 threads: runtime (1) 3 min, 38 seconds (2) 35 seconds
          
    Try to properly parallelize code here: IF want output file X, run rule with file Y. 
    Might not be possible in this case: input is EITHER 
    '../functional_channel1.nii', '../functional_channel2.nii'
    OR 
    '../anatomical_channel1.nii',, '../anatomical_channel2.nii.
    whereas output is always 
    '../bleaching.png'.
    As (I think) I need to use the same wildcard as in the rule all:
    I don't see how this could be achieved.
    If we had only 1 kind of input files, e.g.
    '../channel_1.nii', '../channel_2.nii'
    we could do define do the following
    
    rule all:
        input:
            expand("{path_to_imaging_folder}/bleaching.png", path_to_imaging_folder=all_imaging_oak_paths)
    
    rule bleaching_qc_rule:
        input:
            "{path_to_imaging_folder}/channel_1.nii", "{path_to_imaging_folder}/channel_2.nii", 
        output:
            {path_to_imaging_folder}/bleaching.png"
            
    path_to_imaging_folder would need to be a list of paths pointing to 'imaging', for example:
    ['../fly_004/func0/imaging', '../fly_004/func1/imaging]
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_threads
    input:
        imaging_paths_by_folder_oak
    output:
        bleaching_qc_output_files
    run:
        try:
            preprocessing.bleaching_qc(fly_directory=fly_folder_to_process_oak,
                                        imaging_data_path_read_from=imaging_paths_by_folder_oak, #imaging_paths_by_folder_scratch, # {input} didn't work, I think because it destroyed the list of list we expect to see here #imaging_paths_by_folder_scratch,
                                        imaging_data_path_save_to=bleaching_qc_output_files # can't use output, messes things up here! #imaging_paths_by_folder_oak
                                        #print_output = output
            )
            print('Done with bleaching_qc')
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_bleaching_qc_rule')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)
            print('Error with bleaching_qc' )

rule motion_correction_rule:
    """
    Benchmarking: Two 3Gb files required 19Gb (43% of 44Gb at 6 cores). 1 horu
    
    Tried a bunch of stuff and finally settled on this not super elegant solution.
    
    1) Figure out how many channels exist somehwere before rule all
    2) To have a single wildcard for optimal parallelization, check if a given channel exists and pass nothing if not.
    3) This should be relatively stable if I sometimes only record e.g. Ch1 or only Ch2 or both. I also added a 3rd 
       channel as Jacob's upgrade is due soon. 
    
    Has two sets of input files:
   ../imaging/..channel_1.nii & ../imaging/..channel_2.nii 
    and
    ../imaging/..channel_1_mean.nii & ../imaging/..channel_2_mean.nii 
    
    output is (maybe on scratch?)
    ../imaging/moco/..channel_1_moco.h5 & ../imaging/moco/..channel_2_moco.h5
    
    Because it's a pain to handle filenames that are called differently, I stopped using 
    'anatomy_channel_x' and 'functional_channel_x' and just call it 'channel_x'.
    
    Avoid explicitly calling ch1 and ch2. Will be a pain to adapt code everytime if no ch2 is recorded...
    
    Goal: We want the correct number of 'moco/h5' files. We can have either 1 or 2 (or in the future 3)
    different output files.
    For a given file we need a defined set of input files.
    If we find we want moco/ch1.h5 AND moco/ch2.h5 we need /ch1.nii, /ch2.nii, /ch1_mean.nii and /ch2_mean.nii
    if we only want moco/ch1.h5 we only need /ch1.nii and /ch1_mean.nii
    
    """
    threads: 6
    resources: mem_mb=snake_utils.mem_mb_times_threads
    input:
        # Only use the Channels that exists
        brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_1.nii" if CH1_EXISTS else [],
        brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_2.nii" if CH2_EXISTS else [],
        brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_3.nii" if CH3_EXISTS else [],

        mean_brain_paths_ch1= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_1_mean.nii" if CH1_EXISTS else [],
        mean_brain_paths_ch2= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_2_mean.nii" if CH2_EXISTS else [],
        mean_brain_paths_ch3= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_3_mean.nii" if CH3_EXISTS else [],
    output:
        h5_path_ch1 = str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_1_moco.h5" if CH1_EXISTS else[],
        h5_path_ch2= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_2_moco.h5" if CH2_EXISTS else[],
        h5_path_ch3= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_3_moco.h5" if CH3_EXISTS else[],
        par_output = str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/motcorr_params.npy"
    run:
        try:
            preprocessing.motion_correction(fly_directory=fly_folder_to_process_oak,
                                            dataset_path=[input.brain_paths_ch1, input.brain_paths_ch2, input.brain_paths_ch3],
                                            meanbrain_path=[input.mean_brain_paths_ch1, input.mean_brain_paths_ch2, input.mean_brain_paths_ch3], # NOTE must be input file! # filename of precomputed target meanbrain to register to
                                            type_of_transform="SyN",# For ants.registration(), see ANTsPy docs | Default 'SyN'
                                            output_format='h5', #'OPTIONAL PARAM output_format MUST BE ONE OF: "h5", "nii"'
                                            flow_sigma=3,# For ants.registration(), higher sigma focuses on coarser features | Default 3
                                            total_sigma=0,  # For ants.registration(), higher values will restrict the amount of deformation allowed | Default 0
                                            aff_metric='mattes',# For ants.registration(), metric for affine registration | Default 'mattes'. Also allowed: 'GC', 'meansquares'
                                            h5_path=[output.h5_path_ch1, output.h5_path_ch2, output.h5_path_ch3], # Define as dataset on scratch!
                                            anatomy_channel=ANATOMY_CHANNEL,
                                            functional_channels=FUNCTIONAL_CHANNELS
                                            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_motion_correction')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule temporal_high_pass_filter_rule:
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_threads
    input:
        zscore_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    output:
        temp_HP_filter_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore_highpass.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        temp_HP_filter_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore_highpass.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        temp_HP_filter_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore_highpass.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[],

    run:
        try:
            preprocessing.temporal_high_pass_filter(fly_directory=fly_folder_to_process_oak,
                                                    dataset_path=[input.zscore_path_ch1,
                                                                  input.zscore_path_ch2,
                                                                  input.zscore_path_ch3],
                                                    temporal_high_pass_filtered_path=[output.temp_HP_filter_path_ch1,
                                                                                      output.temp_HP_filter_path_ch2,
                                                                                      output.temp_HP_filter_path_ch3])
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_temporal_high_pass_filter')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule zscore_rule:
    """
    Benchmarking: Did 2.5*input file size and got a 86% efficiency on the memory, 4 threads only 12% efficiency
    Did same with 1 thread and seemed to be enough. Keep at 1 thread for now, might break with larger files.
    """
    threads: 1
    resources: mem_mb=snake_utils.mem_mb_times_input
    input:
        h5_path_ch1 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/moco/channel_1_moco.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        h5_path_ch2 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/moco/channel_2_moco.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        h5_path_ch3 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/moco/channel_3_moco.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[],

    output:
        zscore_path_ch1 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_1_moco_zscore.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_2_moco_zscore.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_3_moco_zscore.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    run:
        try:
            preprocessing.zscore(fly_directory=fly_folder_to_process_oak,
                                dataset_path=[input.h5_path_ch1, input.h5_path_ch2, input.h5_path_ch3],
                                zscore_path=[output.zscore_path_ch1, output.zscore_path_ch2, output.zscore_path_ch3])
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_zscore')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule make_mean_brain_rule:
    """
    Tested with 16 threads, overkill as we wouldn't normally need more than 10Gb
    of memory (each thread is ~8Gb)
    
    Here it should be possible to parallelize quite easily as each input file creates
    one output file!
    
    input would be something like
    paths_to_use = ['../fly_004/func0/imaging/functional_channel_1', '../fly_004/func1/imaging/functional_channel_2']
    
    rule all would request the 'mean' brain of each of those
    expand("{imaging_path}_mean.nii", imaging_path=paths_to_use)
    
    rule make_mean_brain_rule:
        input: "{imaging_path}.nii"
        output: "{imaging_path}_mean.nii"
        run: function(imaging_path)
    
    which will do:
        brain = read(input)
        mean_brain = mean(brain)
        save.mean_brain(output)
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_threads
    input: "{mean_brains_output}.nii" #'/Users/dtadres/Documents/functional_channel_1.nii'

    output: "{mean_brains_output}_mean.nii" # '/Users/dtadres/Documents/functional_channel_1_mean.nii'
    # every nii file is made to a mean brain! Can predict how they
        # are going to be called and put them here.
    run:
        try:
            preprocessing.make_mean_brain(fly_directory=fly_folder_to_process_oak,
                                          meanbrain_n_frames=meanbrain_n_frames,
                                          path_to_read=input,
                                          path_to_save=output  )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_make_mean_brain')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

"""
https://farm.cse.ucdavis.edu/~ctbrown/2023-snakemake-book-draft/chapter_9.html
While wildcards and expand use the same syntax, they do quite different things.

expand generates a list of filenames, based on a template and a list of values to insert
into the template. It is typically used to make a list of files that you want snakemake 
to create for you.

Wildcards in rules provide the rules by which one or more files will be actually created. 
They are recipes that say, "when you want to make a file with name that looks like THIS, 
you can do so from files that look like THAT, and here's what to run to make that happen.

"""

"""
Maybe useful in the future
filenames_present = ['channel_1', 'channel_2'] # I should be able to easily automatically determine this!
rule all:
    input: expand(str(fly_folder_to_process) + "/{moco_imaging_paths}/moco/{moco_filenames}_moco.h5",
            moco_filenames=filenames_present,
            moco_imaging_paths=imaging_file_paths)
rule X:
    input:        
        brain_path = str(fly_folder_to_process) + "/{moco_imaging_paths}/{moco_filenames}.nii",
        mean_brain_paths = str(fly_folder_to_process) + "/{moco_imaging_paths}/{moco_filenames}_mean.nii"
    output:
        str(fly_folder_to_process) + "/{moco_imaging_paths}/moco/{moco_filenames}_moco.h5"

yields:
[Wed Dec 20 14:32:32 2023]
rule motion_correction_rule:
    input: /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_1.nii, /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_1_mean.nii
    output: /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/moco/channel_1_moco.h5
    jobid: 3
    reason: Missing output files: /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/moco/channel_1_moco.h5
    wildcards: moco_imaging_paths=func0/imaging, moco_filenames=channel_1
    threads: 4
    resources: tmpdir=/var/folders/q0/9m96f32j1pl31hsqz1jkr4mr0000gq/T

[Wed Dec 20 14:32:32 2023]
rule motion_correction_rule:
    input: /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_2.nii, /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_2_mean.nii
    output: /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/moco/channel_2_moco.h5
    jobid: 4
    reason: Missing output files: /Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/moco/channel_2_moco.h5
    wildcards: moco_imaging_paths=func0/imaging, moco_filenames=channel_2
    threads: 4
    resources: tmpdir=/var/folders/q0/9m96f32j1pl31hsqz1jkr4mr0000gq/T

"""

##
# https://stackoverflow.com/questions/68882363/snakemake-use-part-of-input-file-name-for-the-output
# expand("start-{sample}_{info}-end_R1.fastq.gz", zip, sample=ID,info=INFO)
##


# how to use expand example
# https://stackoverflow.com/questions/55776952/snakemake-write-files-from-an-array



"""
rule blueprint

rule name:
    input: Files that must be present to run. if not present, will look for other rules that produce
            files needed here
    output: Files produced by this rule. If rule is run and file is not produced, will produce error
    run/shell: python (run) or shell command to be run.
    

#def get_time():
#    day_now = datetime.datetime.now().strftime("%Y%d%m")
#    time_now = datetime.datetime.now().strftime("%I%M%S")
#    return(day_now + '_' + time_now)
# time_string = get_time() # To write benchmark files

# Note : ["{dataset}/a.txt".format(dataset=dataset) for dataset in DATASETS]
         # is the same as expand("{dataset}/a.txt", dataset=DATASETS)
         # https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#
         # With expand we can also do:
         # expand("{dataset}/a.{ext}", dataset=DATASETS, ext=FORMATS) 
"""
