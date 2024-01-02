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
fly_folder_to_process = 'SS94676_DNa01_x_UAS-CD8-GFP/fly_002' # folder to be processed
# ONLY ONE FLY PER RUN for now. The path must be relative to
# what you set in your 'user/username.json' file under 'dataset_path'
# in my case, it's 'user/dtadres.json and it says "/oak/stanford/groups/trc/data/David/Bruker/preprocessed"
#####

# the name of the file in 'user' that you want to use. Ideally it's your SUNet ID
current_user = 'dtadres'

#>>>>
fictrac_fps = 100 # AUTOMATE THIS!!!! ELSE FOR SURE A MISTAKE WILL HAPPEN IN THE FUTURE!!!!
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
with open(pathlib.Path(fly_folder_to_process_oak, 'fly.json'), 'r') as file: # If fails here, means the folder specified doesn't exist. Check name
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
        # this yields for example 'func2/imaging'
    elif 'Fictrac' in key:
        fictrac_file_paths.append(fly_dirs_dict[key][1::])
        # This yields for example 'func1/fictrac/fictrac_behavior_data.dat'
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
'''
# Maybe not used anymore. Might be useful to create paths to SCRATCH, though...
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
    return(list_of_filepaths)'''

# >>>> This will hopefully change
# Fictrac files are named non-deterministically (could be changed of course) but for now
# the full filename is in the fly_dirs_dict
full_fictrac_file_oak_paths = create_path_func(fly_folder_to_process_oak, fictrac_file_paths)
# <<<<

FICTRAC_PATHS = []
for current_path in fictrac_file_paths:
    FICTRAC_PATHS.append(current_path.split('/fictrac_behavior_data.dat')[0])

###
# List of paths for meanbrain
imaging_paths_meanbrain =[]
for current_path in imaging_file_paths:
    imaging_paths_meanbrain.append(current_path.split('/imaging')[0])
channels = []
if CH1_EXISTS:
    channels.append("1")
if CH2_EXISTS:
    channels.append("2")
if CH3_EXISTS:
    channels.append("3")

##
# List of paths for bleaching
# Identical to imaging_paths_meanbrain but define explicitly for readability
imaging_paths_bleaching = []
for current_path in imaging_file_paths:
    imaging_paths_bleaching.append(current_path.split('/imaging')[0])
print("imaging_paths_bleaching" + repr(imaging_paths_bleaching))

##
# List of paths for moco
# Identical to imaging_paths_meanbrain but define explicitly for readability
list_of_imaging_paths_moco = []
for current_path in imaging_file_paths:
    list_of_imaging_paths_moco.append(current_path.split('/imaging')[0])

##
# List of paths for zscore
imaging_paths_zscore = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_zscore.append(current_path.split('/imaging')[0])
##
# list of paths for temporal highpass filter
# identical to zscore imaging paths but for ease of readibility, explicitly create a new one
imaging_paths_temp_HP_filter = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_temp_HP_filter.append(current_path.split('/imaging')[0])

##
# list of paths for correlation
# identical to zscore imaging paths but for ease of readibility, explicitly create a new one
imaging_paths_corr = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_corr.append(current_path.split('/imaging')[0])
# Behavior to be correlated with z scored brain activity
corr_behaviors = ['dRotLabZneg', 'dRotLabZpos', 'dRotLabY']

##
# List of paths for moco meanbrains
# Identical to imaging_paths_meanbrain but define explicitly for readability
imaging_paths_moco_meanbrain = []
for current_path in imaging_file_paths:
    imaging_paths_moco_meanbrain.append(current_path.split('/imaging')[0])
print("imaging_paths_moco_meanbrain" + repr(imaging_paths_moco_meanbrain))

##
# List of paths for clean anatomy - only anatomy folders!
imaging_paths_clean_anatomy = []
for current_path in imaging_file_paths:
    if 'anat' in current_path:
        imaging_paths_clean_anatomy.append(current_path.split('/imaging')[0])

##
# list of paths for supervoxel
# identical to zscore imaging paths but for ease of readibility, explicitly create a new one
atlas_path = pathlib.Path("brain_atlases/jfrc_atlas_from_brainsss.nii") #luke.nii"
imaging_paths_supervoxels = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_supervoxels.append(current_path.split('/imaging')[0])
func_channels=[]
if 'channel_1' in FUNCTIONAL_CHANNELS:
    func_channels.append('1')
if 'channel_2' in FUNCTIONAL_CHANNELS:
    func_channels.append('2')
if 'channel_3' in FUNCTIONAL_CHANNELS:
    func_channels.append('3')

####
# probably not relevant - I think this is what bifrost does (better)
##
# list of paths for func2anat
imaging_paths_func2anat = []
anat_path_func2anat = None
for current_path in imaging_file_paths:
    if 'func' in current_path:
        imaging_paths_func2anat.append(current_path.split('/imaging')[0])
    # the folder name of the anatomical channel
    elif 'anat' in current_path:
        if anat_path_func2anat is None:
            anat_path_func2anat = current_path.split('/imaging')[0]
        else:
            print('!!!! WARNING: More than one folder with "anat"-string in fly to analyze. ')
            print('!!!! func to anat function will likely give unexpected results! ')
# the anatomical channel for func2anat
if 'channel_1' in ANATOMY_CHANNEL:
    file_path_func2anat_fixed = ['channel_1']
elif 'channel_2' in ANATOMY_CHANNEL:
    file_path_func2anat_fixed = ['channel_2']
elif 'channel_3' in ANATOMY_CHANNEL:
    file_path_func2anat_fixed = ['channel_3']

##
# list of paths for anat2atlas

imaging_paths_anat2atlas =[]
for current_path in imaging_file_paths:
    if 'anat' in current_path:
        # here it's ok to have more than one anatomy folder! However, script will break before...
        # but at least this part doesn't have to break!
        imaging_paths_anat2atlas.append(current_path.split('/imaging')[0])

# the anatomical channel for func2anat
file_path_anat2atlas_moving = []
if 'channel_1' in ANATOMY_CHANNEL:
    file_path_anat2atlas_moving.append('channel_1')
elif 'channel_2' in ANATOMY_CHANNEL:
    file_path_anat2atlas_moving.append('channel_2')
elif 'channel_3' in ANATOMY_CHANNEL:
    file_path_anat2atlas_moving.append('channel_3')

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

#####
# Output data path
#####
# Output files for fictrac_qc rule
print("full_fictrac_file_oak_paths" + repr(full_fictrac_file_oak_paths))
fictrac_output_files_2d_hist_fixed = create_output_path_func(list_of_paths=full_fictrac_file_oak_paths,
                                                             filename='fictrac_2d_hist_fixed.png')

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
        expand(str(fly_folder_to_process_oak) + "/{fictrac_paths}/fictrac_2d_hist_fixed.png", fictrac_paths=FICTRAC_PATHS),
        # data in fly_dirs.json!
        ###
        # Bleaching QC
        ###,
        expand(str(fly_folder_to_process_oak) +"/{bleaching_imaging_paths}/imaging/bleaching.png", bleaching_imaging_paths=imaging_paths_bleaching),
        ###
        # Meanbrain
        ###
        expand(str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths}/imaging/channel_{meanbr_ch}_mean.nii", meanbr_imaging_paths=imaging_paths_meanbrain, meanbr_ch=channels),
        ###
        # Motion correction output
        ###
        # While we don't really need this file afterwards, it's a good idea to have it here because the empty h5 file
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
        expand(str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_1_moco_zscore.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[], zscore_imaging_paths=imaging_paths_zscore),
        expand(str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_2_moco_zscore.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[], zscore_imaging_paths=imaging_paths_zscore),
        expand(str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_3_moco_zscore.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[], zscore_imaging_paths=imaging_paths_zscore),
        ###
        # temporal high-pass filter
        ###
        expand(str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore_highpass.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[], temp_HP_filter_imaging_paths=imaging_paths_temp_HP_filter),
        expand(str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore_highpass.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[], temp_HP_filter_imaging_paths=imaging_paths_temp_HP_filter),
        expand(str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore_highpass.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[], temp_HP_filter_imaging_paths=imaging_paths_temp_HP_filter),
        ###
        # correlation with behavior
        ###
        expand(str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/corr/channel_1_corr_{corr_behavior}.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [], corr_imaging_paths=imaging_paths_corr, corr_behavior=corr_behaviors),
        expand(str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/corr/channel_2_corr_{corr_behavior}.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [], corr_imaging_paths=imaging_paths_corr, corr_behavior=corr_behaviors),
        expand(str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/corr/channel_3_corr_{corr_behavior}.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [], corr_imaging_paths=imaging_paths_corr, corr_behavior=corr_behaviors),
        ###
        # Meanbrain of moco brain
        ###
        expand(str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths}/moco/channel_{meanbr_moco_ch}_moco_mean.nii", moco_meanbr_imaging_paths=imaging_paths_moco_meanbrain, meanbr_moco_ch=channels),
        ###
        # Clean anatomy
        expand(str(fly_folder_to_process_oak) + "/{clean_anatomy_paths}/moco/channel_{clean_anat_ch}_moco_mean_clean.nii", clean_anatomy_paths=imaging_paths_clean_anatomy, clean_anat_ch=channels),
        ##
        # make supervoxels
        ###
        expand(str(fly_folder_to_process_oak) + "/{supervoxel_paths}/clustering/channel_{supervoxel_ch}_cluster_labels.npy", supervoxel_paths=imaging_paths_supervoxels, supervoxel_ch=func_channels),
        expand(str(fly_folder_to_process_oak) + "/{supervoxel_paths}/clustering/channel_{supervoxel_ch}_cluster_signals.npy",supervoxel_paths=imaging_paths_supervoxels, supervoxel_ch=func_channels),

        # Below might be Bifrost territory - ignore for now.
        ###
        # func2anat
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{func2anat_paths}/warp/{func2anat_moving}_func-to-{func2anat_fixed}_anat.nii",
            func2anat_paths=imaging_paths_func2anat,
            func2anat_moving=file_path_func2anat_fixed, # This is the channel which is designated as ANATOMY_CHANNEL
            func2anat_fixed=file_path_func2anat_fixed),
        ##
        # anat2atlas
        ##
        expand(str(fly_folder_to_process_oak)
            + "/{anat2atlas_paths}/warp/{anat2atlas_moving}_-to-atlas.nii",
            anat2atlas_paths=imaging_paths_anat2atlas,
            anat2atlas_moving=file_path_anat2atlas_moving),

rule fictrac_qc_rule:
    """
    Benchmark with full (30 min vol dataset)
    State: OUT_OF_MEMORY (exit code 0)
    Cores: 1
    CPU Utilized: 00:00:10
    CPU Efficiency: 30.30% of 00:00:33 core-walltime
    Job Wall-clock time: 00:00:33
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 1000.00 MB (1000.00 MB/node)
    add resources: mem_mb=snake_utils.mem_mb_times_threads
    """
    threads: 1
    resources: mem_mb=snake_utils.mem_mb_times_threads
    input:
        str(fly_folder_to_process_oak) + "/{fictrac_paths}/fictrac_behavior_data.dat"
    output:
        str(fly_folder_to_process_oak) + "/{fictrac_paths}/fictrac_2d_hist_fixed.png"
    run:
        try:
            preprocessing.fictrac_qc(fly_folder_to_process_oak,
                                    fictrac_file_path= full_fictrac_file_oak_paths,
                                    fictrac_fps=fictrac_fps # AUTOMATE THIS!!!! ELSE BUG PRONE!!!!
                                    )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_fictrac_qc_rule')
            utils.write_error(logfile=logfile,
                                 error_stack=error_stack,
                                 width=width)

rule bleaching_qc_rule:
    """
    Benchmarking:
    Cores per node: 2
    CPU Utilized: 00:00:05
    CPU Efficiency: 5.56% of 00:01:30 core-walltime
    Job Wall-clock time: 00:00:45
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 9.00 GB (9.00 GB/node)
    
    Cores per node: 6
    CPU Utilized: 00:00:08
    CPU Efficiency: 1.65% of 00:08:06 core-walltime
    Job Wall-clock time: 00:01:21
    Memory Utilized: 9.87 GB
    Memory Efficiency: 20.98% of 47.07 GB
    
    Cores per node: 6
    CPU Utilized: 00:00:09
    CPU Efficiency: 0.84% of 00:17:48 core-walltime
    Job Wall-clock time: 00:02:58
    Memory Utilized: 7.32 GB
    Memory Efficiency: 14.46% of 50.60 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:08
    CPU Efficiency: 3.60% of 00:03:42 core-walltime
    Job Wall-clock time: 00:01:51
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 9.00 GB (9.00 GB/node)
        
    Now it works but I need to optimize the input/output files - check the memory requirement!!! This is overkill!
    Cores per node: 14
    CPU Utilized: 00:01:38
    CPU Efficiency: 2.09% of 01:18:10 core-walltime
    Job Wall-clock time: 00:05:35
    Memory Utilized: 56.15 GB
    Memory Efficiency: 48.54% of 115.68 GB
    
    Benchmark with full (30min) dataset
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:21
    CPU Efficiency: 16.94% of 00:02:04 core-walltime
    Job Wall-clock time: 00:01:02
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 14.65 GB (14.65 GB/node)
    --> from mem_mb_times_threads to mem_mb_times_input
    
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
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input, # This is probably overkill todo decrease!
        runtime='10m' # In my test cases it was never more than 5 minutes!
    input:
        brains_paths_ch1=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_1.nii" if CH1_EXISTS else [],
        brains_paths_ch2=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_2.nii" if CH2_EXISTS else [],
        brains_paths_ch3=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_3.nii" if CH3_EXISTS else [],
    output:
        str(fly_folder_to_process_oak) +"/{bleaching_imaging_paths}/imaging/bleaching.png"
    run:
        try:
            preprocessing.bleaching_qc(fly_directory=fly_folder_to_process_oak,
                                        path_to_read=[input.brains_paths_ch1, input.brains_paths_ch2, input.brains_paths_ch3], #imaging_paths_by_folder_scratch, # {input} didn't work, I think because it destroyed the list of list we expect to see here #imaging_paths_by_folder_scratch,
                                        path_to_save=output, # can't use output, messes things up here! #imaging_paths_by_folder_oak
                                        #print_output = output
            )
            print('Done with bleaching_qc')
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_bleaching_qc_rule')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)
            print('Error with bleaching_qc' )

rule make_mean_brain_rule:
    """
    Changed memory usage by avoiding call to 'get_fdata()'
    With same 30min vol dataset I now get: 
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:00:11
    CPU Efficiency: 2.75% of 00:06:40 core-walltime
    Job Wall-clock time: 00:01:40
    Memory Utilized: 8.15 GB
    Memory Efficiency: 34.65% of 23.54 GB
    --> to 1.5 times input size memory
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:22
    CPU Efficiency: 4.89% of 00:07:30 core-walltime
    Job Wall-clock time: 00:03:45
    Memory Utilized: 9.24 GB
    Memory Efficiency: 60.85% of 15.18 GB


    ######
    Benchmark with full dataset (30min vol recording)
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:17
    CPU Efficiency: 16.04% of 00:01:46 core-walltime
    Job Wall-clock time: 00:00:53
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 14.65 GB (14.65 GB/node)
    --> set from mem_mb_times_threads to mem_mb_times_input

    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:00:11
    CPU Efficiency: 2.75% of 00:06:40 core-walltime
    Job Wall-clock time: 00:01:40
    Memory Utilized: 27.95 GB
    Memory Efficiency: 118.74% of 23.54 GB
    --> set from mem_mb_times_threads to mem_mb_more_times_input
    State: OUT_OF_MEMORY (exit code 0)

    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:00:15
    CPU Efficiency: 4.69% of 00:05:20 core-walltime
    Job Wall-clock time: 00:01:20
    Memory Utilized: 31.01 GB
    Memory Efficiency: 94.12% of 32.95 GB
    ??? The input file is 10Gb. We are not making any copies of the data. 

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
    threads: 2  # It seems to go a bit faster. Can probably set to 1 if want to save cores
    resources: mem_mb=snake_utils.mem_mb_less_times_input  #snake_utils.mem_mb_times_input #mem_mb=snake_utils.mem_mb_more_times_input
    input:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths}/imaging/channel_{meanbr_ch}.nii"
    output:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths}/imaging/channel_{meanbr_ch}_mean.nii"
    run:
        try:
            preprocessing.make_mean_brain(fly_directory=fly_folder_to_process_oak,
                meanbrain_n_frames=meanbrain_n_frames,
                path_to_read=input,
                path_to_save=output,
                rule_name='make_mean_brain_rule')
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_make_mean_brain')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule motion_correction_rule:
    """
    Yandan file anat file(25GB)
    Nodes: 1
    Cores per node: 18
    CPU Utilized: 2-13:59:13
    CPU Efficiency: 36.76% of 7-00:36:18 core-walltime
    Job Wall-clock time: 09:22:01
    Memory Utilized: 146.73 GB
    Memory Efficiency: 85.64% of 171.34 GB
    
    Yandan func filex
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 18:06:13
    CPU Efficiency: 52.91% of 1-10:13:06 core-walltime
    Job Wall-clock time: 05:42:11
    Memory Utilized: 4.21 GB
    Memory Efficiency: 5.94% of 70.93 GB
    
    
    Had another MMO with a very small file (200Mb)
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 00:03:38
    CPU Efficiency: 19.43% of 00:18:42 core-walltime
    Job Wall-clock time: 00:03:07
    Memory Utilized: 977.49 MB
    Memory Efficiency: 97.75% of 1000.00 MB
    -> Set minimal memory to 4GB? 
    
    Current setting 
    (threads: 6
     resources:
       mem_mb=snake_utils.mem_mb_more_times_input,
       runtime=snake_utils.time_for_moco_input # runtime takes input as seconds!)
    seems to work for large anatomical files (2x10Gb)
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 22:10:11
    CPU Efficiency: 61.96% of 1-11:46:48 core-walltime
    Job Wall-clock time: 05:57:48
    Memory Utilized: 60.57 GB
    Memory Efficiency: 83.56% of 72.49 GB
    
    Unoptimzed for small slices - this is a 10Gb functional recording:
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 1-05:28:29
    CPU Efficiency: 58.07% of 2-02:45:18 core-walltime
    Job Wall-clock time: 08:27:33
    Memory Utilized: 4.13 GB
    Memory Efficiency: 5.82% of 70.93 GB
    
    with: snake_utils.mem_mb_times_input
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 05:12:42
    CPU Efficiency: 59.22% of 08:48:00 core-walltime
    Job Wall-clock time: 01:28:00
    Memory Utilized: 8.20 GB
    Memory Efficiency: 90.45% of 9.06 GB
    
    Benchmarking: Two 3Gb files required 19Gb (43% of 44Gb at 6 cores). 1 hour
    Same 3 gb files another run:
        Cores per node: 6
        CPU Utilized: 02:56:44
        CPU Efficiency: 54.95% of 05:21:36 core-walltime
        Job Wall-clock time: 00:53:36
        Memory Utilized: 18.47 GB
        Memory Efficiency: 42.03% of 43.95 GB
    
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
    resources:
        mem_mb=snake_utils.mem_mb_much_more_times_input,
        runtime=snake_utils.time_for_moco_input # runtime takes input as seconds!
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

rule zscore_rule:
    """
    Benchmarking:
    Cores per node: 2
    CPU Utilized: 00:00:48
    CPU Efficiency: 5.77% of 00:13:52 core-walltime
    Job Wall-clock time: 00:06:56
    Memory Utilized: 2.91 GB
    Memory Efficiency: 31.98% of 9.09 GB
    
    Cores per node: 2
    CPU Utilized: 00:00:55
    CPU Efficiency: 3.43% of 00:26:44 core-walltime
    Job Wall-clock time: 00:13:22
    Memory Utilized: 3.60 GB
    Memory Efficiency: 39.62% of 9.09 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:48
    CPU Efficiency: 5.77% of 00:13:52 core-walltime
    Job Wall-clock time: 00:06:56
    Memory Utilized: 2.91 GB
    Memory Efficiency: 31.98% of 9.09 GB
    
    Cores per node: 7
    CPU Utilized: 00:02:06
    CPU Efficiency: 0.64% of 05:30:10 core-walltime
    Job Wall-clock time: 00:47:10
    Memory Utilized: 26.30 GB
    Memory Efficiency: 48.85% of 53.84 GB
    
    ########
    Benchmarking: Did 2.5*input file size and got a 86% efficiency on the memory, 4 threads only 12% efficiency
    Did same with 1 thread and seemed to be enough. Keep at 1 thread for now, might break with larger files.
    
    Same test file:
    Cores per node: 2
    CPU Utilized: 00:00:23
    CPU Efficiency: 9.66% of 00:03:58 core-walltime
    Job Wall-clock time: 00:01:59
    Memory Utilized: 1.62 GB
    Memory Efficiency: 17.80% of 9.09 GB
    
    And once more: 
    Cores per node: 2
    CPU Utilized: 00:00:54
    CPU Efficiency: 13.99% of 00:06:26 core-walltime
    Job Wall-clock time: 00:03:13
    Memory Utilized: 689.88 MB
    Memory Efficiency: 7.41% of 9.09 GB
    
    """
    threads: 1
    resources: mem_mb=snake_utils.mem_mb_much_more_times_input#mem_mb_times_input
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

rule temporal_high_pass_filter_rule:
    """
    Benchmark:
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:02:41
    CPU Efficiency: 39.46% of 00:06:48 core-walltime
    Job Wall-clock time: 00:03:24
    Memory Utilized: 7.80 GB
    Memory Efficiency: 61.86% of 12.60 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:02:42
    CPU Efficiency: 34.62% of 00:07:48 core-walltime
    Job Wall-clock time: 00:03:54
    Memory Utilized: 7.42 GB
    Memory Efficiency: 58.86% of 12.60 GB
    
    Cores per node: 10
    CPU Utilized: 00:26:42
    CPU Efficiency: 9.26% of 04:48:20 core-walltime
    Job Wall-clock time: 00:28:50
    Memory Utilized: 41.26 GB
    Memory Efficiency: 58.24% of 70.84 GB
    
    ##
    
    Benchmark with the func1 file (~3Gb)
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:05:33
    CPU Efficiency: 40.51% of 00:13:42 core-walltime
    Job Wall-clock time: 00:06:51
    Memory Utilized: 7.62 GB
    Memory Efficiency: 51.99% of 14.65 GB

    Again with 4 cores and 32Gb
    Cores per node: 4
    CPU Utilized: 00:03:09
    CPU Efficiency: 13.66% of 00:23:04 core-walltime
    Job Wall-clock time: 00:05:46
    Memory Utilized: 7.33 GB
    Memory Efficiency: 25.02% of 29.30 GB

    - Without chunking:
    CPU Utilized: 00:02:47
    CPU Efficiency: 17.54% of 00:15:52 core-walltime
    Job Wall-clock time: 00:03:58
    Memory Utilized: 7.76 GB
    Memory Efficiency: 26.47% of 29.30 GB

    - With only 3.5 times input file size as memory and 2 thread:
    Cores per node: 2
    CPU Utilized: 00:02:55
    CPU Efficiency: 34.31% of 00:08:30 core-walltime
    Job Wall-clock time: 00:04:15
    Memory Utilized: 7.50 GB
    Memory Efficiency: 59.48% of 12.60 GB
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_more_times_input
    input:
        zscore_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    output:
        temp_HP_filter_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore_highpass.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        temp_HP_filter_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore_highpass.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        temp_HP_filter_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore_highpass.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else [],

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

rule correlation_rule:
    """
    Benchmark Yandan's data
    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:01:42
    CPU Efficiency: 9.51% of 00:17:52 core-walltime
    Job Wall-clock time: 00:04:28
    Memory Utilized: 21.77 GB
    Memory Efficiency: 70.63% of 30.83 GB
    
    Benchmark:
    Cores: 1
    CPU Utilized: 00:00:16
    CPU Efficiency: 25.81% of 00:01:02 core-walltime
    Job Wall-clock time: 00:01:02
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 5.49 GB (5.49 GB/node)
    
    State: COMPLETED (exit code 0)
    Cores: 1
    CPU Utilized: 00:00:29
    CPU Efficiency: 34.12% of 00:01:25 core-walltime
    Job Wall-clock time: 00:01:25
    Memory Utilized: 643.89 MB
    Memory Efficiency: 11.45% of 5.49 GB
    
    Cores: 1
    CPU Utilized: 00:00:27
    CPU Efficiency: 26.47% of 00:01:42 core-walltime
    Job Wall-clock time: 00:01:42
    Memory Utilized: 2.06 GB
    Memory Efficiency: 37.62% of 5.48 GB
    
    Cores: 1
    CPU Utilized: 00:00:27
    CPU Efficiency: 26.47% of 00:01:42 core-walltime
    Job Wall-clock time: 00:01:42
    Memory Utilized: 2.07 GB
    Memory Efficiency: 37.84% of 5.48 GB
    
    Cores: 1
    CPU Utilized: 00:00:29
    CPU Efficiency: 28.43% of 00:01:42 core-walltime
    Job Wall-clock time: 00:01:42
    Memory Utilized: 1.67 GB
    Memory Efficiency: 30.40% of 5.49 GB
    
    Cores: 1
    CPU Utilized: 00:00:27
    CPU Efficiency: 26.47% of 00:01:42 core-walltime
    Job Wall-clock time: 00:01:42
    Memory Utilized: 2.08 GB
    Memory Efficiency: 37.91% of 5.48 GB
    
    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:01:06
    CPU Efficiency: 5.73% of 00:19:12 core-walltime
    Job Wall-clock time: 00:04:48
    Memory Utilized: 21.68 GB
    Memory Efficiency: 70.38% of 30.81 GB
    
    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:01:05
    CPU Efficiency: 5.64% of 00:19:12 core-walltime
    Job Wall-clock time: 00:04:48
    Memory Utilized: 21.68 GB
    Memory Efficiency: 70.38% of 30.81 GB
    
    Nodes: 1
    Cores per node: 4
    CPU Utilized: 00:01:10
    CPU Efficiency: 7.03% of 00:16:36 core-walltime
    Job Wall-clock time: 00:04:09
    Memory Utilized: 21.81 GB
    Memory Efficiency: 70.78% of 30.81 GB
    
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_input
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:01:02
    CPU Efficiency: 27.93% of 00:03:42 core-walltime
    Job Wall-clock time: 00:01:51
    Memory Utilized: 4.44 GB
    Memory Efficiency: 48.59% of 9.13 GB
    """
    threads: 1
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input,
        runtime=10 #snake_utils.time_for_correlation
    input:
        corr_path_ch1=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/channel_1_moco_zscore_highpass.h5" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        corr_path_ch2=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/channel_2_moco_zscore_highpass.h5" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        corr_path_ch3=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/channel_3_moco_zscore_highpass.h5" if 'channel_3' in FUNCTIONAL_CHANNELS else[],
        fictrac_path=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/fictrac/fictrac_behavior_data.dat",
        metadata_path=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/imaging/recording_metadata.xml"
    output:
        save_path_ch1=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/corr/channel_1_corr_{corr_behavior}.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        save_path_ch2=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/corr/channel_2_corr_{corr_behavior}.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        save_path_ch3=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/corr/channel_3_corr_{corr_behavior}.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else[],
    run:
        try:
            preprocessing.correlation(fly_directory=fly_folder_to_process_oak,
                                dataset_path=[input.corr_path_ch1, input.corr_path_ch2, input.corr_path_ch3],
                                save_path=[output.save_path_ch1, output.save_path_ch2, output.save_path_ch3],
                                #behavior=input.fictrac_path,
                                fictrac_fps=fictrac_fps,
                                metadata_path=input.metadata_path,
                                fictrac_path=input.fictrac_path)
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_correlation')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)
'''           
rule STA_rule:
    """TODO"""'''

rule moco_mean_brain_rule:
    """
    Similar to make mean brain but takes moco corrected brain! 
    Benchmark:
    Cores per node: 2
    CPU Utilized: 00:00:09
    CPU Efficiency: 1.29% of 00:11:38 core-walltime
    Job Wall-clock time: 00:05:49
    Memory Utilized: 4.00 GB
    Memory Efficiency: 43.99% of 9.09 GB
    
    Cores per node: 6 <- automatically increases if memory requirement is hig!
    CPU Utilized: 00:01:25
    CPU Efficiency: 6.62% of 00:21:24 core-walltime
    Job Wall-clock time: 00:03:34
    Memory Utilized: 13.38 GB
    Memory Efficiency: 25.48% of 52.51 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:09
    CPU Efficiency: 2.37% of 00:06:20 core-walltime
    Job Wall-clock time: 00:03:10
    Memory Utilized: 3.21 GB
    Memory Efficiency: 35.31% of 9.09 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:18
    CPU Efficiency: 4.05% of 00:07:24 core-walltime
    Job Wall-clock time: 00:03:42
    Memory Utilized: 3.90 GB
    Memory Efficiency: 42.88% of 9.09 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:21
    CPU Efficiency: 3.86% of 00:09:04 core-walltime
    Job Wall-clock time: 00:04:32
    Memory Utilized: 3.31 GB
    Memory Efficiency: 36.45% of 9.09 GB
    
    Cores per node: 6
    CPU Utilized: 00:00:39
    CPU Efficiency: 3.05% of 00:21:18 core-walltime
    Job Wall-clock time: 00:03:33
    Memory Utilized: 13.27 GB
    Memory Efficiency: 25.28% of 52.51 GB
    
    Cores per node: 6
    CPU Utilized: 00:01:25
    CPU Efficiency: 6.62% of 00:21:24 core-walltime
    Job Wall-clock time: 00:03:34
    Memory Utilized: 13.38 GB
    Memory Efficiency: 25.48% of 52.51 GB
    
    Cores per node: 6
    CPU Utilized: 00:01:21
    CPU Efficiency: 1.46% of 01:32:36 core-walltime
    Job Wall-clock time: 00:15:26
    Memory Utilized: 23.73 GB
    Memory Efficiency: 44.07% of 53.84 GB
    
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 00:00:30
    CPU Efficiency: 0.39% of 02:09:00 core-walltime
    Job Wall-clock time: 00:21:30
    Memory Utilized: 20.80 GB
    Memory Efficiency: 38.64% of 53.84 GB
    
    Nodes: 1
    Cores per node: 6
    CPU Utilized: 00:00:27
    CPU Efficiency: 2.24% of 00:20:06 core-walltime
    Job Wall-clock time: 00:03:21
    Memory Utilized: 16.89 GB
    Memory Efficiency: 37.79% of 44.70 GB
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_input
    input:
        str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths}/moco/channel_{meanbr_moco_ch}_moco.h5"
    output:
        str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths}/moco/channel_{meanbr_moco_ch}_moco_mean.nii"
    run:
        try:
            preprocessing.make_mean_brain(fly_directory=fly_folder_to_process_oak,
                                            meanbrain_n_frames=meanbrain_n_frames,
                                            path_to_read=input,
                                            path_to_save=output,
                                            rule_name='moco_mean_brain_rule',)
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_make_moco_mean_brain')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule clean_anatomy_rule:
    """
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:00:57
    CPU Efficiency: 36.54% of 00:02:36 core-walltime
    Job Wall-clock time: 00:01:18
    Memory Utilized: 2.89 GB
    Memory Efficiency: 29.62% of 9.77 GB
    
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:01:03
    CPU Efficiency: 36.63% of 00:02:52 core-walltime
    Job Wall-clock time: 00:01:26
    Memory Utilized: 3.78 GB
    Memory Efficiency: 38.66% of 9.77 GB
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_much_more_times_input # Todo, optimize memory usage of this function! #mem_mb_more_times_input #snake_utils.mem_mb_times_input # OOM!!!
    input: str(fly_folder_to_process_oak) + "/{clean_anatomy_paths}/moco/channel_{clean_anat_ch}_moco_mean.nii",
    output: str(fly_folder_to_process_oak) + "/{clean_anatomy_paths}/moco/channel_{clean_anat_ch}_moco_mean_clean.nii",
    run:
        try:
            preprocessing.clean_anatomy(fly_directory=fly_folder_to_process_oak,
                                        path_to_read=input,
                                        save_path=output)
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_clean_anatomy')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule make_supervoxels_rule:
    """
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:03:05
    CPU Efficiency: 43.84% of 00:07:02 core-walltime
    Job Wall-clock time: 00:03:31
    Memory Utilized: 4.66 GB
    Memory Efficiency: 51.74% of 9.00 GB
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_input
    input: str(fly_folder_to_process_oak) + "/{supervoxel_paths}/channel_{supervoxel_ch}_moco_zscore_highpass.h5"
    output:
        cluster_labels = str(fly_folder_to_process_oak) + "/{supervoxel_paths}/clustering/channel_{supervoxel_ch}_cluster_labels.npy",
        cluster_signals = str(fly_folder_to_process_oak) + "/{supervoxel_paths}/clustering/channel_{supervoxel_ch}_cluster_signals.npy"
    run:
        try:
            preprocessing.make_supervoxels(fly_directory=fly_folder_to_process_oak,
                                            path_to_read=input,
                                            save_path_cluster_labels=[output.cluster_labels],
                                            save_path_cluster_signals=[output.cluster_signals],
                                            n_clusters = 2000) # for sklearn.cluster.AgglomerativeClustering
                                            # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_make_supervoxels')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

# Probably Bifrost does it better.
rule func_to_anat_rule:
    """
    Nodes: 1
    Cores per node: 2
    CPU Utilized: 00:01:21
    CPU Efficiency: 38.21% of 00:03:32 core-walltime
    Job Wall-clock time: 00:01:46
    Memory Utilized: 1.09 GB
    Memory Efficiency: 27.97% of 3.91 GB

    Cores per node: 2
    CPU Utilized: 00:01:15
    CPU Efficiency: 35.71% of 00:03:30 core-walltime
    Job Wall-clock time: 00:01:45
    Memory Utilized: 1.09 GB
    Memory Efficiency: 27.86% of 3.91 GB

    Cores per node: 2
    CPU Utilized: 00:01:19
    CPU Efficiency: 34.96% of 00:03:46 core-walltime
    Job Wall-clock time: 00:01:53
    Memory Utilized: 1.12 GB
    Memory Efficiency: 28.78% of 3.91 GB

    Cores per node: 2
    CPU Utilized: 00:01:14
    CPU Efficiency: 46.25% of 00:02:40 core-walltime
    Job Wall-clock time: 00:01:20
    Memory Utilized: 0.00 MB (estimated maximum)
    Memory Efficiency: 0.00% of 3.91 GB (3.91 GB/node)
    """
    threads: 1
    resources:
        mem_mb=snake_utils.mem_mb_more_times_input,
        runtime='10m' # should be enough, is super quick already
    input:
        path_to_read_fixed=str(fly_folder_to_process_oak) + "/" + str(anat_path_func2anat) + '/moco/{func2anat_fixed}_moco_mean.nii',
        path_to_read_moving=str(fly_folder_to_process_oak) + "/{func2anat_paths}/moco/{func2anat_moving}_moco_mean.nii"
    output: str(fly_folder_to_process_oak) + "/{func2anat_paths}/warp/{func2anat_moving}_func-to-{func2anat_fixed}_anat.nii"

    run:
        try:
            preprocessing.align_anat(fly_directory=fly_folder_to_process_oak,
                path_to_read_fixed=[input.path_to_read_fixed],
                path_to_read_moving=[input.path_to_read_moving],
                path_to_save=output,
                type_of_transform='Affine',# copy-paste from brainsss
                resolution_of_fixed=(
                0.653, 0.653, 1),# Copy-paste from brainsss, probably can be read from metadate.xml!
                resolution_of_moving=(
                2.611, 2.611, 5),# Copy-paste from brainsss, probably can be read from metadate.xml!
                rule_name='func_to_anat',
                fixed_fly='anat', # this is important for where transform params are saved
                moving_fly='func', # this is important for where transform params are saved # TODO change this to
                # something dynamic, otherwise more than 1 channel won't work as expected!!!
                iso_2um_fixed=True,
                iso_2um_moving=False,
                grad_step=0.2,
                flow_sigma=3,
                total_sigma=0,
                syn_sampling=32
            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_func2anat')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)

rule anat_to_atlas:
    """
    Benchmark - Yandan data
    Cores per node: 2 # one should be enough
    CPU Utilized: 00:01:58
    CPU Efficiency: 62.11% of 00:03:10 core-walltime
    Job Wall-clock time: 00:01:35
    Memory Utilized: 1.21 GB
    Memory Efficiency: 30.91% of 3.91 GB

    """
    threads: 1
    resources: mem_mb=snake_utils.mem_mb_more_times_input
    input:
        path_to_read_fixed=atlas_path,
        path_to_read_moving=str(fly_folder_to_process_oak)
                            + "/{anat2atlas_paths}/moco/{anat2atlas_moving}_moco_mean_clean.nii"
    output:
        str(fly_folder_to_process_oak)
        + "/{anat2atlas_paths}/warp/{anat2atlas_moving}_-to-atlas.nii"

    run:
        try:
            preprocessing.align_anat(fly_directory=fly_folder_to_process_oak,
                path_to_read_fixed=[input.path_to_read_fixed],
                path_to_read_moving=[input.path_to_read_moving],
                path_to_save=output,
                type_of_transform='SyN',# copy-paste from brainsss
                resolution_of_fixed=(2, 2, 2),# Copy-paste from brainsss, probably can be read from metadate.xml!
                resolution_of_moving=(
                0.653, 0.653, 1),# Copy-paste from brainsss, probably can be read from metadate.xml!
                rule_name='anat_to_atlas',
                fixed_fly='meanbrain',# this is important for where transform params are saved
                moving_fly='anat',# this is important for where transform params are saved # TODO change this to
                # something dynamic, otherwise more than 1 channel won't work as expected!!!
                iso_2um_fixed=False,
                iso_2um_moving=True,
                grad_step=0.2,
                flow_sigma=3,
                total_sigma=0,
                syn_sampling=32
            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_anat2atlas')
            utils.write_error(logfile=logfile,
                error_stack=error_stack,
                width=width)
                

rule apply_transforms_rule:
    """
    """
    threads: 2
    resources: mem_mb=snake_utils.mem_mb_times_input
    input:
        path_to_read_fixed=atlas_path,
        path_to_read_moving='foo',
        path_to_syn_linear='foo.mat',
        path_to_syn_nonlinear='foo.nii.gz'
    output:
        path_to_save_result='warp/moving_fly-applied-fixed_fly.nii',
        path_to_save_mat='warp/bar.mat',
        path_to_save_nii_gz='warp/bar.nii.gz'
    run:
        try:
            preprocessing.apply_transforms(fly_directory=fly_folder_to_process_oak,
                                            path_to_read_fixed=[input.path_to_read_fixed],
                                            path_to_read_moving=[input.path_to_read_moving],
                                            path_to_syn_linear=[input.path_to_syn_linear],
                                            path_to_syn_nonlinear=[input.path_to_syn_nonlinear],
                                            path_to_save=output,
                                            resolution_of_fixed=(2,2,2), # copy-paste from brainsss
                                            resolution_of_moving=(2.611, 2.611, 5), # copy-paste from brainsss
                                            final_2um_iso=False, # copy-paste from brainsss
                                            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_apply_transforms')
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
