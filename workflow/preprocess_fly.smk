"""
Here the idea is to do preprocessing a little bit differently:

We assume that the fly_builder already ran and that the fly_dir exists.

Sherlock webpage writes:
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
import natsort

# On sherlock using config file type:
# ml python/3.9.0
# source .env_snakemake/bin/activate
# cd snake_brainsss/workflow
# snakemake --config user=dtadres --profile profiles/simple_slurm -s preprocess_fly.smk --directory /oak/stanford/groups/trc/data/David/Bruker/preprocessed/FS144_x_FS69/fly_008
# The 'user' points to the user file in 'users' folder. **Change to your name! **
# The profile ot the config.yaml in the profiles/simple_slurm folder. Doesn't need to be changed
# The -s indicates the script to be run (this one, so 'preprocess_fly.smk')
# --directory is the fly you want to point to!
########################################################

#>>>>
fictrac_fps = 100 # AUTOMATE THIS!!!! ELSE FOR SURE A MISTAKE WILL HAPPEN IN THE FUTURE!!!!
# TODO!!!! Instead of just believing a framerate, use the voltage signal recorded during imaging
# that defines the position of a given frame!
#<<<<

# First n frames to average over when computing mean/fixed brain | Default None
# (average over all frames).
meanbrain_n_frames =  None

##########################################################
import pathlib
import json
import datetime
# path of workflow i.e. /Users/dtadres/snake_brainsss/workflow
#scripts_path = pathlib.Path(__file__).resolve()
scripts_path = workflow.basedir # Exposes path to this file
from brainsss import utils
from scripts import preprocessing
from scripts import snake_utils
import os
import sys
print(os.getcwd())

#### KEEP for future
# SCRATCH_DIR
#SCRATCH_DIR = '/scratch/users/' + current_user
#print(SCRATCH_DIR)
####

current_user = config['user'] # this is whatever is entered when calling snakemake, i.e.
# snakemake --profile profiles/simple_slurm -s snaketest.smk --config user=jcsimon would
# yield 'jcsimon' here
settings = utils.load_user_settings(current_user)
dataset_path = pathlib.Path(settings['dataset_path'])

# On sherlock this is usually python3 but on a personal computer can be python
shell_python_command = str(settings.get('shell_python_command', "python3"))
print("shell_python_command" + shell_python_command)
moco_temp_folder = str(settings.get('moco_temp_folder', "/scratch/groups/trc"))

# Define path to imports to find fly.json!
#fly_folder_to_process_oak = pathlib.Path(dataset_path,fly_folder_to_process)
fly_folder_to_process_oak = pathlib.Path(os.getcwd())
print('Analyze data in ' + repr(fly_folder_to_process_oak.as_posix()))

# Read channel information from fly.json file
# If fails here, means the folder specified doesn't exist. Check name.
# Note: Good place to let user know to check folder and exit!
with open(pathlib.Path(fly_folder_to_process_oak, 'fly.json'), 'r') as file:
    fly_json = json.load(file)
# This needs to come from some sort of json file the experimenter
# creates while running the experiment. Same as genotype.
FUNCTIONAL_CHANNELS = fly_json['functional_channel']
# It is probably necessary to forcibly define STRUCTURAL_CHANNEL if not defined
# Would be better to have an error to be explicit!

# Throw an error if missing! User must provide this!
STRUCTURAL_CHANNEL = fly_json['structural_channel']
if STRUCTURAL_CHANNEL != 'channel_1' and \
    STRUCTURAL_CHANNEL != 'channel_2' and \
        STRUCTURAL_CHANNEL != 'channel_3':
    print('!!! ERROR !!!')
    print('You must provide "structural_channel" in the "fly.json" file for snake_brainsss to run!')
    sys.exit()
    # This would be a implicit fix. Not great as it'll
    # hide potential bugs. Better explicit
    #STRUCTURAL_CHANNEL = FUNCTIONAL_CHANNELS[0]

def ch_exists_func(channel):
    """
    Check if a given channel exists in global variables STRUCTURAL_CHANNEL and FUNCTIONAL_CHANNELS
    :param channel:
    :return:
    """
    if 'channel_' + str(channel) in FUNCTIONAL_CHANNELS:
        ch_exists = True
    else:
        ch_exists = False
    return(ch_exists)

def ch_exists_struct(channel):
    """
    Check if a given channel exists in global variables STRUCTURAL_CHANNEL and FUNCTIONAL_CHANNELS
    :param channel:
    :return:
    """
    if 'channel_' + str(channel) in STRUCTURAL_CHANNEL:
        ch_exists = True
    else:
        ch_exists = False
    return(ch_exists)

# Bool for which channel exists in this particular recording.
# IMPORTANT: One FLY must have the same channels per recording. This
# makes sense: If we have e.g. GCaMP and tdTomato we would always
# record from both the green and red channel, right?
CH1_EXISTS_FUNC = ch_exists_func("1")
CH2_EXISTS_FUNC = ch_exists_func("2")
CH3_EXISTS_FUNC = ch_exists_func("3")

CH1_EXISTS_STRUCT = ch_exists_struct("1")
CH2_EXISTS_STRUCT = ch_exists_struct("2")
CH3_EXISTS_STRUCT = ch_exists_struct("3")

####
# Load fly_dir.json
####
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
        # With automatic stimpack transfer it'll return "/func0/stimpack/loco/fictrac_behavior_data.dat"

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

FICTRAC_PATHS = []
for current_path in fictrac_file_paths:
    FICTRAC_PATHS.append(current_path.split('/fictrac_behavior_data.dat')[0])
# Fictrac data can be in different folders! For correlation, need to know the
# relative path following 'funcX'.
# IN ONE EXPERIMENT I ASSUME THAT THE FICTRAC STRUCTURE IS CONSISTENT!
fictrac_rel_path_correlation = None
current_fictrac_rel_path = FICTRAC_PATHS[0]
# Remove the first folder which is going to be likely 'func0'
rel_path_parts = pathlib.Path(current_fictrac_rel_path).parts[1::]
# Then put the parts back together
fictrac_rel_path_correlation = pathlib.Path(*rel_path_parts)

# For wildcards we need lists of elements of the path for each folder.
list_of_paths = []
for current_path in imaging_file_paths:
    list_of_paths.append(current_path.split('/imaging')[0])
# This is a list of all imaging paths so something like this
# ['anat0', 'func0', 'func1']
print('list_of_paths ' +repr(list_of_paths) )


list_of_paths_func = []
for current_path in imaging_file_paths:
    if 'func' in current_path:
        list_of_paths_func.append(current_path.split('/imaging')[0])

print("list_of_paths_func" + repr(list_of_paths_func))


list_of_paths_struct = []
for current_path in imaging_file_paths:
    if 'anat' in current_path:
        list_of_paths_struct.append(current_path.split('/imaging')[0])
if len(list_of_paths_struct) > 1:
    print('!!!WARNING!!!')
    print('The following folders have the "anat" keyword:')
    print(list_of_paths_struct)
    print('The folder ' + repr(natsort.natsorted(list_of_paths_struct[0])) + ' will be treated as the "main" anat folder.')
    list_of_paths_struct = natsort.natsorted(list_of_paths_struct[0])
print('list_of_paths_struct' + repr(list_of_paths_struct))

list_of_channels_func = []
if CH1_EXISTS_FUNC:
    list_of_channels_func.append("1")
if CH2_EXISTS_FUNC:
    list_of_channels_func.append("2")
if CH3_EXISTS_FUNC:
    list_of_channels_func.append("3")

print("list_of_channels_func" + repr(list_of_channels_func))

list_of_channels_struct = []
if CH1_EXISTS_STRUCT:
    list_of_channels_struct.append("1")
if CH2_EXISTS_STRUCT:
    list_of_channels_struct.append("2")
if CH3_EXISTS_STRUCT:
    list_of_channels_struct.append("3")

print("list_of_channels_struct" + repr(list_of_channels_struct))

# Behaviors to correlate with neural activity
corr_behaviors = ['dRotLabZneg', 'dRotLabZpos', 'dRotLabY']
# This would be a list like this ['1', '2']

atlas_path = pathlib.Path("brain_atlases/jfrc_atlas_from_brainsss.nii") #luke.nii"

func_channels=[]
if 'channel_1' in FUNCTIONAL_CHANNELS:
    func_channels.append('1')
if 'channel_2' in FUNCTIONAL_CHANNELS:
    func_channels.append('2')
if 'channel_3' in FUNCTIONAL_CHANNELS:
    func_channels.append('3')

struct_channel=[]
if 'channel_1' in STRUCTURAL_CHANNEL:
    struct_channel.append('channel_1')
elif 'channel_2' in STRUCTURAL_CHANNEL:
    struct_channel.append('channel_2')
elif 'channel_3' in STRUCTURAL_CHANNEL:
    struct_channel.append('channel_3')
if len(struct_channel)>1:
    print('!!!!WARNING!!!')
    print('The following channels are defined as anatomy channels: ')
    print(struct_channel)
    print('There should only be a single anatomy channel for the pipeline to work as expected.')

####
# probably not relevant - I think this is what bifrost does (better)
##
# list of paths for func2anat
#imaging_paths_func2anat = []
#anat_path_func2anat = None
#for current_path in imaging_file_paths:
    #if 'func' in current_path:
    #    imaging_paths_func2anat.append(current_path.split('/imaging')[0])
    # the folder name of the anatomical channel
    #elif 'anat' in current_path:
    #    if anat_path_func2anat is None:
    #        anat_path_func2anat = current_path.split('/imaging')[0]
    #    else:
    #        print('!!!! WARNING: More than one folder with "anat"-string in fly to analyze. ')
    #        print('!!!! func to anat function will likely give unexpected results! ')
# the anatomical channel for func2anat
#if 'channel_1' in ANATOMY_CHANNEL:
#    file_path_func2anat_fixed = ['channel_1']
#elif 'channel_2' in ANATOMY_CHANNEL:
#    file_path_func2anat_fixed = ['channel_2']
#elif 'channel_3' in ANATOMY_CHANNEL:
#    file_path_func2anat_fixed = ['channel_3']

##
# list of paths for anat2atlas

#imaging_paths_anat2atlas =[]
#for current_path in imaging_file_paths:
#    if 'anat' in current_path:
#        # here it's ok to have more than one anatomy folder! However, script will break before...
#        # but at least this part doesn't have to break!
#        imaging_paths_anat2atlas.append(current_path.split('/imaging')[0])

# the anatomical channel for func2anat
#file_path_anat2atlas_moving = []
#if 'channel_1' in ANATOMY_CHANNEL:
#    file_path_anat2atlas_moving.append('channel_1')
#elif 'channel_2' in ANATOMY_CHANNEL:
#    file_path_anat2atlas_moving.append('channel_2')
#elif 'channel_3' in ANATOMY_CHANNEL:
#    file_path_anat2atlas_moving.append('channel_3')

"""


"""

"""

        # Below might be Bifrost territory - ignore for now.
        ###
        # func2anat
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{func2anat_paths}/warp/{func2anat_moving}_func-to-{func2anat_fixed}_anat.nii",
               func2anat_paths=list_of_paths_func,
               func2anat_moving=struct_channel,  # This is the channel which is designated as STRUCTURAL_CHANNEL
               func2anat_fixed=struct_channel),

        ##
        # anat2atlas
        ##
        expand(str(fly_folder_to_process_oak)
               + "/{anat2atlas_paths}/warp/{anat2atlas_moving}_-to-atlas.nii",
               anat2atlas_paths=list_of_paths_anat,
               anat2atlas_moving=struct_channel),
"""


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
        # Bleaching QC
        # Since func and struct can have different channels, seperate the two
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{bleaching_imaging_paths}/imaging/bleaching_func.png",
            bleaching_imaging_paths=list_of_paths_func),
        ##
        expand(str(fly_folder_to_process_oak)
               + "/{bleaching_imaging_paths}/imaging/bleaching_struct.png",
            bleaching_imaging_paths=list_of_paths_struct),

        ###
        # Fictrac QC
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{fictrac_paths}/fictrac_2d_hist_fixed.png",
            fictrac_paths=FICTRAC_PATHS),
        # data in fly_dirs.json!

        ###
        # Meanbrain
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch_func}_mean_func.nii",
            meanbr_imaging_paths_func=list_of_paths_func,
            meanbr_ch_func=list_of_channels_func),
        ##
        expand(str(fly_folder_to_process_oak)
               + "/{meanbr_imaging_paths_struct}/imaging/channel_{meanbr_ch_struct}_mean_struct.nii",
            meanbr_imaging_paths_struct=list_of_paths_struct,
            meanbr_ch_struct=list_of_channels_struct),

        ###
        # Motion correction output FUNC
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/motcorr_params_func.npy",
               moco_imaging_paths_func=list_of_paths_func),

        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/channel_1_moco_func.nii" if CH1_EXISTS_FUNC else [],
            moco_imaging_paths_func=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/channel_2_moco_func.nii" if CH2_EXISTS_FUNC else [],
            moco_imaging_paths_func=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_func}/moco/channel_3_moco_func.nii" if CH3_EXISTS_FUNC else [],
            moco_imaging_paths_func=list_of_paths_func),

        ###
        # Motion correction output STRUCT
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_struct}/moco/motcorr_params_struct.npy",
               moco_imaging_paths_struct=list_of_paths_struct),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_struct}/moco/channel_1_moco_struct.nii" if CH1_EXISTS_STRUCT else [],
               moco_imaging_paths_struct=list_of_paths_struct),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_struct}/moco/channel_2_moco_struct.nii" if CH2_EXISTS_STRUCT else [],
               moco_imaging_paths_struct=list_of_paths_struct),
        expand(str(fly_folder_to_process_oak)
               + "/{moco_imaging_paths_struct}/moco/channel_3_moco_struct.nii" if CH3_EXISTS_STRUCT else [],
               moco_imaging_paths_struct=list_of_paths_struct),

        ###
        # Meanbrain of moco brain
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{moco_meanbr_imaging_paths_func}/moco/channel_{meanbr_moco_ch_func}_moco_mean_func.nii",
            moco_meanbr_imaging_paths_func=list_of_paths_func,
            meanbr_moco_ch_func=list_of_channels_func),
        #
        expand(str(fly_folder_to_process_oak)
               + "/{moco_meanbr_imaging_paths_struct}/moco/channel_{meanbr_moco_ch_struct}_moco_mean_struct.nii",
            moco_meanbr_imaging_paths_struct=list_of_paths_struct,
            meanbr_moco_ch_struct=list_of_channels_struct),

        ####
        # Z-score
        ####
        expand(str(fly_folder_to_process_oak)
               + "/{zscore_imaging_paths}/channel_1_moco_zscore.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
               zscore_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{zscore_imaging_paths}/channel_2_moco_zscore.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
               zscore_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{zscore_imaging_paths}/channel_3_moco_zscore.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
               zscore_imaging_paths=list_of_paths_func),

        ###
        # temporal high-pass filter
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
               temp_HP_filter_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
               temp_HP_filter_imaging_paths=list_of_paths_func),
        expand(str(fly_folder_to_process_oak)
               + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
               temp_HP_filter_imaging_paths=list_of_paths_func),

        ###
        # correlation with fictrac behavior
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{corr_imaging_paths}/corr/channel_1_corr_{corr_behavior}.nii" if 'channel_1' in FUNCTIONAL_CHANNELS and len(FICTRAC_PATHS) > 0 else [],
               corr_imaging_paths=list_of_paths_func, corr_behavior=corr_behaviors),
        expand(str(fly_folder_to_process_oak)
               + "/{corr_imaging_paths}/corr/channel_2_corr_{corr_behavior}.nii" if 'channel_2' in FUNCTIONAL_CHANNELS and len(FICTRAC_PATHS) > 0 else [],
               corr_imaging_paths=list_of_paths_func, corr_behavior=corr_behaviors),
        expand(str(fly_folder_to_process_oak)
               + "/{corr_imaging_paths}/corr/channel_3_corr_{corr_behavior}.nii" if 'channel_3' in FUNCTIONAL_CHANNELS and len(FICTRAC_PATHS) > 0 else [],
               corr_imaging_paths=list_of_paths_func, corr_behavior=corr_behaviors),

        ###
        # Clean anatomy
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{clean_anatomy_paths}/moco/channel_{clean_anat_ch}_moco_mean_clean.nii",
               clean_anatomy_paths=list_of_paths_struct,
               clean_anat_ch=list_of_channels_struct),

        ##
        # make supervoxels
        ###
        expand(str(fly_folder_to_process_oak)
               + "/{supervoxel_paths}/clustering/channel_{supervoxel_ch}_cluster_labels.npy",
               supervoxel_paths=list_of_paths_func,
               supervoxel_ch=func_channels),
        expand(str(fly_folder_to_process_oak)
               + "/{supervoxel_paths}/clustering/channel_{supervoxel_ch}_cluster_signals.npy",
               supervoxel_paths=list_of_paths_func,
               supervoxel_ch=func_channels),

rule fictrac_qc_rule:
    """
    """
    threads: snake_utils.threads_per_memory
    resources:
        mem_mb=snake_utils.mem_mb_times_threads,
        runtime='10m'
    input:
        str(fly_folder_to_process_oak) + "/{fictrac_paths}/fictrac_behavior_data.dat"
    output:
        str(fly_folder_to_process_oak) + "/{fictrac_paths}/fictrac_2d_hist_fixed.png"
    run:
        try:
            preprocessing.fictrac_qc(fly_folder_to_process_oak,
                                    fictrac_file_path= input,
                                    fictrac_fps=fictrac_fps # AUTOMATE THIS!!!! ELSE BUG PRONE!!!!
                                    )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_fictrac_qc_rule')
            utils.write_error(logfile=logfile,
                                 error_stack=error_stack)

rule bleaching_qc_rule_func:
    """
    """
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input, # This is probably overkill todo decrease!
        runtime='10m' # In my test cases it was never more than 5 minutes!
    input:
        brains_paths_ch1=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_1.nii" if CH1_EXISTS_FUNC else [],
        brains_paths_ch2=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_2.nii" if CH2_EXISTS_FUNC else [],
        brains_paths_ch3=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_3.nii" if CH3_EXISTS_FUNC else [],
    output:
        str(fly_folder_to_process_oak) +"/{bleaching_imaging_paths}/imaging/bleaching_func.png"
    run:
        try:
            preprocessing.bleaching_qc(fly_directory=fly_folder_to_process_oak,
                                        path_to_read=[input.brains_paths_ch1, input.brains_paths_ch2, input.brains_paths_ch3], #imaging_paths_by_folder_scratch, # {input} didn't work, I think because it destroyed the list of list we expect to see here #imaging_paths_by_folder_scratch,
                                        path_to_save=output, # can't use output, messes things up here! #imaging_paths_by_folder_oak
                                        #print_output = output
            )
            print('Done with bleaching_qc_func')
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_bleaching_qc_rule')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)
            print('Error with bleaching_qc' )

rule bleaching_qc_rule_struct:
    """
    """
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input, # This is probably overkill todo decrease!
        runtime='10m' # In my test cases it was never more than 5 minutes!
    input:
        brains_paths_ch1=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_1.nii" if CH1_EXISTS_STRUCT else [],
        brains_paths_ch2=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_2.nii" if CH2_EXISTS_STRUCT else [],
        brains_paths_ch3=str(fly_folder_to_process_oak) + "/{bleaching_imaging_paths}/imaging/channel_3.nii" if CH3_EXISTS_STRUCT else [],
    output:
        str(fly_folder_to_process_oak) +"/{bleaching_imaging_paths}/imaging/bleaching_struct.png"
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
                error_stack=error_stack)
            print('Error with bleaching_qc' )

rule make_mean_brain_rule_func:
    """
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
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input,  #snake_utils.mem_mb_times_input #mem_mb=snake_utils.mem_mb_more_times_input
        runtime='10m' # should be enough
    input:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch_func}.nii"
    output:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_func}/imaging/channel_{meanbr_ch_func}_mean_func.nii"
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
                error_stack=error_stack)

rule make_mean_brain_rule_struct:
    """
    """
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input,  #snake_utils.mem_mb_times_input #mem_mb=snake_utils.mem_mb_more_times_input
        runtime='10m' # should be enough
    input:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_struct}/imaging/channel_{meanbr_ch_struct}.nii"
    output:
            str(fly_folder_to_process_oak) + "/{meanbr_imaging_paths_struct}/imaging/channel_{meanbr_ch_struct}_mean_struct.nii"
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
                error_stack=error_stack)
rule motion_correction_parallel_processing_rule_func:
    """
    OOM using an anat folder! :
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 32
    CPU Utilized: 1-08:48:23
    CPU Efficiency: 60.01% of 2-06:40:00 core-walltime
    Job Wall-clock time: 01:42:30
    Memory Utilized: 117.43 GB
    Memory Efficiency: 99.69% of 117.79 GB
    And another OOM for an anat 
    State: OUT_OF_MEMORY (exit code 0)
    Nodes: 1
    Cores per node: 32
    CPU Utilized: 1-11:00:39
    CPU Efficiency: 62.09% of 2-08:23:28 core-walltime
    Job Wall-clock time: 01:45:44
    Memory Utilized: 115.60 GB
    Memory Efficiency: 98.14% of 117.79 GB
    
    Same settings, func super happy
    Nodes: 1
    Cores per node: 32
    CPU Utilized: 16:06:07
    CPU Efficiency: 67.37% of 23:54:08 core-walltime
    Job Wall-clock time: 00:44:49
    Memory Utilized: 24.06 GB
    Memory Efficiency: 32.56% of 73.90 GB
    
    -> There's clearly a huge discrepancy of memory used based on the resolution of the image.
    Ideally I could set the memory depending on the resolution of the image instead of the size of the file...
    
    To speed motion correction up, use the multiprocessing module. This requires the target to 
    be a module (not just a function). Hence we have a 'shell' directive here.
    """
    threads: 32 # the max that we can do - check with sh_part
    resources:
        mem_mb=snake_utils.mb_for_moco_input, #.mem_mb_much_more_times_input,
        runtime=snake_utils.time_for_moco_input # runtime takes input as seconds!
    input:
        # Only use the Channels that exists - this organizes the anatomy and functional paths inside the motion correction
        # module.
        brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_1.nii" if CH1_EXISTS_FUNC else [],
        brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_2.nii" if CH2_EXISTS_FUNC else [],
        brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_3.nii" if CH3_EXISTS_FUNC else [],

        mean_brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_1_mean_func.nii" if CH1_EXISTS_FUNC else [],
        mean_brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_2_mean_func.nii" if CH2_EXISTS_FUNC else [],
        mean_brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/imaging/channel_3_mean_func.nii" if CH3_EXISTS_FUNC else []
    output:
        moco_path_ch1 = str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/channel_1_moco_func.nii" if CH1_EXISTS_FUNC else[],
        moco_path_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/channel_2_moco_func.nii" if CH2_EXISTS_FUNC else [],
        moco_path_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/channel_3_moco_func.nii" if CH3_EXISTS_FUNC else [],
        par_output=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_func}/moco/motcorr_params_func.npy"

    shell: shell_python_command + " " + scripts_path + "/scripts/moco_parallel.py "
        "--fly_directory {fly_folder_to_process_oak} "
        "--dataset_path {dataset_path} "
        "--brain_paths_ch1 {input.brain_paths_ch1} "
        "--brain_paths_ch2 {input.brain_paths_ch2} "
        "--brain_paths_ch3 {input.brain_paths_ch3} "
        "--mean_brain_paths_ch1 {input.mean_brain_paths_ch1} "
        "--mean_brain_paths_ch2 {input.mean_brain_paths_ch2} "
        "--mean_brain_paths_ch3 {input.mean_brain_paths_ch3} "
        "--STRUCTURAL_CHANNEL {STRUCTURAL_CHANNEL} "
        "--FUNCTIONAL_CHANNELS {FUNCTIONAL_CHANNELS} "
        "--moco_path_ch1 {output.moco_path_ch1} "
        "--moco_path_ch2 {output.moco_path_ch2} "
        "--moco_path_ch3 {output.moco_path_ch3} "
        "--par_output {output.par_output} "
        "--moco_temp_folder {moco_temp_folder} "

rule motion_correction_parallel_processing_rule_struct:
    """
    """
    threads: 32  # the max that we can do - check with sh_part
    resources:
        mem_mb=snake_utils.mb_for_moco_input,#.mem_mb_much_more_times_input,
        runtime=snake_utils.time_for_moco_input  # runtime takes input as seconds!
    input:
        # Only use the Channels that exists - this organizes the anatomy and functional paths inside the motion correction
        # module.
        brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/imaging/channel_1.nii" if CH1_EXISTS_STRUCT else [],
        brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/imaging/channel_2.nii" if CH2_EXISTS_STRUCT else [],
        brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/imaging/channel_3.nii" if CH3_EXISTS_STRUCT else [],

        mean_brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/imaging/channel_1_mean_struct.nii" if CH1_EXISTS_STRUCT else [],
        mean_brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/imaging/channel_2_mean_struct.nii" if CH2_EXISTS_STRUCT else [],
        mean_brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/imaging/channel_3_mean_struct.nii" if CH3_EXISTS_STRUCT else []
    output:
        moco_path_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/moco/channel_1_moco_struct.nii" if CH1_EXISTS_STRUCT else [],
        moco_path_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/moco/channel_2_moco_struct.nii" if CH2_EXISTS_STRUCT else [],
        moco_path_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/moco/channel_3_moco_struct.nii" if CH3_EXISTS_STRUCT else [],
        par_output=str(fly_folder_to_process_oak) + "/{moco_imaging_paths_struct}/moco/motcorr_params_struct.npy"

    shell: shell_python_command + " " + scripts_path + "/scripts/moco_parallel.py "
                                       "--fly_directory {fly_folder_to_process_oak} "
                                       "--dataset_path {dataset_path} "
                                       "--brain_paths_ch1 {input.brain_paths_ch1} "
                                       "--brain_paths_ch2 {input.brain_paths_ch2} "
                                       "--brain_paths_ch3 {input.brain_paths_ch3} "
                                       "--mean_brain_paths_ch1 {input.mean_brain_paths_ch1} "
                                       "--mean_brain_paths_ch2 {input.mean_brain_paths_ch2} "
                                       "--mean_brain_paths_ch3 {input.mean_brain_paths_ch3} "
                                       "--STRUCTURAL_CHANNEL {STRUCTURAL_CHANNEL} "
                                       "--FUNCTIONAL_CHANNELS {FUNCTIONAL_CHANNELS} "
                                       "--moco_path_ch1 {output.moco_path_ch1} "
                                       "--moco_path_ch2 {output.moco_path_ch2} "
                                       "--moco_path_ch3 {output.moco_path_ch3} "
                                       "--par_output {output.par_output} "
                                       "--moco_temp_folder {moco_temp_folder} "
rule moco_mean_brain_rule_func:
    """
    """
    threads: snake_utils.threads_per_memory
    resources:
        mem_mb=snake_utils.mem_mb_times_input,
        runtime='10m'# should be enough
    input:
        str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths_func}/moco/channel_{meanbr_moco_ch_func}_moco_func.nii"
    output:
        str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths_func}/moco/channel_{meanbr_moco_ch_func}_moco_mean_func.nii"
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
                error_stack=error_stack)

rule moco_mean_brain_rule_struct:
    """
    """
    threads: snake_utils.threads_per_memory
    resources:
        mem_mb=snake_utils.mem_mb_times_input,
        runtime='10m'# should be enough
    input:
        str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths_struct}/moco/channel_{meanbr_moco_ch_struct}_moco_struct.nii"
    output:
        str(fly_folder_to_process_oak) + "/{moco_meanbr_imaging_paths_struct}/moco/channel_{meanbr_moco_ch_struct}_moco_mean_struct.nii"
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
                error_stack=error_stack)

rule zscore_rule:
    """
    David func0
    Nodes: 1
    Cores per node: 15
    CPU Utilized: 00:01:32
    CPU Efficiency: 1.84% of 01:23:30 core-walltime
    Job Wall-clock time: 00:05:34
    Memory Utilized: 65.03 GB
    Memory Efficiency: 58.42% of 111.33 GB
    
    Cores per node: 16
    CPU Utilized: 00:01:31
    CPU Efficiency: 1.56% of 01:37:04 core-walltime
    Job Wall-clock time: 00:06:04
    Memory Utilized: 45.13 GB
    Memory Efficiency: 36.11% of 124.96 GB

    Yandan func
    Cores per node: 15
    CPU Utilized: 00:01:22
    CPU Efficiency: 2.31% of 00:59:15 core-walltime
    Job Wall-clock time: 00:03:57
    Memory Utilized: 27.29 GB
    Memory Efficiency: 24.52% of 111.33 GB
    
    
    TODO what's the runtime??
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
    threads: snake_utils.threads_per_memory_much_more
    resources:
        mem_mb=snake_utils.mem_mb_much_more_times_input,

    input:
        path_ch1 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/moco/channel_1_moco_func.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        path_ch2 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/moco/channel_2_moco_func.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        path_ch3 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/moco/channel_3_moco_func.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else[],

    output:
        zscore_path_ch1 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_1_moco_zscore.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_2_moco_zscore.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3 = str(fly_folder_to_process_oak) + "/{zscore_imaging_paths}/channel_3_moco_zscore.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    run:
        try:
            preprocessing.zscore(fly_directory=fly_folder_to_process_oak,
                                dataset_path=[input.path_ch1, input.path_ch2, input.path_ch3],
                                zscore_path=[output.zscore_path_ch1, output.zscore_path_ch2, output.zscore_path_ch3])
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_zscore')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)

rule temporal_high_pass_filter_rule:
    """
    David func 30min -> Timeout at 50 min runtime! Increase to 90
    State: TIMEOUT (exit code 0)
    Nodes: 1
    Cores per node: 10
    CPU Utilized: 00:00:00
    CPU Efficiency: 0.00% of 08:34:10 core-walltime
    Job Wall-clock time: 00:51:25
    Memory Utilized: 109.00 KB
    Memory Efficiency: 0.00% of 70.84 GB
    
    David func 30min -> Timeout!
    Nodes: 1
    Cores per node: 10
    CPU Utilized: 00:00:00
    CPU Efficiency: 0.00% of 05:06:20 core-walltime
    Job Wall-clock time: 00:30:38
    Memory Utilized: 110.00 KB
    Memory Efficiency: 0.00% of 70.84 GB
    
    Just made it
    CPU Utilized: 00:25:45
    CPU Efficiency: 8.47% of 05:04:00 core-walltime
    Job Wall-clock time: 00:30:24
    Memory Utilized: 46.22 GB
    Memory Efficiency: 58.13% of 79.52 GB
    
    Yandan func0 dat
    Cores per node: 10
    CPU Utilized: 00:19:31
    CPU Efficiency: 8.78% of 03:42:10 core-walltime
    Job Wall-clock time: 00:22:13
    Memory Utilized: 41.26 GB
    Memory Efficiency: 58.24% of 70.84 GB
    
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
    threads: snake_utils.threads_per_memory_more
    resources:
        mem_mb=snake_utils.mem_mb_more_times_input,
        runtime='90m' # The call to 1d smooth takes quite a bit of time! Todo< make dynamic for longer recordings!
    input:
        zscore_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        zscore_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],
    output:
        temp_HP_filter_path_ch1=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_1_moco_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else [],
        temp_HP_filter_path_ch2=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_2_moco_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else [],
        temp_HP_filter_path_ch3=str(fly_folder_to_process_oak) + "/{temp_HP_filter_imaging_paths}/channel_3_moco_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else [],

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
                error_stack=error_stack)

rule correlation_rule:
    """
fictrac_path=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/fictrac/fictrac_behavior_data.dat",
        
    """
    threads: snake_utils.threads_per_memory_less
    resources:
        mem_mb=snake_utils.mem_mb_less_times_input,
        runtime='60m' # vectorization made this super fast
    input:
        corr_path_ch1=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/channel_1_moco_zscore_highpass.nii" if 'channel_1' in FUNCTIONAL_CHANNELS else[],
        corr_path_ch2=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/channel_2_moco_zscore_highpass.nii" if 'channel_2' in FUNCTIONAL_CHANNELS else[],
        corr_path_ch3=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/channel_3_moco_zscore_highpass.nii" if 'channel_3' in FUNCTIONAL_CHANNELS else[],
        fictrac_path=str(fly_folder_to_process_oak) + "/{corr_imaging_paths}/" + str(fictrac_rel_path_correlation) + '/fictrac_behavior_data.dat',
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
                                fictrac_path=input.fictrac_path,
                                 
            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_correlation')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)
'''           
rule STA_rule:
    """TODO"""'''



rule clean_anatomy_rule:
    """
    """
    threads: snake_utils.threads_per_memory_much_more
    resources:
        mem_mb=snake_utils.mem_mb_much_more_times_input, # Todo, optimize memory usage of this function! #mem_mb_more_times_input #snake_utils.mem_mb_times_input # OOM!!!
        runtime='5m'
    input: str(fly_folder_to_process_oak) + "/{clean_anatomy_paths}/moco/channel_{clean_anat_ch}_moco_mean_struct.nii",
    output: str(fly_folder_to_process_oak) + "/{clean_anatomy_paths}/moco/channel_{clean_anat_ch}_moco_mean_clean.nii",
    run:
        try:
            preprocessing.clean_anatomy(fly_directory=fly_folder_to_process_oak,
                                        path_to_read=input,
                                        save_path=output)
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_clean_anatomy')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)

rule make_supervoxels_rule:
    """
    David, 30min func
    Cores per node: 8
    CPU Utilized: 00:11:23
    CPU Efficiency: 10.34% of 01:50:08 core-walltime
    Job Wall-clock time: 00:13:46
    Memory Utilized: 27.31 GB
    Memory Efficiency: 48.08% of 56.80 GB
    
    Yandan fun0
    Cores per node: 6
    CPU Utilized: 00:08:18
    CPU Efficiency: 13.41% of 01:01:54 core-walltime
    Job Wall-clock time: 00:10:19
    Memory Utilized: 30.96 GB
    Memory Efficiency: 61.18% of 50.60 GB
    
    TODO: Find optimal runtime!
    Yamda, func0
    Cores per node: 6
    CPU Utilized: 00:07:40
    CPU Efficiency: 11.63% of 01:05:54 core-walltime
    Job Wall-clock time: 00:10:59
    Memory Utilized: 22.95 GB
    Memory Efficiency: 45.36% of 50.60 GB
    """
    threads: snake_utils.threads_per_memory
    resources:
        mem_mb=snake_utils.mem_mb_times_input,
        runtime='20m'
    input: str(fly_folder_to_process_oak) + "/{supervoxel_paths}/channel_{supervoxel_ch}_moco_zscore_highpass.nii"
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
                error_stack=error_stack)

# Probably Bifrost does it better.
rule func_to_anat_rule:
    """
  
    """
    threads: snake_utils.threads_per_memory_more
    resources:
        mem_mb=snake_utils.mem_mb_more_times_input,
        runtime='10m' # should be enough, is super quick already
    input:
        path_to_read_fixed=str(fly_folder_to_process_oak) + "/" + str(list_of_paths_struct) + '/moco/{func2anat_fixed}_moco_mean.nii',
        path_to_read_moving=str(fly_folder_to_process_oak) + "/{func2anat_paths}/moco/{func2anat_moving}_moco_mean.nii"
    output: str(fly_folder_to_process_oak) + "/{func2anat_paths}/warp/{func2anat_moving}_func-to-{func2anat_fixed}_anat.nii"

    run:
        try:
            preprocessing.align_anat(fly_directory=fly_folder_to_process_oak,
                rule_name='func_to_anat',
                fixed_fly='anat',# this is important for where transform params are saved
                moving_fly='func',# this is important for where transform params are saved # TODO change this to
                # something dynamic, otherwise more than 1 channel won't work as expected!!!
                path_to_read_fixed=[input.path_to_read_fixed],
                path_to_read_moving=[input.path_to_read_moving],
                path_to_save=output,
                resolution_of_fixed=None,#(
                #0.653, 0.653, 1),# Copy-paste from brainsss, probably can be read from metadate.xml!
                resolution_of_moving=None,#(
                #2.611, 2.611, 5),# Copy-paste from brainsss, probably can be read from metadate.xml!
                iso_2um_resample = 'fixed',
                #iso_2um_fixed=True,
                #iso_2um_moving=False,
                type_of_transform='Affine',# copy-paste from brainsss
                grad_step=0.2,
                flow_sigma=3,
                total_sigma=0,
                syn_sampling=32
            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_func2anat')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)

rule anat_to_atlas:
    """
    """
    threads: snake_utils.threads_per_memory_more
    resources:
        mem_mb=snake_utils.mem_mb_more_times_input,
        runtime='10m'
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
                rule_name='anat_to_atlas',
                fixed_fly='meanbrain',# this is important for where transform params are saved
                moving_fly='anat',# this is important for where transform params are saved # TODO change this to
                # something dynamic, otherwise more than 1 channel won't work as expected!!!
                path_to_read_fixed=[input.path_to_read_fixed],
                path_to_read_moving=[input.path_to_read_moving],
                path_to_save=output,
                resolution_of_fixed=(2, 2, 2),# Copy-paste from brainsss, probably can be read from metadate.xml!
                resolution_of_moving=None,#(
                #0.653, 0.653, 1),# Copy-paste from brainsss, probably can be read from metadate.xml!
                iso_2um_resample='moving',
                #iso_2um_fixed=False,
                #iso_2um_moving=True,
                type_of_transform='SyN',# copy-paste from brainsss
                grad_step=0.2,
                flow_sigma=3,
                total_sigma=0,
                syn_sampling=32
            )
        except Exception as error_stack:
            logfile = utils.create_logfile(fly_folder_to_process_oak,function_name='ERROR_anat2atlas')
            utils.write_error(logfile=logfile,
                error_stack=error_stack)

rule apply_transforms_rule:
    """
    Not finished/tested yet!
    """
    threads: snake_utils.threads_per_memory
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
                error_stack=error_stack)


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

'''rule motion_correction_parallel_CHUNKS_rule:
    """
    Old - didn't work as well as the other parallel motion correction rule (took longer and higher memory
    requirements)
    """
    threads: 32 # the max that we can do - check with sh_part
    resources:
        mem_mb=snake_utils.mem_mb_much_more_times_input,
        runtime=snake_utils.time_for_moco_input # runtime takes input as seconds!
    input:
        # Only use the Channels that exists - this organizes the anatomy and functional paths inside the motion correction
        # module.
        brain_paths_ch1=str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_1.nii" if CH1_EXISTS else [],
        brain_paths_ch2=str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_2.nii" if CH2_EXISTS else [],
        brain_paths_ch3=str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_3.nii" if CH3_EXISTS else [],

        mean_brain_paths_ch1= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_1_mean.nii" if CH1_EXISTS else [],
        mean_brain_paths_ch2= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_2_mean.nii" if CH2_EXISTS else [],
        mean_brain_paths_ch3= str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/imaging/channel_3_mean.nii" if CH3_EXISTS else [],
    output:
        moco_path_ch1 = str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_1_moco.nii" if CH1_EXISTS else[],
        moco_path_ch2 = str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_2_moco.nii" if CH2_EXISTS else[],
        moco_path_ch3 = str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/channel_3_moco.nii" if CH3_EXISTS else[],
        par_output = str(fly_folder_to_process_oak) + "/{moco_imaging_paths}/moco/motcorr_params.npy"

    shell: "python3 scripts/motion_correction_parallel_chunks.py "
            "--fly_directory {fly_folder_to_process_oak} "
            "--brain_paths_ch1 {input.brain_paths_ch1} "
            "--brain_paths_ch2 {input.brain_paths_ch2} "
            "--brain_paths_ch3 {input.brain_paths_ch3} "
            "--mean_brain_paths_ch1 {input.mean_brain_paths_ch1} "
            "--mean_brain_paths_ch2 {input.mean_brain_paths_ch2} "
            "--mean_brain_paths_ch3 {input.mean_brain_paths_ch3} "
            "--ANATOMY_CHANNEL {ANATOMY_CHANNEL} "
            "--FUNCTIONAL_CHANNELS {FUNCTIONAL_CHANNELS} "
            "--moco_path_ch1 {output.moco_path_ch1} "
            "--moco_path_ch2 {output.moco_path_ch2} "
            "--moco_path_ch3 {output.moco_path_ch3} "
            "--par_output {output.par_output} "'''
'''    
rule motion_correction_rule:
    """
    First try: essentially emulates Bella's moco. Takes long as multiprocessing is 'hidden' in ants.registration

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
                error_stack=error_stack)

'''