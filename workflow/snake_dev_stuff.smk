"""
How to use - this is just another way of calling a script on sherlock.
Just import the py file that contains your function of interest and put it
under the 'run' as indicated.
Needs to be a function

ONLY ONE RULE PER RUN - COMMENT WHAT YOU DONT NEED
"""

import pathlib
scripts_path = pathlib.Path(__file__).resolve()  # path of workflow i.e. /Users/dtadres/snake_brainsss/workflow
from dev import compare_h5_large_data
from dev import compare_registration_results
#from dev import moco_timing
#from dev import visualize_brain_original
root_path = pathlib.Path('/oak/stanford/groups/trc/data')
path_original = pathlib.Path(root_path, 'Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/func_0/moco/functional_channel_1_moco.h5')
path_my = pathlib.Path(root_path, 'David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_004/func_0/moco/channel_1_moco.nii')
savepath = pathlib.Path(root_path, 'David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_004/testing/func0_moco_comparison.png')

rule compare_registration_results_rule:
    threads: 16
    input:
        path_original=path_original,
        path_my=path_my
    run:
        compare_registration_results.compare_moco_results_nib(
            input.path_original,
            input.path_my, savepath)
'''
CH1_EXISTS = True
CH2_EXISTS = True
CH3_EXISTS = False

ANATOMY_CHANNEL = 'channel_1'
FUNCTIONAL_CHANNELS = []

rule test_moco_timing_rule:
    threads: 32
    input:
        brain_paths_ch1 = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_1.nii') if CH1_EXISTS else [],
        brain_paths_ch2 = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_2.nii') if CH2_EXISTS else [],
        brain_paths_ch3 = pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_3.nii') if CH3_EXISTS else [],

        mean_brain_paths_ch1= pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_1_mean.nii') if CH1_EXISTS else [],
        mean_brain_paths_ch2= pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_2_mean.nii') if CH2_EXISTS else [],
        mean_brain_paths_ch3= pathlib.Path('/Volumes/groups/trc/data/David/Bruker/preprocessed/fly_002/func0/imaging/channel_3_mean.nii') if CH3_EXISTS else []
    output:

    resources: mem_mb='60G'
    shell: "python3 scripts/motion_correction_parallel_chunks.py "
            "--brain_paths_ch1 {input.brain_paths_ch1} "
            "--brain_paths_ch2 {input.brain_paths_ch2} "
            "--brain_paths_ch3 {input.brain_paths_ch3} "
            "--mean_brain_paths_ch1 {input.mean_brain_paths_ch1} "
            "--mean_brain_paths_ch2 {input.mean_brain_paths_ch2} "
            "--mean_brain_paths_ch3 {input.mean_brain_paths_ch3} "
            "--ANATOMY_CHANNEL {ANATOMY_CHANNEL} "
            "--FUNCTIONAL_CHANNELS {FUNCTIONAL_CHANNELS}"

'''
'''
rule compare_correlation_results_rule:
    threads: 2
    resources: mem_mb='10G'
    run:
        visualize_brain_original.compare()
'''

'''

path_original = pathlib.Path('/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/func_0')
path_my = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0')

#file_path_original = pathlib.Path(path_original, 'functional_channel_2_moco_zscore_highpass.h5')
#file_path_original = pathlib.Path(path_original, 'moco/functional_channel_1_moco.h5')

#file_path_my = pathlib.Path(path_my, 'channel_2_moco_zscore_highpass.h5')
file_path_my = pathlib.Path(path_my, 'moco1/channel_1_moco.h5')
file_path_original = pathlib.Path(path_my, 'moco2/channel_1_moco.h5')

rule compare_large_arrays_rule:
    #shell:
    #    'python3 hello_world.py $args'
    threads: 8
    input:
        file_path_original=file_path_original,
        file_path_my=file_path_my
    resources: mem_mb='100G'
    run:
        compare_h5_large_data.run_comparison(
            path_original=input.file_path_original,
            path_my=input.file_path_my
        )
'''

####
#

'''
path_original = pathlib.Path(
    '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0/moco2/channel_1_moco.h5')
path_new = pathlib.Path(
    '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0/moco1/channel_1_moco.h5')
savepath = pathlib.Path(
    '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/testing/time_series_moco_twice_abs_diff.png')
rule compare_registration_rule:
    threads: 2
    resources: mem_mb='150G'
    run: compare_registration_results.compare_moco_results(path_original, path_new, savepath)
'''