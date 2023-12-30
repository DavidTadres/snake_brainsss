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


path_original = pathlib.Path('/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_308/func_0')
path_my = pathlib.Path('/oak/stanford/groups/trc/data/David/Bruker/preprocessed/nsybGCaMP_tdTomato/fly_002/func_0')

file_path_original = pathlib.Path(path_original, 'functional_channel_2_moco_zscore.h5')
file_path_my = pathlib.Path(path_my, 'channel_2_moco_zscore.h5')

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