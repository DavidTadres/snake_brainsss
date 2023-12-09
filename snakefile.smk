## Testing ##
import hello_world

rule HelloSnake:
    #shell:
    #    "python3 hello_world.py"
    run:
        hello_world.print_hi('test')

"""
import pathlib
from stitch_split_nii import find_split_files

# David's datapaths
original_data_path = '/oak/stanford/groups/trc/data/David/Bruker/imports'
#target_data_path = '/Volumes/groups/trc/data/David/Bruker/preprocessed'

current_fly = pathlib.Path(original_data_path, '20231201')
print(current_fly)

rule stitch_split_nii:
    input:
        current_fly
    run:
        find_split_files(input)
        """