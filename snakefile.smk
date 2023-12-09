## Testing ##
"""
snakemake \
    --s snakefile.smk \
    --jobs 1 \
    --default-resource mem_mb=100\
    --cluster '
        sbatch \
            --partition trc
            --cpus-per-task 16
            --ntasks 1
            --output ./logs/stitchlog3.out
            --open-mode append
            --mail-type=ALL
        ` \
        output.txt
"""

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