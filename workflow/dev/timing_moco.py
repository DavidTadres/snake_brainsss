"""
Moco is by far the slowest step in the preprocessing pipeline.
It's originally in 2 for loops, an outer one that splits the volume into 'chunks' and an inner loop that feeds
single frames to the ants.registration function.

I want to know how fast one call to ants.registration is.
"""

import h5py
import nibabel as nib
import pathlib
import matplotlib.pyplot as plt

fixed_path = pathlib.Path('')