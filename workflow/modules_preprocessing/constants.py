import numpy as np

class Constants():
    """
    Easy access to constants across modules
    """
    def __init__(self):
        ####################
        # GLOBAL VARIABLES #
        ####################
        self.WIDTH = 120  # This is used in all logging files

        # Bruker gives us data going from 0-8191 so we load it as uint16.
        # However, as soon as we do the ants.registration step (e.g in
        # function motion_correction() we get floats back.
        # The original brainsss usually saved everything in float32 but some
        # calculations were done with float64 (e.g. np.mean w/o defining dtype).
        # To be more explicit, I define here two global variables.
        # Dtype is what was mostly called when saving and loading data
        self.DTYPE = np.float32
        # Dtype_calculation is what I explicity call e.g. during np.mean
        self.DTYPE_CACLULATIONS = np.float32

