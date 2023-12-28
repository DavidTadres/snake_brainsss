"""
How long does it take to check large arrays for np.nan and np.inf?
"""

import numpy as np

# x,y,z,t
brain = np.random.rand(256, 128, 49, 600)  # that's a short 5 minute recording


def check_for_nan_and_inf_func():
    """
    python -m timeit -s 'import check_nan_and_inf' 'check_nan_and_inf.check_for_nan_and_inf_func()'
    1 loop, best of 5: 892 msec per loop (on my Mac)
    -> Not super expensive. Will be ok to check every array. Will at most cost a few minutes
    on sherlock.

    Check if there are any nan or inf in the array that is being passed.

    :param array:
    :return:
    """
    if np.isnan(brain).any():
        print("!!!!! WARNING - THERE ARE NAN IN THE ARRAY !!!!!")
        print("The position(s) of np.nan is/are: " + repr(np.where(np.isnan(brain))))
    if np.isinf(brain).any():
        print("!!!!! WARNING - THERE ARE INF IN THE ARRAY !!!!!")
        print("The position(s) of np.inf is/are " + repr(np.where(np.isnan(brain))))
