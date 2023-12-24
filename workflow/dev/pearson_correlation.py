from scipy.stats import pearsonr
import numpy as np

brain = np.random.rand(8,16,3,20)
fictrac = np.random.rand(20)

def pearson_func():
    """
    python -m timeit -s 'import pearson_correlation' 'pearson_correlation.pearson_func()'
    2 loops, best of 5: 105 msec per loop
    :return:
    """
    corr_brain = np.zeros((brain.shape[0], brain.shape[1], brain.shape[2]))

    for z in range(brain.shape[2]):
        for x in range(brain.shape[0]):
            for y in range(brain.shape[1]):
                corr_brain[x,y,z] = pearsonr(fictrac, brain[x,y,z,:])[0]

def vectorized_func():
    """
    python -m timeit -s 'import pearson_correlation' 'pearson_correlation.vectorized_func()'
    10000 loops, best of 5: 32.6 usec per loop

    > This should therefore only take ~0.03% of the time the scipy pearson function takes. For
    > 30 minutes volume where Yandan's log showed ~30 minutes, it should only take an instant.
    :return:
    """
    # Formula for correlation coefficient
    # r = sum(x-m_x)*(y-m_y) / sqrt(sum(x-m_x)^2 * sum(y-m_y)^2)
    brain_mean = brain.mean(axis=-1, dtype=np.float64)
    fictrac_mean = fictrac.mean(dtype=np.float64)

    brain_mean_m = brain.astype(np.float64) - brain_mean[:,:,:,None]
    fictrac_mean_m = fictrac.astype(np.float64) - fictrac_mean

    normbrain = np.linalg.norm(brain_mean_m, axis=-1)
    normfictrac = np.linalg.norm(fictrac_mean_m)

    vect_corr_brain = np.dot(brain_mean_m/normbrain[:,:,:,None], fictrac_mean_m/normfictrac)#, -1.0, 1.0

import timeit
timeit.timeit(stmt=vectorized_func, number=10)
"""
# the values are slightly different, I think because of floating point rounding.
# If I round to 14 decimals, the result is NOT the same
decimals = 14
if (np.round(corr_brain,decimals) == np.round(vect_corr_brain,decimals)).all():
    print('identical results at ' + repr(decimals))
else:
    print('not identical results' + repr(decimals))

decimals = 13
if (np.round(corr_brain,decimals) == np.round(vect_corr_brain,decimals)).all():
    print('identical results at ' + repr(decimals))
else:
    print('not identical results' + repr(decimals))"""