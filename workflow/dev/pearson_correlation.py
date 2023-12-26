from scipy.stats import pearsonr
import numpy as np

# x,y,z,t
brain = np.random.rand(8,16,49,20)
# time, z
fictrac = np.random.rand(49, 20) # THIS WAS SWAPPED FOR EASE OF READING
idx_to_use = list(range(20))

def pearson_func():
    """
    python -m timeit -s 'import pearson_correlation' 'pearson_correlation.pearson_func()'
    49 z slices: 1 loop, best of 5: 1.77 sec per loop

    2 loops, best of 5: 105 msec per loop
    :return:
    """
    corr_brain = np.zeros((brain.shape[0], brain.shape[1], brain.shape[2]))

    for z in range(brain.shape[2]):
        fictrac_now = fictrac[z,:]
        for x in range(brain.shape[0]):
            for y in range(brain.shape[1]):
                corr_brain[x,y,z] = pearsonr(fictrac_now[idx_to_use], brain[x,y,z,:][idx_to_use])[0]

def vectorized_func():
    """
    python -m timeit -s 'import pearson_correlation' 'pearson_correlation.vectorized_func()'
    49 z slices: 200 loops, best of 5: 1.16 msec per loop

    > This should therefore only take ~0.1% of the time the scipy pearson function takes. For
    > 30 minutes volume where Yandan's log showed ~30 minutes, it should only take an instant.
    :return:
    """
    # Formula for correlation coefficient
    # r = sum(x-m_x)*(y-m_y) / sqrt(sum(x-m_x)^2 * sum(y-m_y)^2)
    vect_corr_brain = np.zeros((brain.shape[0], brain.shape[1], brain.shape[2]))
    for z in range(brain.shape[2]):
        # Here we calculate the mean of x and y over t for a give z slice
        brain_mean = brain[:,:,z,:].mean(axis=-1, dtype=np.float32)
        fictrac_mean = fictrac[z,:].mean(dtype=np.float32)

        # This yields brain_mean_m for xy and t for a given z
        brain_mean_m = brain[:,:,z,:].astype(np.float32) - brain_mean[:,:,None]
        fictrac_mean_m = fictrac[z,:].astype(np.float32) - fictrac_mean

        normbrain = np.linalg.norm(brain_mean_m, axis=-1)
        normfictrac = np.linalg.norm(fictrac_mean_m)

        vect_corr_brain[:,:,z] = np.dot(brain_mean_m/normbrain[:,:,None], fictrac_mean_m/normfictrac)#, -1.0, 1.0

def fully_vectorized_func():
    """
    DONT THINK THIS CAN WORK! Do the mostly vectorized function instead. Already 1000x times.
    :return:
    """
    brain = np.random.rand(8, 16, 49, 20)
    # Here we calculate the mean of x and y and z over time.
    brain_mean = brain.mean(axis=-1)
    # We get the mean for each z
    fictrac_mean = fictrac.mean(axis=-1, dtype=np.float32)

    # We get a (8,16,49,20) shape array here
    #brain_mean_m = brain.astype(np.float32) - brain_mean[:, :, :, np.newaxis]
    #inplace
    brain-=brain_mean[:,:,:,np.newaxis]
    fictrac_mean_m = fictrac.astype(np.float32) - fictrac_mean[:, np.newaxis]

    # (8, 16, 49)
    #normbrain = np.linalg.norm(brain_mean_m, axis=-1)
    normbrain = np.linalg.norm(brain, axis=-1)
    # (49,)
    normfictrac = np.linalg.norm(fictrac_mean_m, axis=-1)

    #foo = brain_mean_m / normbrain[:, :, :, np.newaxis]
    # inplace
    brain/=normbrain[:,:,:,np.newaxis]
    bar = fictrac_mean_m / normfictrac[:,np.newaxis]

    #fvect_corr_brain = np.dot(brain_mean_m / normbrain[:, :, :, np.newaxis], fictrac_mean_m / normfictrac[:,np.newaxis])  # , -1.0, 1.
    fvect_corr_brain = np.dot(brain.reshape(brain.shape[0], brain.shape[1], int(brain.shape[2]*brain.shape[3])),
                              bar.reshape(int(bar.shape[0]*bar.shape[1])))

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