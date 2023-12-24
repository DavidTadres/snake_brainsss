from scipy.stats import pearsonr
import numpy as np

brain = np.random.rand(128,256,49,20)
fictrac = np.random.rand(20)

corr_brain = np.zeros((brain.shape[0], brain.shape[1], brain.shape[2]))

for z in range(brain.shape[2]):
    for x in range(brain.shape[0]):
        for y in range(brain.shape[1]):
            corr_brain[x,y,z] = pearsonr(fictrac, brain[x,y,z,:])[0]
