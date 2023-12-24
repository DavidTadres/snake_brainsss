from scipy.stats import pearsonr
import numpy as np

brain = np.random.rand(128,256,49,20)
fictrac = np.random.rand(20)

corr_brain = np.zeros((brain.shape[0], brain.shape[1], brain.shape[2]))

for z in range(brain.shape[2]):
