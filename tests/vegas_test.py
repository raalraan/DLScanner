from dlscanner.utilities.vegas import vegas_map_samples
import numpy as np


def fvgm(x):
    gau1 = np.exp(-((x - np.array([0.5, 0.5, 0.5]))**2).sum(axis=1)/0.2**2)
    gau0 = np.exp(-((x - np.array([0., 0., 0.]))**2).sum(axis=1)/0.2**2)
    gau2 = np.exp(-((x - np.array([-0.5, -0.5, -0.5]))**2).sum(axis=1)/0.2**2)
    fvgm = gau1 + gau2 + gau0
    return fvgm


xtst = np.random.uniform(0, 1, (1000, 3))

vgmap = vegas_map_samples(xtst, fvgm(xtst))
print(
    vgmap(100)
)




