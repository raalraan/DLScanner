import vegas
import numpy as np


def convert_to_unit_cube(x, limits):
    ndim = x.shape[1]
    new_x = np.empty(x.shape)

    for k in range(ndim):
        width = limits[k][1] - limits[k][0]
        new_x[:, k] = (x[:, k] - limits[k][0])/width

    return new_x


def convert_to_limits(x, limits):
    ndim = x.shape[1]
    new_x = np.empty(x.shape)

    for k in range(ndim):
        width = limits[k][1] - limits[k][0]
        # new_x[:, k] = (x[:, k] - limits[k][0])/width
        new_x[:, k] = x[:, k]*width + limits[k][0]

    return new_x



def vegas_map_samples(
        xtrain, ftrain, limits,
        ninc=100,
        nitn=5,
        alpha=1.0,
        nproc=1
):
    '''Train a mapping of the parameter space using vegas and a sample of points.
    Input Args:
        xtrain: array
            Coordinates of the sample. All the coordinates must be normalized to the [0, 1] range
        ftrain: array
            Result of evaluating a function on xtrain.
        ninc: int, optional
            number of increments used in the mapping (see vegas documentation for AdaptiveMap)
        nitn: int, optional
            number of iterations used to refine mapping (see vegas documentation for AdaptiveMap.adapt_to_samples())
        alpha: float, optional
            Damping parameter (see vegas documentation for AdaptiveMap.adapt_to_samples())
        nproc: int, optional
            Number of processes/processors to use (see vegas documentation for AdaptiveMap.adapt_to_samples())
    Returns:
        Callable function to create a random sample using the trained mapping
    '''
    ndim = xtrain.shape[1]
    _xtrain = convert_to_unit_cube(xtrain, limits)
    vg_AdMap = vegas.AdaptiveMap([[0, 1]]*ndim, ninc=ninc)
    vg_AdMap.adapt_to_samples(
        _xtrain, ftrain,
        nitn=nitn, alpha=alpha, nproc=nproc
    )

    def _vegas_sample(npts):
        '''Obtain an array of points from a trained vegas map.
        Input Args:
            npts: int
                Number of points
        Returns:
            sample: array
                Sample of points created according to mapping
            jacobian: array
                Jacobian corresponding to mapping of the points
        '''
        xrndu = np.random.uniform(0, 1, (int(_xtrain.shape[0]),int(_xtrain.shape[1])))
        xrndmap = np.empty(xrndu.shape, xrndu.dtype)
        jacmap = np.empty(xrndu.shape[0], xrndu.dtype)
        vg_AdMap.map(xrndu, xrndmap, jacmap)

        return convert_to_limits(xrndmap, limits), jacmap
        

    return _vegas_sample
