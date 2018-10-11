"""Helper utilities for file readers."""

import numpy as np
import pyproj
import util

class Interpolator:
    """
    Generic interpolator

    This model is based upon a linear interpolation scheme for pressure,
    temperature, and relative humidity.

    """
    #TODO: handle 1-D, 2-D cases correctly
    def __init__(self, xs = None, ys = None, zs = None, old_proj = None, new_proj = None):
        """Initialize a NetCDFModel."""
        self._xs = []
        self._ys = []
        self._zs = []
        self._shape = None
        self._Npoints = 0
        self.setPoints(xs, ys, zs)
        self._proj = old_proj
        self._fill_value = np.nan
        self._interp = []
        
        if new_proj is not None:
            self.reProject(self._old_proj, self._new_proj)
        
    def __call__(self, newX):
        '''
        Interpolate onto new coordinates
        '''
        res = []
        for intFcn in self._interp:
            res.append(intFcn(newX))
        return res
        
    def __repr__(self):
        '''
        print method for Interpolator object
        '''
        string = '='*6 + 'Interpolator Object' + '='*6 + '\n'
        string += 'Total Number of points: {}\n'.format(len(self._xs))
        string += 'Number of points: {}\n'.format(self._Npoints)
        string += 'Current Projection: {}\n'.format(self._proj.srs if self._proj.srs is not None else None)
        string += 'Number of functions to interpolate: {}\n'.format(len(self._interp))
        string += '='*30 + '\n'
        return str(string)

    def setProjection(self, proj):
        ''' 
        proj should be a prproj object
        '''
        self._proj = proj
 
    def reProject(self, new_proj):
        '''
        Project a set of points into a different coordinate system.
        old_proj and new_proj should be pyproj projection objects. 
        '''
        self._xs, self._ys, self._zs = pyproj.transform(self._proj,new_proj, self._xs, self._ys, self._zs)
        self._proj = new_proj

    def setPoints(self, xs, ys, zs):
        '''
        Assign x, y, z points to use for interpolating
        If the z-dimension is not to be used, set that dimension 
        to an empty list, e.g.: zs = []
        '''
        if xs is None:
            xs, ys, zs = [], [], []

        if isinstance(xs, list):
            xs,ys,zs = np.array(xs),np.array(ys),np.array(zs)

        self._shape = max(xs.shape, ys.shape, zs.shape)
        self._Npoints = len(xs)
        self._xs = xs.flatten()
        self._ys = ys.flatten()
        self._zs = zs.flatten()

    def printPoints(self):
        for pnt in zip(self._xs, self._ys, self._zs):
            print('{}\n'.format(pnt))

    def getInterpFcns(self, *args, interpType= 'scipy', **kwargs):
        '''
        return of list of interpolators, corresponding to the list of 
        inputs arguments
        '''
        if interpType=='scipy':
            self._make_scipy_interpolators(*args)
        elif interpType=='_sane':
            self._make_sane_interpolators(*args)
        else:
            self._make_3D_interpolators(*args)


    def _make_scipy_interpolators(self, *args): 
        '''
        Interpolate a fcn or list of functions (contained 
        in *args) that are defined on a regular grid using
        scipy.  
        '''
        from scipy.interpolate import LinearNDInterpolator as lndi
        points = np.stack([self._xs, self._ys, self._zs], axis = -1)
        _interpolators = []
        for arg in args:
            _interpolators.append(lndi(points, 
                                      arg.flatten(), 
                                      fill_value = self._fill_value
                                      )
                                 )
        self._interp = _interpolators


    def _make_3D_interpolators(self, *args): 
        '''
        Interpolate a fcn or list of functions (contained 
        in *args) that are defined on a regular grid. Use
        the _sane_interpolate method. 
        '''
        _interpolators = []
        for arg in args:
            _interpolators.append(
                          _interp3D(self._xs,self._ys, self._zs,arg, self._shape)
                                 )
        self._interp = _interpolators


def _interp3D(xs, ys, zs, values, shape):
    '''
    3-D interpolation on a non-uniform grid, where z is non-uniform but x, y are uniform
    '''
    from scipy.interpolate import RegularGridInterpolator as rgi

    # First interpolate uniformly in the z-direction
    flipflag = False
    Nfac = 10
    NzLevels = Nfac * shape[2]
    nx, ny = shape[:2]
    zmax = np.nanmax(zs)
    zmin = np.nanmin(zs)
    zshaped = np.reshape(zs, shape)
    # if zs are upside down, flip to interpolate vertically
    if np.nanmean(zshaped[..., 0]) > np.nanmean(zshaped[...,-1]):
        flipflag = True
        zshaped = np.flip(zshaped, axis = 2)
        values = np.flip(values, axis = 2)

    dz = (zmax - zmin)/NzLevels
    zvalues = zmin + dz*np.arange(NzLevels)
    new_zs = np.tile(zvalues, (nx,ny,1))
    new_var = interp_along_axis(zshaped, new_zs,
                                  values, axis = 2, 
                                  method='linear', pad = True)
    
    import pdb
    pdb.set_trace()
    # This assumes that the input data is in the correct projection; i.e.
    # the native weather grid projection
    xvalues = np.unique(xs)
    yvalues = np.unique(ys)
    interp= rgi((xvalues, yvalues, zvalues), new_var,
                           bounds_error=False, fill_value = np.nan)
    return interp


def _sane_interpolate(xs, ys, zs, values_list, old_proj, new_proj, zmin):
    '''
    do some interpolation
    '''
    # just a check for the consistency with Ray's old code: 
    ecef = pyproj.Proj(proj='geocent')
    if old_proj != ecef:
        import pdb
        pdb.set_trace()

    NzLevels = 2 * zs.shape[0]
    # First, find the maximum height
    new_top = np.nanmax(zs)

    new_zs = np.linspace(zmin, new_top, NzLevels)

    inp_values = [np.zeros((len(new_zs),) + values.shape[1:])
                  for values in values_list]

    # TODO: do without a for loop
    for iv in range(len(values_list)):
        for x in range(values_list[iv].shape[1]):
            for y in range(values_list[iv].shape[2]):
                not_nan = np.logical_not(np.isnan(zs[:, x, y]))
                inp_values[iv][:, x, y] = scipy.interpolate.griddata(
                        zs[:, x, y][not_nan],
                        values_list[iv][:, x, y][not_nan],
                        new_zs,
                        method='cubic')
        inp_values[iv] = _propagate_down(inp_values[iv])

    interps = list()
    for iv in range(len(values_list)):
        # Indexing as height, ys, xs is a bit confusing, but it'll error
        # if the sizes don't match, so we can be sure it's the correct
        # order.
        f = scipy.interpolate.RegularGridInterpolator((new_zs, ys, xs),
                                                      inp_values[iv],
                                                      bounds_error=False)

        # Python has some weird behavior here, eh?
        def ggo(interp):
            def go(pts):
                xs, ys, zs = np.moveaxis(pts, -1, 0)
                a, b, c = pyproj.transform(old_proj, new_proj, xs, ys, zs)
                # Again we index as ys, xs
                llas = np.stack((c, b, a), axis=-1)
                return interp(llas)
            return go
        interps.append(ggo(f))

    return interps


def _just_pull_down(a, direction=-1):
    """Pull real values down to cover NaNs

    a might contain some NaNs which live under real values. We replace
    those NaNs with actual values. a must be a 3D array.
    """
    out = a.copy()
    z, x, y = out.shape
    for i in range(x):
        for j in range(y):
            held = None
            if direction == 1:
                r = range(z)
            elif direction == -1:
                r = range(z - 1, -1, -1)
            else:
                raise ValueError(
                        'Unsupported direction. direction should be 1 or -1')
            for k in r:
                val = out[k][i][j]
                if np.isnan(val) and held is not None:
                    out[k][i][j] = held
                elif not np.isnan(val):
                    held = val
    return out


def _propagate_down(a):
    """Try to fill in NaN values in a."""
    out = np.zeros_like(a)
    z, x, y = a.shape
    xs = np.arange(x)
    ys = np.arange(y)
    xgrid, ygrid = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack((xgrid, ygrid), axis=-1)
    for i in range(z):
        nans = np.isnan(a[i])
        nonnan = np.logical_not(nans)
        inpts = points[nonnan]
        apts = a[i][nonnan]
        outpoints = points.reshape(-1, 2)
        try:
            ans = scipy.interpolate.griddata(inpts, apts, outpoints,
                                             method='nearest')
        except ValueError:
            # Likely there aren't any (non-nan) values here, but we'll
            # copy over the whole thing to be safe.
            ans = a[i]
        out[i] = ans.reshape(out[i].shape)
    # I honestly have no idea if this will work
    return _just_pull_down(out)


# Minimum z (height) value on the earth

def _least_nonzero(a):
    """Fill in a flat array with the lowest nonzero value.
    
    Useful for interpolation below the bottom of the weather model.
    """
    out = np.full(a.shape[1:], np.nan)
    xlim, ylim = out.shape
    zlim = len(a)
    for x in range(xlim):
        for y in range(ylim):
            for z in range(zlim):
                val = a[z][x][y]
                if not np.isnan(val):
                    out[x][y] = val
                    break
    return out



def interp_along_axis(oldCoord, newCoord, data, axis = 2, inverse=False, method='linear', pad = True):
    """ 

    ***
    The following was taken from https://stackoverflow.com/questions/
    28934767/best-way-to-interpolate-a-numpy-ndarray-along-an-axis
    ***

    Interpolate vertical profiles of 3-D data, e.g. of atmospheric 
    variables using vectorized numpy operations

    This function assumes that the x-xoordinate increases monotonically

    ps:
    * Updated to work with irregularly spaced x-coordinate.
    * Updated to work with irregularly spaced newx-coordinate
    * Updated to easily inverse the direction of the x-coordinate
    * Updated to fill with nans outside extrapolation range
    * Updated to include a linear interpolation method as well
        (it was initially written for a cubic function)

    Peter Kalverla
    March 2018

    ***
    Modified by J. Maurer in Sept 2018.
    Added a fillna function to pad nans outside the bounds of the 
    data with the closest non-nan value, and re-named inputs
    ***

    --------------------
    More info:
    Algorithm from: http://www.paulinternet.nl/?page=bicubic
    It approximates y = f(x) = ax^3 + bx^2 + cx + d
    where y may be an ndarray input vector
    Returns f(newx)

    The algorithm uses the derivative f'(x) = 3ax^2 + 2bx + c
    and uses the fact that:
    f(0) = d
    f(1) = a + b + c + d
    f'(0) = c
    f'(1) = 3a + 2b + c

    Rewriting this yields expressions for a, b, c, d:
    a = 2f(0) - 2f(1) + f'(0) + f'(1)
    b = -3f(0) + 3f(1) - 2f'(0) - f'(1)
    c = f'(0)
    d = f(0)

    These can be evaluated at two neighbouring points in x and
    as such constitute the piecewise cubic interpolator.
    """

    # View of x and y with axis as first dimension
    if inverse:
        _x = np.moveaxis(oldCoord, axis, 0)[::-1, ...]
        _y = np.moveaxis(data, axis, 0)[::-1, ...]
        _newx = np.moveaxis(newCoord, axis, 0)[::-1, ...]
    else:
        _y = np.moveaxis(data, axis, 0)
        _x = np.moveaxis(oldCoord, axis, 0)
        _newx = np.moveaxis(newCoord, axis, 0)

    # Sanity checks
    if np.any(_newx[0] < _x[0]) or np.any(_newx[-1] > _x[-1]):
        print('Values outside the valid range will be filled in '\
              'with NaNs or the closest non-zero value (default)')
    if np.any(np.diff(_x, axis=0) < 0):
        raise ValueError('x should increase monotonically')
    if np.any(np.diff(_newx, axis=0) < 0):
        raise ValueError('newx should increase monotonically')

    # Cubic interpolation needs the gradient of y in addition to its values
    if method == 'cubic':
        # For now, simply use a numpy function to get the derivatives
        # This produces the largest memory overhead of the function and
        # could alternatively be done in passing.
        ydx = np.gradient(_y, axis=0, edge_order=2)

    # This will later be concatenated with a dynamic '0th' index
    ind = [i for i in np.indices(_y.shape[1:])]

    # Allocate the output array
    original_dims = _y.shape
    newdims = list(original_dims)
    newdims[0] = len(_newx)
    newy = np.zeros(newdims)

    # set initial bounds
    i_lower = np.zeros(_x.shape[1:], dtype=int)
    i_upper = np.ones(_x.shape[1:], dtype=int)
    x_lower = _x[0, ...]
    x_upper = _x[1, ...]

    for i, xi in enumerate(_newx):
        # Start at the 'bottom' of the array and work upwards
        # This only works if x and newx increase monotonically

        # Update bounds where necessary and possible
        needs_update = (xi > x_upper) & (i_upper+1<len(_x))
        # print x_upper.max(), np.any(needs_update)
        while np.any(needs_update):
            i_lower = np.where(needs_update, i_lower+1, i_lower)
            i_upper = i_lower + 1
            x_lower = _x[[i_lower]+ind]
            x_upper = _x[[i_upper]+ind]

            # Check again
            needs_update = (xi > x_upper) & (i_upper+1<len(_x))

        # Express the position of xi relative to its neighbours
        xj = (xi-x_lower)/(x_upper - x_lower)

        # Determine where there is a valid interpolation range
        within_bounds = (_x[0, ...] < xi) & (xi < _x[-1, ...])

        if method == 'linear':
            f0, f1 = _y[[i_lower]+ind], _y[[i_upper]+ind]
            a = f1 - f0
            b = f0

            newy[i, ...] = np.where(within_bounds, a*xj+b, np.nan)

        elif method=='cubic':
            f0, f1 = _y[[i_lower]+ind], _y[[i_upper]+ind]
            df0, df1 = ydx[[i_lower]+ind], ydx[[i_upper]+ind]

            a = 2*f0 - 2*f1 + df0 + df1
            b = -3*f0 + 3*f1 - 2*df0 - df1
            c = df0
            d = f0

            new_data = np.where(within_bounds, a*xj**3 + b*xj**2 + c*xj + d, np.nan)

            if pad:
                newy[i, ...] = fillna(new_data)
            else:
                newy[i, ...] = new_data

        else:
            raise ValueError("invalid interpolation method"
                             "(choose 'linear' or 'cubic')")

    if inverse:
        newy = newy[::-1, ...]


    return np.moveaxis(newy, 0, axis)


def fillna(array):
    '''
    Fcn to fill in NaNs in a 2-D array of interferometric phase
    '''
    from scipy.ndimage import distance_transform_edt as dte

    mask = np.isnan(array)
    ind = dte(mask,return_distances=False,return_indices=True)
    new_array = array[tuple(ind)] 
    return new_array

