"""Helper utilities for file readers."""

import numpy as np
import pyproj
import scipy
from scipy.interpolate import RegularGridInterpolator as rgi
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
        self.setPoints(xs, ys, zs)
        self._proj = old_proj
        self._fill_value = np.nan

        if new_proj is not None:
            self.project(self._old_proj, self._new_proj)

    def __repr__(self):
        '''
        print method for Interpolator object
        '''
        string = '='*6 + 'Interpolator Object' + '='*6 + '\n'
        string += 'Total Number of points: {}\n'.format(len(self._xs))
        string += 'Shape of points: {}\n'.format(self._shape)
        string += 'Current Projection: {}\n'.format(self._proj)
        string += '='*30 + '\n'
        return str(string)
 
    def project(self, old_proj, new_proj):
        '''
        Project a set of points into a different coordinate system.
        old_proj and new_proj should be pyproj projection objects. 
        '''
        self._xs, self._ys, self._zs = pyproj.transform(old_proj,new_proj, self._xs, self._ys, self._zs)
        self._proj = new_proj

    def setPoints(self, xs, ys, zs):
        '''
        Assign x, y, z points to use for interpolating
        '''
        if xs is not None:

            try:
                self._shape = xs.shape
            except AttributeError:
                self._shape = (len(xs), len(ys), len(zs))

            self._xs = xs.flatten()
        if ys is not None:
            self._ys = ys.flatten()
        if zs is not None:
            self._zs = zs.flatten()

    def getInterpFcns(self, *args, use_pure_scipy = False, **kwargs):
        '''
        return of list of interpolators, corresponding to the list of 
        inputs arguments
        '''
        if use_pure_scipy:
            return self._make_scipy_interpolators()
        else:
            return self._make_sane_interpolators()

    def _make_scipy_interpolators(self): 
        '''
        Interpolate a fcn or list of functions (contained 
        in *args) that are defined on a regular grid using
        scipy.  
        '''
        points = np.stack([self._xs, self._ys, self._zs], axis = -1)
        _interpolators = []
        for arg in self._argsList:
            _interpolators.append(rgi(points, 
                                      arg.flatten(), 
                                      method = 'linear', 
                                      bounds_error = False, 
                                      fill_value = self._fill_value)
                                       )
        return _interpolators

    def _make_sane_interpolators(self): 
        '''
        Interpolate a fcn or list of functions (contained 
        in *args) that are defined on a regular grid. Use
        the _sane_interpolate method. 
        '''
        _interpolators = []
        for arg in self._argsList:
            _interpolators.append(
                          _sane_interpolate(self._xs,
                                            self._ys, 
                                            self._zs, 
                                            self._argsList,
                                            self._old_proj, 
                                            self._new_proj,
                                            zmin)
                                            )

        return _interpolators

def _sane_interpolate(xs, ys, zs, values_list, old_proj, new_proj, zmin):
    '''
    do some interpolation
    '''
    # just a check for the consistency with Ray's old code: 
    ecef = pyproj.Proj(proj='geocent')
    if old_proj != ecef:
        import pdb
        pdb.set_trace()

    num_levels = 2 * zs.shape[0]
    # First, find the maximum height
    new_top = np.nanmax(zs)

    new_zs = np.linspace(zmin, new_top, num_levels)

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


