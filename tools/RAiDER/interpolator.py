#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
from RAiDER.util import parallel_apply_along_axis


class Interpolator:
    """
    Generic interpolator

    This model is based upon a linear interpolation scheme for pressure,
    temperature, and relative humidity.

    """
    #TODO: handle 1-D, 2-D cases correctly
    def __init__(self, xs = [], ys = [], zs = [], proj = None):
        """Initialize a NetCDFModel."""
        self._xs = xs
        self._ys = ys
        self._zs = zs
        self._zlevels = None
        self._values = None
        self._getShape(xs, ys, zs)
        self._Npoints = 0
        self.setPoints(xs, ys, zs)
        self._proj = proj
        self._fill_value = np.nan
        self._interp = []
        self._getBBox()

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

        s2 = '\n'
        s2 += 'Lat/Lon bounding box: \n'
        s2 += '\n'
        s2 += '-------{:.2f}-------\n'.format(self._bbox[1][1])
        s2 += '|                   |\n'
        s2 += '|                   |\n'
        s2 += '|                   |\n'
        s2 += '{:.2f}         {:.2f}\n'.format(self._bbox[0][0], self._bbox[1][0])
        s2 += '|                   |\n'
        s2 += '|                   |\n'
        s2 += '|                   |\n'
        s2 += '-------{:.2f}-------\n'.format(self._bbox[0][1])
        string += s2

        return str(string)

    def _getShape(self, xs, ys, zs):
        try:
            self._shape = max(xs.shape, ys.shape, zs.shape)
        except AttributeError:
            self._shape = (len(xs), len(ys), len(zs))

    def _getBBox(self):
        if len(self._xs) > 0:
            x1 = np.nanmin(self._xs)
            x2 = np.nanmax(self._xs)
            y1 = np.nanmin(self._ys)
            y2 = np.nanmax(self._ys)
            z1 = np.nanmin(self._zs)
            z2 = np.nanmax(self._zs)
            self._bbox = [(x1, y1, z1), (x2, y2, z2)]
        else:
            self._bbox = None

    def setProjection(self, proj):
        ''' 
        proj should be a prproj object
        '''
        self._proj = proj
 
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

        self._getShape(xs, ys, zs)
        self._Npoints = len(xs)
        
        try:
            self._zlevels = np.nanmean(zs, axis=(0,1))
        except:
            pass

        self._xs = xs.flatten()
        self._ys = ys.flatten()
        self._zs = zs.flatten()
        self._getBBox()
 
    def checkPoint(self, pnts):
        '''
        Checks whether the given point is within the bounding box
        represented by the bounds on x, y, and z. 
        Inputs:
          pnt    - should be an Mx3 numpy array (M number of points, 
                   3 is the dimension. Should be ordered [x, y, z])
        Outputs:
          inBBox - an Mx1 numpy array of True/False
        '''
        xs, ys, zs = pnts[:,0], pnts[:,1], pnts[:,2]
        ix = np.logical_and(np.less(self._bbox[0][0],xs), np.greater(self._bbox[1][0], xs))
        iy = np.logical_and(np.less(self._bbox[0][1],ys), np.greater(self._bbox[1][1], ys))
        iz = np.logical_and(np.less(self._bbox[0][2],zs), np.greater(self._bbox[1][2], zs))
        inBBox = np.logical_and.reduce((ix,iy,iz))
        return inBBox

    def printPoints(self):
        for pnt in zip(self._xs, self._ys, self._zs):
            print('{}\n'.format(pnt))

    def getInterpFcns(self, *args, interpType = 'scipy', **kwargs):
        '''
        return of list of interpolators, corresponding to the list of 
        inputs arguments
        '''
        if interpType=='scipy':
            self._make_scipy_interpolators(*args)
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
            newArg = arg.flatten()
            ix = np.isnan(newArg)
            _interpolators.append(lndi(points[~ix], 
                                      newArg[~ix], 
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
                          _interp3D(self._xs,self._ys, self._zs,arg, self._zlevels, self._shape)
                                 )
        self._interp = _interpolators

    def _make_sane_interpolators(self, *args, **kwargs):
        pass


def _interp3D(xs, ys, zs, values, zlevels, shape = None):
    '''
    3-D interpolation on a non-uniform grid, where z is non-uniform but x, y are uniform
    '''
    from scipy.interpolate import RegularGridInterpolator as rgi

    if shape is None:
       shape = max(xs.shape, ys.shape, zs.shape)

    # First interpolate uniformly in the z-direction
    nx, ny = shape[:2]
    zshaped = np.reshape(zs, shape)

    # if zs are upside down, flip to interpolate vertically
    if np.nanmean(zshaped[..., 0]) > np.nanmean(zshaped[...,-1]):
        zshaped = np.flip(zshaped, axis = 2)
        values = np.flip(values, axis = 2)

    # TODO: requires zs increase along the 2-axis  
    # zvalues = np.nanmean(zs, axis=(0,1))

    new_zs = np.tile(zlevels, (nx,ny,1))
    values = fillna3D(values)

    new_var = interp_along_axis(zshaped, new_zs,
                                  values, axis = 2)

    # This assumes that the input data is in the correct projection; i.e.
    # the native weather grid projection
    xvalues = np.unique(xs)
    yvalues = np.unique(ys)

    # TODO: is it preferable to have lats first? 
    interp= rgi((yvalues,xvalues, zlevels), new_var,
                           bounds_error=False, fill_value = np.nan)
    return interp


def interp_along_axis(oldCoord, newCoord, data, axis = 2, pad = False):
    '''
    Interpolate an array of 3-D data along one axis. This function 
    assumes that the x-xoordinate increases monotonically.

    Jeremy Maurer
    '''
    stackedData = np.concatenate([oldCoord, data, newCoord], axis = axis)
    try:
       out = parallel_apply_along_axis(interpVector, arr=stackedData, axis=axis, Nx=oldCoord.shape[axis])
    except: 
       out = np.apply_along_axis(interpVector, axis=axis,arr=stackedData, Nx=oldCoord.shape[axis])
    
    return out


def interpVector(vec, Nx): 
    '''
    Interpolate data from a single vector containing the original 
    x, the original y, and the new x, in that order. Nx tells the 
    number of original x-points. 
    '''
    from scipy import interpolate 
    x = vec[:Nx] 
    y = vec[Nx:2*Nx] 
    xnew = vec[2*Nx:] 
    f = interpolate.interp1d(x, y, bounds_error=False) 
    return f(xnew)


def fillna3D(array, axis = -1):
    '''
    Fcn to fill in NaNs in a 3D array by interpolating over one axis only
    '''
    # Need to handle each axis
    narr = np.moveaxis(array, axis, -1)
    shape = narr.shape
    y = narr.flatten()

    test_nan = np.isnan(y)
    finder = lambda z: z.nonzero()[0]
    
    y[test_nan]= np.interp(finder(test_nan), finder(~test_nan), y[~test_nan])
    newy = np.reshape(y, shape)
    final = np.moveaxis(newy, -1, axis)
    return final
    

class RegularConeInterpolator(object):
    """
    Interpolation on a regular cone in arbitrary 3d 
    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    def __init__(self, xgrid, ygrid, zgrid=None, values, method="linear", 
                 bounds_error=False,fill_value=np.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        self._xgrid = xgrid.copy()
        self._ygrid = ygrid.copy()
        self._zgrid = zgrid.copy()
        points = np.array([xgrid.flatten(), ygrid.flatten(), zgrid.flatten()])

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype')
                    and not np.can_cast(fill_value_dtype, values.dtype,
                                        casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        #for i, p in enumerate(points):
            #if not values.shape[i] == len(p):
            #    raise ValueError("There are %d points and %d values in "
            #                     "dimension %d" % (len(p), values.shape[i], i))

        grid_1 = np.arange(xgrid.shape[0])
        grid_2 = np.arange(ygrid.shape[1])
        grid_3 = np.arange(zgrid.shape[2])
        self.grid = tuple([np.asarray(p) for p in [grid_1, grid_2, grid_3]])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".
        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances, out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances, out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= .5, i, i + 1))
        return self.values[idx_res]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, xg, yg, zg in zip(xi, self._xgrid, self._ygrid, self._zgrid):
            xi_1 = np.argmin(np.abs(x[0]-xg), axis = (0,1))
            yi_1 = np.argmin(np.abs(x[1]-yg), axis = (0,1))
            zi_1 = np.argmin(np.abs(x[2]-zg), axis = 2)

            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds

