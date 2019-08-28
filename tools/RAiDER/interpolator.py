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
    
