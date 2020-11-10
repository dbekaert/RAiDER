#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import copy
import os
import warnings
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from skgstat import estimators, models, binning


class Variogram():
    '''
    Class definition for a variogram 
    '''
    def __init__(
            self, 
            coordinates = None, 
            values = None, 
            dist_func = 'euclidean',
            bin_func = 'even',
            model = 'spherical',
            deramp = False,
            use_nugget = False, 
            maxlag = None,
            verbose = False
        ):

        self._X = np.array(coordinates)

        # Distance function can be a string or a function
        self._dist = None
        self.set_dist_function(dist_func)

        self._values = values
        
        # Model can be a function or a string
        self._model = None
        self.set_model(model)

        self._use_nugget = use_nugget
        self._deramp = deramp

        self.verbose = verbose
        self.cov = None
        self.cof = None

        # Experimental variogram parameters
        self._bin_func = None
        self._groups = None
        self._bins = None
        self.set_bin_func(bin_func=bin_func)

        self._lags = None
        self._n_lags = None
        self._maxlag = maxlag

        if coordinates is None and values is None:
            pass
        else:
            self.fit()

    @property
    def coordinates(self):
        """Coordinates of the Variogram instance""" 
        return self._X

    @property
    def values(self):
        """Values of the Variogram instance """ 
        return self._values

    @property
    def bin_func(self):
        """Binning function

        Returns an instance of the function used for binning the separating
        distances into the given amount of bins. Both functions use the same
        signature of func(distances, n, maxlag).

        The setter of this property utilizes the Variogram.set_bin_func to
        set a new function.

        Returns
        -------
        binning_function : function

        See Also
        --------
        Variogram.set_bin_func

        """
        return self._bin_func

    @bin_func.setter
    def bin_func(self, bin_func):
        self.set_bin_func(bin_func=bin_func)

    def set_bin_func(self, bin_func):
        """Set binning function

        Sets a new binning function to be used. The new binning method is set
        by a string identifying the new function to be used. Can be one of:
        ['even', 'uniform'].

        Parameters
        ----------
        bin_func : str
            Can be one of:

            * **'even'**: Use skgstat.binning.even_width_lags for using
              n_lags lags of equal width up to maxlag.
            * **'uniform'**: Use skgstat.binning.uniform_count_lags for using
              n_lags lags up to maxlag in which the pairwise differences
              follow a uniform distribution.

        Returns
        -------
        void

        See Also
        --------
        Variogram.bin_func
        skgstat.binning.uniform_count_lags
        skgstat.binning.even_width_lags

        """
        # switch the input
        if bin_func.lower() == 'even':
            self._bin_func = binning.even_width_lags
        elif bin_func.lower() == 'uniform':
            self._bin_func = binning.uniform_count_lags
        else:
            raise ValueError('%s binning method is not known' % bin_func)

        # reset groups and bins
        self._groups = None
        self._bins = None
        self.cof, self.cov = None, None


    @property
    def bins(self):
        if self._bins is None:
            self._bins = self.bin_func(self.distance, self.n_lags, self.maxlag)
        return self._bins.copy()

    @bins.setter
    def bins(self, bins):
        # set the new bins
        self._bins = bins

        # clean the groups as they are not valid anymore
        self._groups = None
        self.cov = None
        self.cof = None

    @property
    def n_lags(self):
        """Number of lag bins

        Pass the number of lag bins to be used on
        this Variogram instance. This will reset
        the grouping index and fitting parameters

        """
        return self._n_lags

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self.set_model(model_name=value)

    def set_model(self, model_name):
        """
        Set model as the new theoretical variogram function.

        """
        # reset the fitting
        self.cof, self.cov = None, None

        if isinstance(model_name, str):
            if model_name.lower() == 'spherical':
                self._model = models.spherical
            elif model_name.lower() == 'exponential':
                self._model = models.exponential
            elif model_name.lower() == 'gaussian':
                self._model = models.gaussian
            elif model_name.lower() == 'cubic':
                self._model = models.cubic
            elif model_name.lower() == 'stable':
                self._model = models.stable
            elif model_name.lower() == 'matern':
                self._model = models.matern
            else:
                raise ValueError(
                    'The theoretical Variogram function %s is not understood,'
                    'please provide the function' % model_name)
        else:
            self._model = model_name

    @property
    def use_nugget(self):
        return self._use_nugget

    @use_nugget.setter
    def use_nugget(self, nugget):
        if not isinstance(nugget, bool):
            raise ValueError('use_nugget has to be of type bool.')

        # set new value
        self._use_nugget = nugget

    @property
    def dist_function(self):
        return self._dist_func

    def _dist_func_wrapper(self, x):
        if callable(self._dist_func):
            return self._dist_func(x)
        else:
            return pdist(X=x, metric=self._dist_func)

    @dist_function.setter
    def dist_function(self, func):
        self.set_dist_function(func=func)

    def set_dist_function(self, func):
        """Set distance function

        Set the function used for distance calculation. func can either be a
        callable or a string. The ranked distance function is not implemented
        yet. strings will be forwarded to the scipy.spatial.distance.pdist
        function as the metric argument.
        If func is a callable, it has to return the upper triangle of the
        distance matrix as a flat array (Like the pdist function).

        Parameters
        ----------
        func : string, callable

        Returns
        -------
        numpy.array

        """
        # reset the distances and fitting
        self._dist = None
        self.cof, self.cov = None, None

        if isinstance(func, str):
            if func.lower() == 'rank':
                raise NotImplementedError
            else:
                # if not ranks, it has to be a scipy metric
                self._dist_func = func

        elif callable(func):
            self._dist_func = func
        else:
            raise ValueError('Input not supported. Pass a string or callable.')

    @property
    def distance(self):
        if self._dist is None:
            self._calc_distances()
        return self._dist

    @distance.setter
    def distance(self, dist_array):
        self._dist = dist_array

    @property
    def maxlag(self):
        return self._maxlag

    @maxlag.setter
    def maxlag(self, value):
        # reset fitting
        self.cof, self.cov = None, None

        # remove bins
        self._bins = None
        self._groups = None

        # set new maxlag
        if value is None:
            self._maxlag = None
        elif isinstance(value, str):
            if value == 'median':
                self._maxlag = np.median(self.distance)
            elif value == 'mean':
                self._maxlag = np.mean(self.distance)
        elif value < 1:
            self._maxlag = value * np.max(self.distance)
        else:
            self._maxlag = value

    def preprocessing(self, force=False):
        """Preprocessing function

        Prepares all input data for the fit and transform functions. Namely,
        the distances are calculated and the value differences. Then the
        binning is set up and bin edges are calculated. If any of the listed
        subsets are already prepared, their processing is skipped. This
        behaviour can be changed by the force parameter. This will cause a
        clean preprocessing.

        Parameters
        ----------
        force : bool
            If set to True, all preprocessing data sets will be deleted. Use
            it in case you need a clean preprocessing.

        Returns
        -------
        void

        """
        # call the _calc functions
        self.deramp()
        self._calc_distances(force=force)
        self._calc_diff(force=force)

    def fit(self, force=False, method=None, **kwargs):
        """Fit the variogram

        The fit function will fit the theoretical variogram function to the
        experimental. The preprocessed distance matrix, pairwise differences
        and binning will not be recalculated, if already done. This could be
        forced by setting the force parameter to true.

        In case you call fit function directly, with method or sigma,
        the parameters set on Variogram object instantiation will get
        overwritten. All other keyword arguments will be passed to
        scipy.optimize.curve_fit function.

        Parameters
        ----------
        force : bool
            If set to True, a clean preprocessing of the distance matrix,
            pairwise differences and the binning will be forced. Default is
            False.
        method : string
            A string identifying one of the implemented fitting procedures.
            Can be one of ['lm', 'trf']:

              * lm: Levenberg-Marquardt algorithms implemented in
                scipy.optimize.leastsq function.
              * trf: Trust Region Reflective algorithm implemented in
                scipy.optimize.least_squares(method='trf')

        sigma : string, array
            Uncertainty array for the bins. Has to have the same dimension as
            self.bins. Refer to Variogram.fit_sigma for more information.

        Returns
        -------
        void

        See Also
        --------
        scipy.optimize
        scipy.optimize.curve_fit
        scipy.optimize.leastsq
        scipy.optimize.least_squares

        """
        # TODO: the kwargs need to be preserved somehow
        self.cof = None
        self.cov = None

        # if force, force a clean preprocessing
        self.preprocessing(force=force)

        # load the data
        x = self.distance
        y = self.semivar

        # overwrite fit setting if new params are given
        if method is not None:
            self.fit_method = method

        # remove nans
        _x = x[~np.isnan(y)]
        _y = y[~np.isnan(y)]

        # Switch the method
        # Trust Region Reflective
        if self.fit_method == 'trf':
            bounds = (0, self.__get_fit_bounds(x, y))
            self.cof, self.cov = curve_fit(
                self._model,
                _x, _y,
                method='trf',
                sigma=self.fit_sigma,
                p0=bounds[1],
                bounds=bounds,
                **kwargs
            )

        # Levenberg-Marquardt
        elif self.fit_method == 'lm':
            self.cof, self.cov = curve_fit(
                self.model,
                _x, _y,
                method='lm',
                sigma=self.fit_sigma,
                **kwargs
            )

        else:
            raise ValueError("fit method has to be one of ['trf', 'lm']")

    def transform(self, x):
        """Transform

        Transform a given set of lag values to the theoretical variogram
        function using the actual fitting and preprocessing parameters in
        this Variogram instance

        Parameters
        ----------
        x : numpy.array
            Array of lag values to be used as model input for the fitted
            theoretical variogram model

        Returns
        -------
        numpy.array

        """
        self.preprocessing()

        # if instance is not fitted, fit it
        if self.cof is None:
            self.fit(force=True)

        # return the result
        return self.fitted_model(x)

    @property
    def deramp(self):
        if self._deramp:
            self._values = deramp(self._X[:,0], self._X[:,1], self._values)

    @property
    def fitted_model(self):
        """Fitted Model

        Returns a callable that takes a distance value and returns a
        semivariance. This model is fitted to the current Variogram
        parameters. The function will be interpreted at return time with the
        parameters hard-coded into the function code.

        Returns
        -------
        model : callable
            The current semivariance model fitted to the current Variogram
            model parameters.
        """
        if self.cof is None:
            self.fit(force=True)

        # get the pars
        cof = self.cof

        # get the function
        func = self._model

        code = """model = lambda x: func(x, %s)""" % \
               (', '.join([str(_) for _ in cof]))

        # run the code
        loc = dict(func=func)
        exec(code, loc, loc)
        model = loc['model']

        return model

    def _calc_distances(self, force=False):
        if self._dist is not None and not force:
            return

        # if self._X is of just one dimension, concat zeros.
        if self._X.ndim == 1:
            _x = np.vstack(zip(self._X, np.zeros(len(self._X))))
        else:
            _x = self._X
        # else calculate the distances
        self._dist = self._dist_func_wrapper(_x)

    @property
    def raw_vario(self):
        return self._raw_vario

    @property
    def _raw_vario(self):
        if self._diff is None:
            raw_vario(self, force=False)
        return self._diff


    def clone(self):
        """Deep copy of self

        Return a deep copy of self.

        Returns
        -------
        Variogram

        """
        return copy.deepcopy(self)

    def __get_fit_bounds(self, x, y):
        """
        Return the bounds for parameter space in fitting a variogram model.
        The bounds are depended on the Model that is used

        Returns
        -------
        list

        """
        mname = self._model.__name__

        # use range, sill and smoothness parameter
        if mname == 'matern':
            # a is max(x), C0 is max(y) s is limited to 20?
            bounds = [np.nanmax(x), np.nanmax(y), 20.]

        # use range, sill and shape parameter
        elif mname == 'stable':
            # a is max(x), C0 is max(y) s is limited to 2?
            bounds = [np.nanmax(x), np.nanmax(y), 2.]

        # use only sill
        elif mname == 'nugget':
            # a is max(x):
            bounds = [np.nanmax(x)]

        # use range and sill
        else:
            # a is max(x), C0 is max(y)
            bounds = [np.nanmax(x), np.nanmax(y)]

        # if use_nugget is True add the nugget
        if self.use_nugget:
            bounds.append(0.99)

        return bounds

    def predict(self, new_dist):
        """Theoretical variogram function

        Calculate the experimental variogram and apply the binning. On
        success, the variogram model will be fitted and applied to n lag
        values. Returns the lags and the calculated semi-variance values.
        If force is True, a clean preprocessing and fitting run will be
        executed.

        Parameters
        ----------
        new_dist : ndarray
            array of distances for which to calculate the covariance

        Returns
        -------
        variogram : tuple
            first element is the created lags array
            second element are the calculated semi-variance values

        """
        return self._model(new_dist, *self.cof)

    @property
    def parameters(self):
        return self._cof


    def plot(self, axes=None, grid=True, show=True, hist=True):
        """Variogram Plot

        Plot the experimental variogram, the fitted theoretical function and
        an histogram for the lag classes. The axes attribute can be used to
        pass a list of AxesSubplots or a single instance to the plot
        function. Then these Subplots will be used. If only a single instance
        is passed, the hist attribute will be ignored as only the variogram
        will be plotted anyway.

        Parameters
        ----------
        axes : list, tuple, array, AxesSubplot or None
            If None, the plot function will create a new matplotlib figure.
            Otherwise a single instance or a list of AxesSubplots can be
            passed to be used. If a single instance is passed, the hist
            attribute will be ignored.
        grid : bool
            Defaults to True. If True a custom grid will be drawn through
            the lag class centers
        show : bool
            Defaults to True. If True, the show method of the passed or
            created matplotlib Figure will be called before returning the
            Figure. This should be set to False, when used in a Notebook,
            as a returned Figure object will be plotted anyway.
        hist : bool
            Defaults to True. If False, the creation of a histogram for the
            lag classes will be suppressed.

        Returns
        -------
        matplotlib.Figure

        """
        # get the parameters
        _bins = self.bins
        _exp = self.experimental
        x = np.linspace(0, np.nanmax(_bins), 100)  # make the 100 a param?
                # do the plotting
        if axes is None:
            if hist:
                fig = plt.figure(figsize=(8, 5))
                ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
                ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)
                fig.subplots_adjust(hspace=0)
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
                ax2 = None
        elif isinstance(axes, (list, tuple, np.ndarray)):
            ax1, ax2 = axes
            fig = ax1.get_figure()
        else:
            ax1 = axes
            ax2 = None
            fig = ax1.get_figure()

        # apply the model
        y = self.transform(x)

        # handle the relative experimental variogram
        if self.normalized:
            _bins /= np.nanmax(_bins)
            y /= np.max(_exp)
            _exp /= np.nanmax(_exp)
            x /= np.nanmax(x)

        # ------------------------
        # plot Variograms
        ax1.plot(_bins, _exp, '.b')
        ax1.plot(x, y, '-g')

        # ax limits
        if self.normalized:
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])
        if grid:
            ax1.grid(False)
            ax1.vlines(_bins, *ax1.axes.get_ybound(), colors=(.85, .85, .85),
                       linestyles='dashed')

        ax1.axes.set_xlabel('Lag (-)')

        # ------------------------
        # plot histogram
        if ax2 is not None and hist:
            # calc the histogram
            _count = np.fromiter(
                (g.size for g in self.lag_classes()), dtype=int
            )

            # set the sum of hist bar widths to 70% of the x-axis space
            w = (np.max(_bins) * 0.7) / len(_count)

            # plot
            ax2.bar(_bins, _count, width=w, align='center', color='red')

            # adjust
            plt.setp(ax2.axes.get_xticklabels(), visible=False)
            ax2.axes.set_yticks(ax2.axes.get_yticks()[1:])

            # need a grid?
            if grid:  #pragma: no cover
                ax2.grid(False)
                ax2.vlines(_bins, *ax2.axes.get_ybound(),
                           colors=(.85, .85, .85), linestyles='dashed')

            # anotate
            ax2.axes.set_ylabel('N')

        # show the figure
        if show:  # pragma: no cover
            fig.show()

        return fig




class VariogramAnalysis():
    '''
        Class which ingests dataframe output from 'RaiderStats' class and performs variogram analysis.
    '''

    def __init__(self, filearg, gridpoints, col_name, unit='m', workdir='./', seasonalinterval=None, densitythreshold=10, binnedvariogram=False, numCPUs=8, variogram_per_timeslice=False):
        self.df = filearg
        self.col_name = col_name
        self.unit = unit
        self.gridpoints = gridpoints
        self.workdir = workdir
        self.seasonalinterval = seasonalinterval
        self.densitythreshold = densitythreshold
        self.binnedvariogram = binnedvariogram
        self.numCPUs = numCPUs
        self.variogram_per_timeslice = variogram_per_timeslice



    def _emp_vario(self, x, y, data, Nsamp=1000):
        '''
        Compute empirical semivariance
        '''
        # remove NaNs if possible
        mask = ~np.isnan(data)
        if False in mask:
            data = data[mask]
            x = x[mask]
            y = y[mask]

        # deramp
        data = deramp(data)

        samples, indpars = self._get_samples(data, Nsamp)
        x, y = self._get_XY(x, y, indpars)
        dists = self._get_distances(
            np.array([[x[:, 0], y[:, 0]], [x[:, 1], y[:, 1]]]).T)
        vario = self._get_semivariance(samples[:, 0], samples[:, 1])

        return dists, vario

    def _get_XY(self, x2d, y2d, indpars):
        '''
        Given a list of indices, return the x,y locations
        from two matrices
        '''
        x = np.array([[x2d[r[0]], x2d[r[1]]] for r in indpars])
        y = np.array([[y2d[r[0]], y2d[r[1]]] for r in indpars])
        return x, y

    def _get_distances(self, XY, method = 'cdist'):
        '''
        Return the distances between each point in a list of points
        '''
        from scipy.spatial.distance import cdist
        return np.diag(cdist(XY[:, :, 0], XY[:, :, 1], metric='euclidean'))

    def _get_samples(self, data, Nsamp=1000):
        '''
        pull samples from a 2D image for variogram analysis
        '''
        if len(data) < self.densitythreshold:
            logger.warning('Less than {} points for this gridcell', self.densitythreshold)
            logger.info('Will pass empty list')
            d = []
            indpars = []
        else:
            d, indpars = sample(data, Nsamp = Nsamp)


    def _get_semivariance(self, XY, xy=None):
        '''
        Return variograms
        '''
        return 0.5 * np.square(XY - xy)  # XY = 1st col xy= 2nd col

    def _binned_vario(self, hEff, rawVario, xBin=None):
        '''
        return a binned empirical variogram
        '''
        if xBin is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                xBin = np.linspace(0, np.nanmax(hEff) * .67, 20)

        nBins = len(xBin) - 1
        hExp, expVario = [], []

        for iBin in range(nBins):
            iBinMask = np.logical_and(xBin[iBin] < hEff, hEff <= xBin[iBin + 1])
            # circumvent indexing
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    hExp.append(np.nanmean(hEff[iBinMask]))
                    expVario.append(np.nanmean(rawVario[iBinMask]))
            except:
                pass

        if False in ~np.isnan(hExp):
            # NaNs present in binned histogram
            hExp = [x for x in hExp if str(x) != 'nan']
            expVario = [x for x in expVario if str(x) != 'nan']

        return np.array(hExp), np.array(expVario)

    def _fit_vario(self, dists, vario, model=None, x0=None, Nparm=None, ub=None):
        '''
        Fit a variogram model to data
        '''
        from scipy.optimize import least_squares

        def resid(x, d, v, m):
            return (m(x, d) - v)

        if ub is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                ub = np.array([np.nanmax(dists) * 0.8, np.nanmax(vario)
                               * 0.8, np.nanmax(vario) * 0.8])

        if x0 is None and Nparm is None:
            raise RuntimeError(
                'Must specify either x0 or the number of model parameters')
        if x0 is not None:
            lb = np.zeros(len(x0))
        if Nparm is not None:
            lb = np.zeros(Nparm)
            x0 = (ub - lb) / 2
        bounds = (lb, ub)

        mask = np.isnan(dists) | np.isnan(vario)
        d = dists[~mask].copy()
        v = vario[~mask].copy()

        res_robust = least_squares(resid, x0, bounds=bounds,
                                   loss='soft_l1', f_scale=0.1,
                                   args=(d, v, model))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            d_test = np.linspace(0, np.nanmax(dists), 100)
        # v_test is my y., # res_robust.x =a, b, c, where a = range, b = sill, and c = nugget model, d_test=x
        v_test = model(res_robust.x, d_test)

        return res_robust, d_test, v_test

    # this would be expontential plus nugget
    def __exponential__(self, parms, h):
        '''
        returns a variogram model given a set of arguments and
        key-word arguments
        '''
        # a = range, b = sill, c = nugget model
        a, b, c = parms
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in true_divide")
            return b * (1 - np.exp(-h / a)) + c

    # this would be gaussian plus nugget
    def __gaussian__(self, parms, h):
        '''
        returns a Gaussian variogram model
        '''
        a, b, c = parms
        return b * (1 - np.exp(-np.square(h) / (a**2))) + c

    def _append_variogram(self, grid_ind, grid_subset):
        '''
        For a given grid-cell, iterate through time slices to generate/append empirical variogram(s)
        '''
        # Comprehensive arrays recording data across all time epochs for given station
        dists_arr = []
        vario_arr = []
        dists_binned_arr = []
        vario_binned_arr = []
        res_robust_arr = []
        d_test_arr = []
        v_test_arr = []
        for j in sorted(list(set(grid_subset['Date']))):
            # If insufficient sample size, skip slice and record occurence
            if len(np.array(grid_subset[grid_subset['Date'] == j][self.col_name])) < self.densitythreshold:
                # Record skipped [gridnode, timeslice]
                self.skipped_slices.append([grid_ind, j.strftime("%Y-%m-%d")])
            else:
                self.gridcenterlist.append(['grid{} '.format(
                    grid_ind) + 'Lat:{} Lon:{}'.format(
                    str(self.gridpoints[grid_ind][1]), str(self.gridpoints[grid_ind][0]))])
                lonarr = np.array(
                    grid_subset[grid_subset['Date'] == j]['Lon'])
                latarr = np.array(
                    grid_subset[grid_subset['Date'] == j]['Lat'])
                delayarray = np.array(
                    grid_subset[grid_subset['Date'] == j][self.col_name])
                # fit empirical variogram for each time AND grid
                dists, vario = self._emp_vario(lonarr, latarr, delayarray)
                dists_binned, vario_binned = self._binned_vario(
                    dists, vario)
                # fit experimental variogram for each time AND grid, model default is exponential
                res_robust, d_test, v_test = self._fit_vario(
                    dists_binned, vario_binned, model=self.__exponential__, x0=None, Nparm=3)
                # Plot empirical + experimental variogram for this gridnode and timeslice
                if not os.path.exists(os.path.join(self.workdir, 'variograms/grid{}'.format(grid_ind))):
                    os.makedirs(os.path.join(
                        self.workdir, 'variograms/grid{}'.format(grid_ind)))
                # Make variogram plots for each time-slice
                if self.variogram_per_timeslice:
                    # Plot empirical variogram for this gridnode and timeslice
                    self.plot_variogram(grid_ind, j.strftime("%Y%m%d"), [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                                        workdir=os.path.join(self.workdir, 'variograms/grid{}'.format(grid_ind)), dists=dists, vario=vario,
                                        dists_binned=dists_binned, vario_binned=vario_binned)
                    # Plot experimental variogram for this gridnode and timeslice
                    self.plot_variogram(grid_ind, j.strftime("%Y%m%d"), [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                                        workdir=os.path.join(self.workdir, 'variograms/grid{}'.format(grid_ind)), d_test=d_test, v_test=v_test,
                                        res_robust=res_robust.x, dists_binned=dists_binned, vario_binned=vario_binned)
                # append for plotting
                self.good_slices.append([grid_ind, j.strftime("%Y%m%d")])
                dists_arr.append(dists)
                vario_arr.append(vario)
                dists_binned_arr.append(dists_binned)
                vario_binned_arr.append(vario_binned)
                res_robust_arr.append(res_robust.x)
                d_test_arr.append(d_test)
                v_test_arr.append(v_test)
        # fit experimental variogram for each grid
        if dists_binned_arr != []:
            # TODO: need to change this from accumulating binned data to raw data
            dists_arr = np.concatenate(dists_arr).ravel()
            vario_arr = np.concatenate(vario_arr).ravel()
            # if specified, passed binned empirical variograms
            if self.binnedvariogram:
                dists_binned_arr = np.concatenate(dists_binned_arr).ravel()
                vario_binned_arr = np.concatenate(vario_binned_arr).ravel()
            else:
                #dists_binned_arr = dists_arr ; vario_binned_arr = vario_arr
                dists_binned_arr, vario_binned_arr = self._binned_vario(
                    dists_arr, vario_arr)
            TOT_res_robust, TOT_d_test, TOT_v_test = self._fit_vario(
                dists_binned_arr, vario_binned_arr, model=self.__exponential__, x0=None, Nparm=3)
            tot_timetag = self.good_slices[0][1] + '-' + self.good_slices[-1][1]
            # Append TOT arrays
            self.TOT_good_slices.append([grid_ind, tot_timetag])
            self.TOT_res_robust_arr.append(TOT_res_robust.x)
            self.TOT_tot_timetag.append(tot_timetag)
            # Plot empirical variogram for this gridnode
            self.plot_variogram(grid_ind, tot_timetag, [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                                workdir=os.path.join(self.workdir, 'variograms/grid{}'.format(grid_ind)), dists=dists_arr, vario=vario_arr,
                                dists_binned=dists_binned_arr, vario_binned=vario_binned_arr, seasonalinterval=self.seasonalinterval)
            # Plot experimental variogram for this gridnode
            self.plot_variogram(grid_ind, tot_timetag, [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                                workdir=os.path.join(self.workdir, 'variograms/grid{}'.format(grid_ind)), d_test=TOT_d_test, v_test=TOT_v_test,
                                res_robust=TOT_res_robust.x, seasonalinterval=self.seasonalinterval, dists_binned=dists_binned_arr, vario_binned=vario_binned_arr)
        # Record sparse grids which didn't have sufficient sample size of data through any of the timeslices
        else:
            self.sparse_grids.append(grid_ind)

        return self.TOT_good_slices, self.TOT_res_robust_arr, self.gridcenterlist

    def create_variograms(self):
        '''
        Iterate through grid-cells and time slices to generate empirical variogram(s)
        '''
        # track data for plotting
        self.TOT_good_slices = []
        self.TOT_res_robust_arr = []
        self.TOT_tot_timetag = []
        # track pass/rejected grids
        self.sparse_grids = []
        self.good_slices = []
        self.skipped_slices = []
        # record grid-centers for lookup-table
        self.gridcenterlist = []
        args = []
        for i in sorted(list(set(self.df['gridnode']))):
            # pass subset of all stations corresponding to given grid-cell
            grid_subset = self.df[self.df['gridnode'] == i]
            args.append((i, grid_subset))
        # Parallelize iteration through all grid-cells and time slices
        with multiprocessing.Pool(self.numCPUs) as multipool:
            for i, j, k in multipool.starmap(self._append_variogram, args):
                self.TOT_good_slices.extend(i)
                self.TOT_res_robust_arr.extend(j)
                self.gridcenterlist.extend(k)

        # save grid-center lookup table
        self.gridcenterlist = [list(i) for i in set(tuple(j)
                                                    for j in self.gridcenterlist)]
        self.gridcenterlist.sort(key=lambda x: int(x[0][4:6]))
        gridcenter = open(
            (os.path.join(self.workdir, 'variograms/gridlocation_lookup.txt')), "w")
        for element in self.gridcenterlist:
            gridcenter.writelines("\n".join(element))
            gridcenter.write("\n")
        gridcenter.close()

        TOT_grids = [i[0] for i in self.TOT_good_slices]

        return TOT_grids, self.TOT_res_robust_arr

    def plot_variogram(self, gridID, timeslice, coords, workdir='./', d_test=None, v_test=None, res_robust=None, dists=None, vario=None, dists_binned=None, vario_binned=None, seasonalinterval=None):
        '''
        Make empirical and/or experimental variogram fit plots
        '''
        # If specified workdir doesn't exist, create it
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        # make plot title
        title_str = ' \nLat:{:.2f} Lon:{:.2f}\nTime:{}'.format(
            coords[1], coords[0], str(timeslice))
        if seasonalinterval:
            title_str += ' Season(mm/dd): {}/{} – {}/{}'.format(int(timeslice[4:6]), int(
                timeslice[6:8]), int(timeslice[-4:-2]), int(timeslice[-2:]))

        if dists is not None and vario is not None:
            plt.scatter(dists, vario, s=1, facecolor='0.5', label='raw')
        if dists_binned is not None and vario_binned is not None:
            plt.plot(dists_binned, vario_binned, 'bo', label='binned')
        if res_robust is not None:
            plt.axhline(y=res_robust[1], color='g',
                        linestyle='--', label='ɣ\u0332\u00b2({}\u00b2)'.format(self.unit))
            plt.axvline(x=res_robust[0], color='c',
                        linestyle='--', label='h\u0332(°)')
        if d_test is not None and v_test is not None:
            plt.plot(d_test, v_test, 'r-', label='experimental fit')
        plt.xlabel('Distance (°)')
        plt.ylabel('Dissimilarity ({}\u00b2)'.format(self.unit))
        plt.legend(bbox_to_anchor=(1.02, 1),
                   loc='upper left', borderaxespad=0., framealpha=1.)
        # Plot empirical variogram
        if d_test is None and v_test is None:
            plt.title('Empirical variogram' + title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(
                workdir, 'grid{}_timeslice{}_justEMPvariogram.eps'.format(gridID, timeslice)))
        # Plot just experimental variogram
        else:
            plt.title('Experimental variogram' + title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(
                workdir, 'grid{}_timeslice{}_justEXPvariogram.eps'.format(gridID, timeslice)))
        plt.close()

        return


def sample(data, Nsamp):
    ''' Sampling a raster '''

    # Create a set of random indices
    indpars = list(itertools.combinations(range(len(data)), 2))
    random.shuffle(indpars)

    # subsample
    Nvalidsamp = int(len(data) * (len(data) - 1) / 2)

    # Only downsample if Nsamps>specified value
    if Nvalidsamp > Nsamp:
        indpars = indpars[:Nsamp]

    d = np.array([[data[r[0]], data[r[1]]] for r in indpars])

    return d, indpars

def deramp(x, y, data):
    ''' Deramp data '''
    A = np.array([x, y, np.ones(len(x))]).T
    ramp = np.linalg.lstsq(A, data.T, rcond=None)[0]
    data = data - (np.matmul(A, ramp))
    return data


#TODO: implement raw semivariance calculation
def raw_vario(v, deramp = False):
    ''' Compute the raw semivariance '''
    l = len(v)
    self._diff = np.zeros(int((l**2 - l) / 2))

    # calculate the pairwise differences
    k = 0
    for i in range(l):
        for j in range(i+1, l):
             self._diff[k] = np.abs(v[i] - v[j])
             k += 1

def semivariance(x1, x2):
    return 0.5 * np.square(x1 - x2)

