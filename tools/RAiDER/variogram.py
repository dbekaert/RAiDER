#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
import argparse
import copy
import datetime as dt
import itertools
import multiprocessing
import os
import warnings
import random

from RAiDER.logger import *
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon
from pandas.plotting import register_matplotlib_converters
from matplotlib import pyplot as plt

import matplotlib as mpl
# must switch to Agg to avoid multiprocessing crashes
mpl.use('Agg')



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
            tot_timetag = self.good_slices[0][1] + '–' + self.good_slices[-1][1]
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

