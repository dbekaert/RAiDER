# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import argparse
import copy
import datetime as dt
import itertools
import multiprocessing as mp
import os
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from rasterio.transform import Affine
from scipy import optimize
from scipy.optimize import OptimizeWarning
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from RAiDER.cli.parser import add_cpus
from RAiDER.logger import logger, logging
from RAiDER.utilFcns import WGS84_to_UTM


# must switch to Agg to avoid multiprocessing crashes
mpl.use('Agg')


def create_parser():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Perform basic statistical analyses concerning the spatiotemporal distribution of zenith delays.

Specifically, make any of the following specified plot(s):
scatterplot of station locations, total empirical and experimental variogram fits for data in each grid cell
(and for each valid time-slice if -variogram_per_timeslice specified), and gridded heatmaps of data, station distribution,
range and sill values associated with experimental variogram fits. The default is to generate all of these.

Example call to plot gridded station mean delay in a specific time interval :
raiderStats.py -f <filename> -grid_delay_mean -ti '2016-01-01 2018-01-01'

Example call to plot gridded station mean delay in a specific time interval with superimposed gridlines and station scatterplots :
raiderStats.py -f <filename> -grid_delay_mean -ti '2016-01-01 2018-01-01' --drawgridlines --stationsongrids

Example call to plot gridded station variogram in a specific time interval and through explicitly the summer seasons:
raiderStats.py -f <filename> -grid_delay_mean -ti '2016-01-01 2018-01-01' --seasonalinterval '06-21 09-21' -variogramplot
""",
    )

    # User inputs
    userinps = parser.add_argument_group('User inputs/options for which especially careful review is recommended')
    userinps.add_argument(
        '-f',
        '--file',
        dest='fname',
        type=str,
        required=True,
        help='Final output file generated from downloadGNSSDelays.py which contains GPS zenith delays for a specified time period and spatial footprint. ',
    )
    userinps.add_argument(
        '-c',
        '--column_name',
        dest='col_name',
        type=str,
        default='ZTD',
        help='Name of the input column to plot. Input assumed to be in units of meters',
    )
    userinps.add_argument(
        '-u',
        '--unit',
        dest='unit',
        type=str,
        default='m',
        help='Specified output unit (as distance or time), by default m. Input unit assumed to be m following convention in downloadGNSSDelays.py. Refer to "convert_SI" for supported units. Note if you specify time unit here, you must specify input for "--obs_errlimit" to be in units of m',
    )
    userinps.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        default='./',
        help='Specify directory to deposit all outputs. Default is local directory where script is launched.',
    )
    add_cpus(userinps)
    userinps.add_argument(
        '-verbose',
        '--verbose',
        action='store_true',
        dest='verbose',
        help='Run in verbose (debug) mode. Default False'
    )

    # Spatiotemporal subset options
    dtsubsets = parser.add_argument_group('Controls for spatiotemporal subsetting.')
    dtsubsets.add_argument(
        '-b',
        '--bounding_box',
        dest='bounding_box',
        type=str,
        default=None,
        help="Provide either valid shapefile or Lat/Lon Bounding SNWE. -- Example : '19 20 -99.5 -98.5'",
    )
    dtsubsets.add_argument(
        '-sp',
        '--spacing',
        dest='spacing',
        type=float,
        default='1',
        help='Specify spacing of grid-cells for statistical analyses. By default 1 deg.',
    )
    dtsubsets.add_argument(
        '-ti',
        '--timeinterval',
        dest='timeinterval',
        type=str,
        default=None,
        help="Subset in time by specifying earliest YYYY-MM-DD date followed by latest date YYYY-MM-DD. -- Example : '2016-01-01 2019-01-01'.",
    )
    dtsubsets.add_argument(
        '-si',
        '--seasonalinterval',
        dest='seasonalinterval',
        type=str,
        default=None,
        help="Subset in by an specific interval for each year by specifying earliest MM-DD time followed by latest MM-DD time. -- Example : '03-21 06-21'.",
    )
    dtsubsets.add_argument(
        '-oe',
        '--obs_errlimit',
        dest='obs_errlimit',
        type=float,
        default='inf',
        help='Observation error threshold to discard observations with large uncertainties.',
    )

    # Plot formatting/options
    pltformat = parser.add_argument_group('Optional controls for plot formatting/options.')
    pltformat.add_argument(
        '-figdpi',
        '--figdpi',
        dest='figdpi',
        type=int,
        default=100,
        help='DPI to use for saving figures'
    )
    pltformat.add_argument(
        '-title',
        '--user_title',
        dest='user_title',
        type=str,
        default=None,
        help='Specify custom title for plots.'
    )
    pltformat.add_argument(
        '-fmt',
        '--plot_format',
        dest='plot_fmt',
        type=str,
        default='png',
        help='Plot format to use for saving figures'
    )
    pltformat.add_argument(
        '-cb',
        '--color_bounds',
        dest='cbounds',
        type=str,
        default=None,
        help='List of two floats to use as color axis bounds',
    )
    pltformat.add_argument(
        '-cp',
        '--colorpercentile',
        dest='colorpercentile',
        type=float,
        default=None,
        nargs=2,
        help='Set low and upper percentile for plot colorbars. By default 25%% and 95%%, respectively.',
    )
    pltformat.add_argument(
        '-cm', '--colormap', dest='usr_colormap', type=str, default='hot_r', help='Specify matplotlib colorbar.'
    )
    pltformat.add_argument(
        '-dt',
        '--densitythreshold',
        dest='densitythreshold',
        type=int,
        default='10',
        help='For variogram plots, given grid-cell is only valid if it contains this specified threshold of stations. By default 10 stations.',
    )
    pltformat.add_argument(
        '-sg',
        '--stationsongrids',
        dest='stationsongrids',
        action='store_true',
        help='In gridded plots, superimpose your gridded array with a scatterplot of station locations.',
    )
    pltformat.add_argument(
        '-dg', '--drawgridlines', dest='drawgridlines', action='store_true', help='Draw gridlines on gridded plots.'
    )
    pltformat.add_argument(
        '-tl',
        '--time_lines',
        dest='time_lines',
        action='store_true',
        help='Draw central longitudinal lines with respect to datetime. Most useful for local-time analyses.',
    )
    pltformat.add_argument(
        '-plotall',
        '--plotall',
        action='store_true',
        dest='plotall',
        help='Generate all supported plots, including variogram plots.',
    )
    pltformat.add_argument(
        '-min_span',
        '--min_span',
        dest='min_span',
        type=float,
        default=[2, 0.6],
        nargs=2,
        help='Minimum TS span (years) and minimum fractional observations in span (fraction) imposed for seasonal amplitude/phase analyses to be performed for a given station.',
    )
    pltformat.add_argument(
        '-period_limit',
        '--period_limit',
        dest='period_limit',
        type=float,
        default=0.0,
        help='period limit (years) imposed for seasonal amplitude/phase analyses to be performed for a given station.',
    )

    # All plot types
    # Station scatter-plots
    pltscatter = parser.add_argument_group('Supported types of individual station scatter-plots.')
    pltscatter.add_argument(
        '-station_distribution',
        '--station_distribution',
        action='store_true',
        dest='station_distribution',
        help='Plot station distribution.',
    )
    pltscatter.add_argument(
        '-station_delay_mean',
        '--station_delay_mean',
        action='store_true',
        dest='station_delay_mean',
        help='Plot station mean delay.',
    )
    pltscatter.add_argument(
        '-station_delay_median',
        '--station_delay_median',
        action='store_true',
        dest='station_delay_median',
        help='Plot station median delay.',
    )
    pltscatter.add_argument(
        '-station_delay_stdev',
        '--station_delay_stdev',
        action='store_true',
        dest='station_delay_stdev',
        help='Plot station delay stdev.',
    )
    pltscatter.add_argument(
        '-station_seasonal_phase',
        '--station_seasonal_phase',
        action='store_true',
        dest='station_seasonal_phase',
        help='Plot station delay phase/amplitude.',
    )
    pltscatter.add_argument(
        '-phaseamp_per_station',
        '--phaseamp_per_station',
        action='store_true',
        dest='phaseamp_per_station',
        help='Save debug figures of curve-fit vs data per station.',
    )

    # Gridded plots
    pltgrids = parser.add_argument_group('Supported types of gridded plots.')
    pltgrids.add_argument(
        '-grid_heatmap',
        '--grid_heatmap',
        action='store_true',
        dest='grid_heatmap',
        help='Plot gridded station heatmap.',
    )
    pltgrids.add_argument(
        '-grid_delay_mean',
        '--grid_delay_mean',
        action='store_true',
        dest='grid_delay_mean',
        help='Plot gridded station-wise mean delay.',
    )
    pltgrids.add_argument(
        '-grid_delay_median',
        '--grid_delay_median',
        action='store_true',
        dest='grid_delay_median',
        help='Plot gridded station-wise median delay.',
    )
    pltgrids.add_argument(
        '-grid_delay_stdev',
        '--grid_delay_stdev',
        action='store_true',
        dest='grid_delay_stdev',
        help='Plot gridded station-wise delay stdev.',
    )
    pltgrids.add_argument(
        '-grid_seasonal_phase',
        '--grid_seasonal_phase',
        action='store_true',
        dest='grid_seasonal_phase',
        help='Plot gridded station-wise delay phase/amplitude.',
    )
    pltgrids.add_argument(
        '-grid_delay_absolute_mean',
        '--grid_delay_absolute_mean',
        action='store_true',
        dest='grid_delay_absolute_mean',
        help='Plot absolute gridded station mean delay.',
    )
    pltgrids.add_argument(
        '-grid_delay_absolute_median',
        '--grid_delay_absolute_median',
        action='store_true',
        dest='grid_delay_absolute_median',
        help='Plot absolute gridded station median delay.',
    )
    pltgrids.add_argument(
        '-grid_delay_absolute_stdev',
        '--grid_delay_absolute_stdev',
        action='store_true',
        dest='grid_delay_absolute_stdev',
        help='Plot absolute gridded station delay stdev.',
    )
    pltgrids.add_argument(
        '-grid_seasonal_absolute_phase',
        '--grid_seasonal_absolute_phase',
        action='store_true',
        dest='grid_seasonal_absolute_phase',
        help='Plot absolute gridded station delay phase/amplitude.',
    )
    pltgrids.add_argument(
        '-grid_to_raster',
        '--grid_to_raster',
        action='store_true',
        dest='grid_to_raster',
        help='Save gridded array as raster. May directly load/plot in successive script call.',
    )

    # Variogram plots
    pltvario = parser.add_argument_group('Supported types of variogram plots.')
    pltvario.add_argument(
        '-variogramplot',
        '--variogramplot',
        action='store_true',
        dest='variogramplot',
        help='Plot gridded station variogram.',
    )
    pltvario.add_argument(
        '-binnedvariogram',
        '--binnedvariogram',
        action='store_true',
        dest='binnedvariogram',
        help='Apply experimental variogram fit to total binned empirical variograms for each time slice. Default is to pass total unbinned empiricial variogram.',
    )
    pltvario.add_argument(
        '-variogram_per_timeslice',
        '--variogram_per_timeslice',
        action='store_true',
        dest='variogram_per_timeslice',
        help='Generate variogram plots per gridded station AND time-slice.',
    )
    pltvario.add_argument(
        '-variogram_errlimit',
        '--variogram_errlimit',
        dest='variogram_errlimit',
        type=float,
        default='inf',
        help='Variogram RMSE threshold to discard grid-cells with large uncertainties.',
    )

    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    return parser.parse_args(args=iargs)


def convert_SI(val, unit_in, unit_out):
    """Convert input to desired units."""
    SI = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.0, 'mm^2': 1e-6, 'cm^2': 1e-4, 'm^2': 1.0, 'km^2': 1e6}

    # avoid conversion if output unit in time
    if unit_out in ['minute', 'hour', 'day', 'year']:
        # adjust if input isn't datetime, and assume it to be part of workflow
        # e.g. sigZTD filter, already extracted datetime object
        try:
            datetime = val.apply(pd.to_datetime).dt
            return getattr(datetime, unit_out).astype(float).astype("Int32")
        except AttributeError:
            return val

    # check if output spatial unit is supported
    if unit_out not in SI:
        raise ValueError(f'User-specified output unit {unit_out} not recognized.')

    return val * SI[unit_in] / SI[unit_out]


def midpoint(p1, p2):
    """Calculate central longitude for '--time_lines' option."""
    import math

    if p1[1] == p2[1]:
        return p1[1]

    lat1, lon1, lat2, lon2 = map(math.radians, (p1[0], p1[1], p2[0], p2[1]))
    dlon = lon2 - lon1
    dx = math.cos(lat2) * math.cos(dlon)
    dy = math.cos(lat2) * math.sin(dlon)
    lon3 = lon1 + math.atan2(dy, math.cos(lat1) + dx)

    return int(math.degrees(lon3))


def save_gridfile(
    df,
    gridfile_type,
    fname,
    plotbbox,
    spacing,
    unit,
    colorbarfmt='%.2f',
    stationsongrids=False,
    time_lines=False,
    dtype='float32',
    noData=np.nan,
):
    """Function to save gridded-arrays as GDAL-readable file."""
    # Pass metadata
    metadata_dict = {}
    metadata_dict['gridfile_type'] = gridfile_type
    metadata_dict['plotbbox'] = ' '.join([str(i) for i in plotbbox])
    metadata_dict['spacing'] = str(spacing)
    metadata_dict['unit'] = unit
    if unit in ['minute', 'hour', 'day', 'year']:
        colorbarfmt = '%1i'
    metadata_dict['colorbarfmt'] = colorbarfmt

    if stationsongrids:
        metadata_dict['stationsongrids'] = ' '.join([str(i) for i in stationsongrids])
    else:
        metadata_dict['stationsongrids'] = 'False'

    if time_lines:
        metadata_dict['time_lines'] = ' '.join([str(i) for i in time_lines])
    else:
        metadata_dict['time_lines'] = 'False'

    # Write data to file
    transform = Affine(spacing, 0.0, plotbbox[0], 0.0, -1 * spacing, plotbbox[-1])
    with rasterio.open(
        fname,
        mode='w',
        count=1,
        width=df.shape[1],
        height=df.shape[0],
        dtype=dtype,
        nodata=noData,
        crs='+proj=latlong',
        transform=transform,
        driver='GTiff',
    ) as dst:
        dst.update_tags(0, **metadata_dict)
        dst.write(df, 1)

    return metadata_dict


def load_gridfile(fname, unit):
    """Function to load gridded-arrays saved from previous runs."""
    try:
        with rasterio.open(fname) as src:
            grid_array = src.read(1).astype(float)
    except TypeError:
        raise ValueError('fname is not a valid file')

    # Read metadata variables needed for plotting
    metadata_dict = src.tags()

    # Initiate no-data array to mask data
    nodat_arr = [0, np.nan, np.inf]
    if unit in ['minute', 'hour', 'day', 'year']:
        nodat_arr = [np.nan, np.inf]
    # set masked values as nans
    for i in nodat_arr:
        grid_array = np.ma.masked_where(grid_array == i, grid_array)
    grid_array = np.ma.filled(grid_array, np.nan)

    # Make plotting command a global variable
    print('metadata_dict', metadata_dict)
    gridfile_type = metadata_dict['gridfile_type']
    globals()[gridfile_type] = True

    plotbbox = [float(i) for i in metadata_dict['plotbbox'].split()]
    spacing = float(metadata_dict['spacing'])
    colorbarfmt = metadata_dict['colorbarfmt']
    inputunit = metadata_dict['unit']
    # adjust conversion if native units are squared
    if '^2' in inputunit:
        unit = unit.split('^2')[0] + '^2'
    # convert to specified output unit
    grid_array = convert_SI(grid_array, inputunit, unit)

    # Backwards compatible for cases where this key doesn't exist
    try:
        time_lines = metadata_dict['time_lines']
    except KeyError:
        time_lines = False

    if metadata_dict['stationsongrids'] == 'False':
        stationsongrids = False
    else:
        stationsongrids = [float(i) for i in metadata_dict['stationsongrids'.split()]]

    if metadata_dict['time_lines'] == 'False':
        time_lines = False
    else:
        time_lines = [float(i) for i in metadata_dict['time_lines'].split()]

    return grid_array, plotbbox, spacing, colorbarfmt, stationsongrids, time_lines


class VariogramAnalysis:
    """Class which ingests dataframe output from 'RaiderStats' class and performs variogram analysis."""

    def __init__(
        self,
        filearg,
        gridpoints,
        col_name,
        unit='m',
        workdir='./',
        seasonalinterval=None,
        densitythreshold=10,
        binnedvariogram=False,
        numCPUs=8,
        variogram_per_timeslice=False,
        variogram_errlimit='inf',
    ) -> None:
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
        self.variogram_errlimit = float(variogram_errlimit)

    def _get_samples(self, data, Nsamp=1000):
        """Pull samples from a 2D image for variogram analysis."""
        import random

        if len(data) < self.densitythreshold:
            logger.warning('Less than {} points for this gridcell', self.densitythreshold)
            logger.info('Will pass empty list')
            d = []
            indpars = []
        else:
            indpars = list(itertools.combinations(range(len(data)), 2))
            random.shuffle(indpars)
            # subsample
            Nvalidsamp = int(len(data) * (len(data) - 1) / 2)
            # Only downsample if Nsamps>specified value
            if Nvalidsamp > Nsamp:
                indpars = indpars[:Nsamp]
            d = np.array([[data[r[0]], data[r[1]]] for r in indpars])

        return d, indpars

    def _get_XY(self, x2d, y2d, indpars):
        """Given a list of indices, return the x,y locations from two matrices."""
        x = np.array([[x2d[r[0]], x2d[r[1]]] for r in indpars])
        y = np.array([[y2d[r[0]], y2d[r[1]]] for r in indpars])

        return x, y

    def _get_distances(self, XY):
        """Return the distances between each point in a list of points."""
        from scipy.spatial.distance import cdist

        return np.diag(cdist(XY[:, :, 0], XY[:, :, 1], metric='euclidean'))

    def _get_variogram(self, XY, xy=None):
        """Return variograms."""
        return 0.5 * np.square(XY - xy)  # XY = 1st col xy= 2nd col

    def _emp_vario(self, x, y, data, Nsamp=1000):
        """Compute empirical semivariance."""
        # remove NaNs if possible
        mask = ~np.isnan(data)
        if False in mask:
            data = data[mask]
            x = x[mask]
            y = y[mask]

        # deramp
        _, _, x, y = WGS84_to_UTM(x, y, common_center=True)
        A = np.array([x, y, np.ones(len(x))]).T
        ramp = np.linalg.lstsq(A, data.T, rcond=None)[0]
        data = data - (np.matmul(A, ramp))

        samples, indpars = self._get_samples(data, Nsamp)
        x, y = self._get_XY(x, y, indpars)
        dists = self._get_distances(np.array([[x[:, 0], y[:, 0]], [x[:, 1], y[:, 1]]]).T)
        vario = self._get_variogram(samples[:, 0], samples[:, 1])

        return dists, vario

    def _binned_vario(self, hEff, rawVario, xBin=None):
        """Return a binned empirical variogram."""
        if xBin is None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                xBin = np.linspace(0, np.nanmax(hEff) * 0.67, 20)

        nBins = len(xBin) - 1
        hExp, expVario = [], []

        for iBin in range(nBins):
            iBinMask = np.logical_and(xBin[iBin] < hEff, hEff <= xBin[iBin + 1])
            # circumvent indexing
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Mean of empty slice')
                    hExp.append(np.nanmean(hEff[iBinMask]))
                    expVario.append(np.nanmean(rawVario[iBinMask]))
            except:  # TODO: Which error(s)?
                pass

        if False in ~np.isnan(hExp):
            # NaNs present in binned histogram
            hExp = [x for x in hExp if str(x) != 'nan']
            expVario = [x for x in expVario if str(x) != 'nan']

        return np.array(hExp), np.array(expVario)

    def _fit_vario(self, dists, vario, model=None, x0=None, Nparm=None, ub=None):
        """Fit a variogram model to data."""
        from scipy.optimize import least_squares

        def resid(x, d, v, m):
            return m(x, d) - v

        if ub is None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                ub = np.array([np.nanmax(dists) * 0.8, np.nanmax(vario) * 0.8, np.nanmax(vario) * 0.8])

        if x0 is None and Nparm is None:
            raise RuntimeError('Must specify either x0 or the number of model parameters')
        if x0 is not None:
            lb = np.zeros(len(x0))
        if Nparm is not None:
            lb = np.zeros(Nparm)
            x0 = (ub - lb) / 2
        bounds = (lb, ub)

        mask = np.isnan(dists) | np.isnan(vario)
        d = dists[~mask].copy()
        v = vario[~mask].copy()

        res_robust = least_squares(
            resid,
            x0,
            bounds=bounds,
            loss='soft_l1',
            f_scale=0.1,
            args=(d, v, model),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            d_test = np.linspace(0, np.nanmax(dists), 100)
        # v_test is my y., # res_robust.x =a, b, c, where a = range, b = sill, and c = nugget model, d_test=x
        v_test = model(res_robust.x, d_test)

        return res_robust, d_test, v_test

    # this would be exponential plus nugget
    def __exponential__(self, parms, h, nugget=False):
        """Return variogram model given a set of arguments and keyword arguments."""
        # a = range, b = sill, c = nugget model
        a, b, c = parms
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='overflow encountered in true_divide')
            if nugget:
                return b * (1 - np.exp(-h / a)) + c
            else:
                return b * (1 - np.exp(-h / a))

    # this would be gaussian plus nugget
    def __gaussian__(self, parms, h):
        """Returns a Gaussian variogram model."""
        a, b, c = parms
        return b * (1 - np.exp(-np.square(h) / (a**2))) + c

    def _append_variogram(self, grid_ind, grid_subset):
        """For a given grid-cell, iterate through time slices to generate/append empirical variogram(s)."""
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
                self.skipped_slices.append([grid_ind, j.strftime('%Y-%m-%d')])
            else:
                self.gridcenterlist.append(
                    [
                        f'grid{grid_ind} '
                        + f'Lat:{str(self.gridpoints[grid_ind][1])} Lon:{str(self.gridpoints[grid_ind][0])}'
                    ]
                )
                lonarr = np.array(grid_subset[grid_subset['Date'] == j]['Lon'])
                latarr = np.array(grid_subset[grid_subset['Date'] == j]['Lat'])
                delayarray = np.array(grid_subset[grid_subset['Date'] == j][self.col_name])
                # fit empirical variogram for each time AND grid
                dists, vario = self._emp_vario(lonarr, latarr, delayarray)
                dists_binned, vario_binned = self._binned_vario(dists, vario)
                # fit experimental variogram for each time AND grid, model default is exponential
                res_robust, d_test, v_test = self._fit_vario(
                    dists_binned, vario_binned, model=self.__exponential__, x0=None, Nparm=3
                )
                # Plot empirical + experimental variogram for this gridnode and timeslice
                if not os.path.exists(os.path.join(self.workdir, f'variograms/grid{grid_ind}')):
                    os.makedirs(os.path.join(self.workdir, f'variograms/grid{grid_ind}'))
                # Make variogram plots for each time-slice
                if self.variogram_per_timeslice:
                    # Plot empirical variogram for this gridnode and timeslice
                    self.plot_variogram(
                        grid_ind,
                        j.strftime('%Y%m%d'),
                        [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                        workdir=os.path.join(self.workdir, f'variograms/grid{grid_ind}'),
                        dists=dists,
                        vario=vario,
                        dists_binned=dists_binned,
                        vario_binned=vario_binned,
                    )
                    # Plot experimental variogram for this gridnode and timeslice
                    self.plot_variogram(
                        grid_ind,
                        j.strftime('%Y%m%d'),
                        [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                        workdir=os.path.join(self.workdir, f'variograms/grid{grid_ind}'),
                        d_test=d_test,
                        v_test=v_test,
                        res_robust=res_robust.x,
                        dists_binned=dists_binned,
                        vario_binned=vario_binned,
                    )
                # append for plotting
                self.good_slices.append([grid_ind, j.strftime('%Y%m%d')])
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
                # dists_binned_arr = dists_arr ; vario_binned_arr = vario_arr
                dists_binned_arr, vario_binned_arr = self._binned_vario(dists_arr, vario_arr)
            TOT_res_robust, TOT_d_test, TOT_v_test = self._fit_vario(
                dists_binned_arr, vario_binned_arr, model=self.__exponential__, x0=None, Nparm=3
            )
            tot_timetag = self.good_slices[0][1] + '–' + self.good_slices[-1][1]
            # Append TOT arrays
            self.TOT_good_slices.append([grid_ind, tot_timetag])
            self.TOT_res_robust_arr.append(TOT_res_robust.x)
            self.TOT_tot_timetag.append(tot_timetag)
            var_rmse = np.sqrt(np.nanmean((TOT_res_robust.fun) ** 2))
            if var_rmse <= self.variogram_errlimit:
                self.TOT_res_robust_rmse.append(var_rmse)
            else:
                self.TOT_res_robust_rmse.append(np.array(np.nan))
            # Plot empirical variogram for this gridnode
            self.plot_variogram(
                grid_ind,
                tot_timetag,
                [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                workdir=os.path.join(self.workdir, f'variograms/grid{grid_ind}'),
                dists=dists_arr,
                vario=vario_arr,
                dists_binned=dists_binned_arr,
                vario_binned=vario_binned_arr,
                seasonalinterval=self.seasonalinterval,
            )
            # Plot experimental variogram for this gridnode
            self.plot_variogram(
                grid_ind,
                tot_timetag,
                [self.gridpoints[grid_ind][1], self.gridpoints[grid_ind][0]],
                workdir=os.path.join(self.workdir, f'variograms/grid{grid_ind}'),
                d_test=TOT_d_test,
                v_test=TOT_v_test,
                res_robust=TOT_res_robust.x,
                seasonalinterval=self.seasonalinterval,
                dists_binned=dists_binned_arr,
                vario_binned=vario_binned_arr,
            )
        # Record sparse grids which didn't have sufficient sample size of data through any of the timeslices
        else:
            self.sparse_grids.append(grid_ind)

        return self.TOT_good_slices, self.TOT_res_robust_arr, self.TOT_res_robust_rmse, self.gridcenterlist

    def create_variograms(self):
        """Iterate through grid-cells and time slices to generate empirical variogram(s)."""
        # track data for plotting
        self.TOT_good_slices = []
        self.TOT_res_robust_arr = []
        self.TOT_res_robust_rmse = []
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
        with mp.Pool(self.numCPUs) as multipool:
            for i, j, k, l in multipool.starmap(self._append_variogram, args):
                self.TOT_good_slices.extend(i)
                self.TOT_res_robust_arr.extend(j)
                self.TOT_res_robust_rmse.extend(k)
                self.gridcenterlist.extend(l)

        # save grid-center lookup table
        self.gridcenterlist = [list(i) for i in set(tuple(j) for j in self.gridcenterlist)]
        self.gridcenterlist.sort(key=lambda x: int(x[0][4:6]))
        gridcenter = open((os.path.join(self.workdir, 'variograms/gridlocation_lookup.txt')), 'w')
        for element in self.gridcenterlist:
            gridcenter.writelines('\n'.join(element))
            gridcenter.write('\n')
        gridcenter.close()

        TOT_grids = [i[0] for i in self.TOT_good_slices]

        return TOT_grids, self.TOT_res_robust_arr, self.TOT_res_robust_rmse

    def plot_variogram(
        self,
        gridID,
        timeslice,
        coords,
        workdir='./',
        d_test=None,
        v_test=None,
        res_robust=None,
        dists=None,
        vario=None,
        dists_binned=None,
        vario_binned=None,
        seasonalinterval=None,
    ) -> None:
        """Make empirical and/or experimental variogram fit plots."""
        # If specified workdir doesn't exist, create it
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        # make plot title
        title_str = f' \nLat:{coords[1]:.2f} Lon:{coords[0]:.2f}\nTime:{str(timeslice)}'
        if seasonalinterval:
            title_str += (
                ' Season(mm/dd): '
                f'{int(timeslice[4:6])}/{int(timeslice[6:8])} – {int(timeslice[-4:-2])}/{int(timeslice[-2:])}'
            )

        if dists is not None and vario is not None:
            # scale from m to user-defined units
            dists = [convert_SI(i, 'm', self.unit) for i in dists]
            plt.scatter(dists, vario, s=1, facecolor='0.5', label='raw')
        if dists_binned is not None and vario_binned is not None:
            # scale from m to user-defined units
            dists_binned = [convert_SI(i, 'm', self.unit) for i in dists_binned]
            plt.plot(dists_binned, vario_binned, 'bo', label='binned')
        if res_robust is not None:
            plt.axhline(y=res_robust[1], color='g', linestyle='--', label=f'ɣ\u0332\u00b2({self.unit}\u00b2)')
            # scale from m to user-defined units
            res_robust[0] = convert_SI(res_robust[0], 'm', self.unit)
            plt.axvline(x=res_robust[0], color='c', linestyle='--', label=f'h ({self.unit})')
        if d_test is not None and v_test is not None:
            # scale from m to user-defined units
            d_test = [convert_SI(i, 'm', self.unit) for i in d_test]
            plt.plot(d_test, v_test, 'r-', label='experimental fit')
        plt.xlabel(f'Distance ({self.unit})')
        plt.ylabel(f'Dissimilarity ({self.unit}\u00b2)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0, framealpha=1.0)
        # Plot empirical variogram
        if d_test is None and v_test is None:
            plt.title('Empirical variogram' + title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(workdir, f'grid{gridID}_timeslice{timeslice}_justEMPvariogram.eps'))
        # Plot just experimental variogram
        else:
            plt.title('Experimental variogram' + title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(workdir, f'grid{gridID}_timeslice{timeslice}_justEXPvariogram.eps'))
        plt.close()


class RaiderStats:
    """Class which loads standard weather model/GPS delay files and generates a series of user-requested statistics and graphics."""

    # import dependencies
    import glob

    def __init__(
        self,
        filearg,
        col_name,
        unit='m',
        workdir='./',
        bbox=None,
        spacing=1,
        timeinterval=None,
        seasonalinterval=None,
        obs_errlimit='inf',
        time_lines=False,
        stationsongrids=False,
        station_seasonal_phase=False,
        cbounds=None,
        colorpercentile=[25, 95],
        usr_colormap='hot_r',
        grid_heatmap=False,
        grid_delay_mean=False,
        grid_delay_median=False,
        grid_delay_stdev=False,
        grid_seasonal_phase=False,
        grid_delay_absolute_mean=False,
        grid_delay_absolute_median=False,
        grid_delay_absolute_stdev=False,
        grid_seasonal_absolute_phase=False,
        grid_to_raster=False,
        min_span=[2, 0.6],
        period_limit=0.0,
        numCPUs=8,
        phaseamp_per_station=False,
    ) -> None:
        self.fname = filearg
        self.col_name = col_name
        self.unit = unit
        self.workdir = workdir
        self.bbox = bbox
        self.spacing = spacing
        self.timeinterval = timeinterval
        self.seasonalinterval = seasonalinterval
        self.obs_errlimit = float(obs_errlimit)
        self.time_lines = time_lines
        self.stationsongrids = stationsongrids
        self.station_seasonal_phase = station_seasonal_phase
        self.cbounds = cbounds
        self.colorpercentile = colorpercentile
        self.usr_colormap = usr_colormap
        self.grid_heatmap = grid_heatmap
        self.grid_delay_mean = grid_delay_mean
        self.grid_delay_median = grid_delay_median
        self.grid_delay_stdev = grid_delay_stdev
        self.grid_seasonal_phase = grid_seasonal_phase
        self.grid_seasonal_amplitude = False
        self.grid_seasonal_period = False
        self.grid_seasonal_phase_stdev = False
        self.grid_seasonal_amplitude_stdev = False
        self.grid_seasonal_period_stdev = False
        self.grid_seasonal_fit_rmse = False
        self.grid_delay_absolute_mean = grid_delay_absolute_mean
        self.grid_delay_absolute_median = grid_delay_absolute_median
        self.grid_delay_absolute_stdev = grid_delay_absolute_stdev
        self.grid_seasonal_absolute_phase = grid_seasonal_absolute_phase
        self.grid_seasonal_absolute_amplitude = False
        self.grid_seasonal_absolute_period = False
        self.grid_seasonal_absolute_phase_stdev = False
        self.grid_seasonal_absolute_amplitude_stdev = False
        self.grid_seasonal_absolute_period_stdev = False
        self.grid_seasonal_absolute_fit_rmse = False
        self.grid_to_raster = grid_to_raster
        self.min_span = min_span
        self.period_limit = period_limit
        self.numCPUs = numCPUs
        self.phaseamp_per_station = phaseamp_per_station
        self.grid_range = False
        self.grid_variance = False
        self.grid_variogram_rmse = False

        # create workdir if it doesn't exist
        if not os.path.exists(self.workdir):
            os.mkdir(self.workdir)

        # get colorbounds
        if self.cbounds:
            self.cbounds = [float(val) for val in self.cbounds.split()]

        # Pass color percentile and check for input error
        if self.colorpercentile is None:
            self.colorpercentile = [25, 95]
        if self.colorpercentile[0] > self.colorpercentile[1]:
            raise Exception(
                f'Input colorpercentile lower threshold {self.colorpercentile[0]} higher than upper threshold {self.colorpercentile[1]}'
            )

        # load dataframe directly if previously generated TIF grid-file
        if self.fname.endswith('.tif'):
            if 'grid_heatmap' in self.fname:
                (
                    self.grid_heatmap,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_heatmap')[0]
            if 'grid_delay_mean' in self.fname:
                (
                    self.grid_delay_mean,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_delay_mean')[0]
            if 'grid_delay_median' in self.fname:
                (
                    self.grid_delay_median,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_delay_median')[0]
            if 'grid_delay_stdev' in self.fname:
                (
                    self.grid_delay_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_delay_stdev')[0]
            if 'grid_seasonal_phase' in self.fname:
                (
                    self.grid_seasonal_phase,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_phase')[0]
            if 'grid_seasonal_period' in self.fname:
                (
                    self.grid_seasonal_period,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_period')[0]
            if 'grid_seasonal_amplitude' in self.fname:
                (
                    self.grid_seasonal_amplitude,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_amplitude')[0]
            if 'grid_seasonal_phase_stdev' in self.fname:
                (
                    self.grid_seasonal_phase_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_phase_stdev')[0]
            if 'grid_seasonal_amplitude_stdev' in self.fname:
                (
                    self.grid_seasonal_amplitude_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_amplitude_stdev')[0]
            if 'grid_seasonal_period_stdev' in self.fname:
                (
                    self.grid_seasonal_period_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_period_stdev')[0]
            if 'grid_seasonal_fit_rmse' in self.fname:
                (
                    self.grid_seasonal_fit_rmse,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_fit_rmse')[0]
            if 'grid_delay_absolute_mean' in self.fname:
                (
                    self.grid_delay_absolute_mean,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_delay_absolute_mean')[0]
            if 'grid_delay_absolute_median' in self.fname:
                (
                    self.grid_delay_absolute_median,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_delay_absolute_median')[0]
            if 'grid_delay_absolute_stdev' in self.fname:
                (
                    self.grid_delay_absolute_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_delay_absolute_stdev')[0]
            if 'grid_seasonal_absolute_phase' in self.fname:
                (
                    self.grid_seasonal_absolute_phase,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_phase')[0]
            if 'grid_seasonal_absolute_period' in self.fname:
                (
                    self.grid_seasonal_absolute_period,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_period')[0]
            if 'grid_seasonal_absolute_amplitude' in self.fname:
                (
                    self.grid_seasonal_absolute_amplitude,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_amplitude')[0]
            if 'grid_seasonal_absolute_phase_stdev' in self.fname:
                (
                    self.grid_seasonal_absolute_phase_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_phase_stdev')[0]
            if 'grid_seasonal_absolute_amplitude_stdev' in self.fname:
                (
                    self.grid_seasonal_absolute_amplitude_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_amplitude_stdev')[0]
            if 'grid_seasonal_absolute_period_stdev' in self.fname:
                (
                    self.grid_seasonal_absolute_period_stdev,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_period_stdev')[0]
            if 'grid_seasonal_absolute_fit_rmse' in self.fname:
                (
                    self.grid_seasonal_absolute_fit_rmse,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_seasonal_absolute_fit_rmse')[0]
            if 'grid_range' in self.fname:
                (
                    self.grid_range,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_range')[0]
            if 'grid_variance' in self.fname:
                (
                    self.grid_variance,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_variance')[0]
            if 'grid_variogram_rmse' in self.fname:
                (
                    self.grid_variogram_rmse,
                    self.plotbbox,
                    self.spacing,
                    self.colorbarfmt,
                    self.stationsongrids,
                    self.time_lines,
                ) = load_gridfile(self.fname, self.unit)
                self.col_name = os.path.basename(self.fname).split('_' + 'grid_variogram_rmse')[0]
        # setup dataframe for statistical analyses (if CSV)
        if self.fname.endswith('.csv'):
            self.create_DF()

    def _get_extent(self):  # dataset, spacing=1, userbbox=None
        """Get the bbox, spacing in deg (by default 1deg), optionally pass user-specified bbox. Output array in WESN degrees."""
        extent = [
            np.floor(min(self.df['Lon'])),
            np.ceil(max(self.df['Lon'])),
            np.floor(min(self.df['Lat'])),
            np.ceil(max(self.df['Lat'])),
        ]
        if self.bbox is not None:
            dfextents_poly = Polygon(
                np.column_stack(
                    (
                        np.array([extent[0], extent[0], extent[1], extent[1], extent[0]]),
                        np.array([extent[2], extent[3], extent[3], extent[2], extent[2]]),
                    )
                )
            )
            userbbox_poly = Polygon(
                np.column_stack(
                    (
                        np.array([self.bbox[2], self.bbox[3], self.bbox[3], self.bbox[2], self.bbox[2]]),
                        np.array([self.bbox[0], self.bbox[0], self.bbox[1], self.bbox[1], self.bbox[0]]),
                    )
                )
            )
            if userbbox_poly.intersects(dfextents_poly):
                extent = [np.floor(self.bbox[2]), np.ceil(self.bbox[-1]), np.floor(self.bbox[0]), np.ceil(self.bbox[1])]
            else:
                raise Exception(
                    'User-specified bounds do not overlap with dataset bounds, adjust bounds and re-run program.'
                )
            if extent[0] < -180.0 or extent[1] > 180.0 or extent[2] < -90.0 or extent[3] > 90.0:
                raise Exception(
                    'Specified bounds exceed -180/180 lon and/or -90/90 lat, adjust bounds and re-run program.'
                )
            del dfextents_poly, userbbox_poly

        # ensure that extents do not exceed -180/180 lon and -90/90 lat
        if extent[0] < -180.0:
            extent[0] = -180.0
        if extent[1] > 180.0:
            extent[1] = 180.0
        if extent[2] < -90.0:
            extent[2] = -90.0
        if extent[3] > 90.0:
            extent[3] = 90.0

        # ensure even spacing, set spacing to 1 if specified spacing is not even multiple of bounds
        if (extent[1] - extent[0]) % self.spacing != 0 or (extent[-1] - extent[-2]) % self.spacing:
            logger.warning(
                'User-specified spacing %s is not even multiple of bounds, resetting spacing to 1\N{DEGREE SIGN}',
                self.spacing,
            )
            self.spacing = 1

        # Create corners of rectangle to be transformed to a grid
        nw = [extent[0] + (self.spacing / 2), extent[-1] - (self.spacing / 2)]
        se = [extent[1] - (self.spacing / 2), extent[2] + (self.spacing / 2)]

        # Store grid dimension [y,x]
        grid_dim = [int((extent[1] - extent[0]) / self.spacing), int((extent[-1] - extent[-2]) / self.spacing)]

        # Iterate over 2D area
        gridpoints = []
        y_shape = []
        x_shape = []
        x = se[0]
        while x >= nw[0]:
            y = se[1]
            while y <= nw[1]:
                y_shape.append(y)
                gridpoints.append([x, y])
                y += self.spacing
            x_shape.append(x)
            x -= self.spacing
        gridpoints.reverse()

        return extent, grid_dim, gridpoints

    def _check_stationgrid_intersection(self, stat_ID):
        """
        Return index of grid cell which intersects with station
        Note: Fast, but assumes station locations don't change.
        """
        coord = Point(
            (
                self.unique_points[1][self.unique_points[0].index(stat_ID)],
                self.unique_points[2][self.unique_points[0].index(stat_ID)],
            )
        )
        # Get grid cell polygon which intersect with station coordinate
        grid_int = self.polygon_tree.query(coord)
        # Pass corresponding grid cell index
        if len(grid_int) != 0:
            return grid_int[0]
        else:
            return 'NaN'

    def _reader(self):
        """Read a input file."""
        try:
            data = pd.read_csv(self.fname, parse_dates=['Datetime'])
            data['Date'] = data['Datetime'].apply(lambda x: x.date())
            data['Date'] = data['Date'].apply(lambda x: dt.datetime.strptime(x.strftime('%Y-%m-%d'), '%Y-%m-%d'))
        except:
            data = pd.read_csv(self.fname, parse_dates=['Date'])

        # check if user-specified key is valid
        if self.col_name not in data.keys():
            raise Exception(
                f'User-specified key {self.col_name} not found in input file {self.fname}. Must specify valid key.'
            )

        # if user-specified key is the same as the 'Date' field, rename
        if self.col_name == 'Date':
            logger.warning(f'Input key {self.col_name} same as "Date" field name, rename the former')
            self.col_name += '_plot'
            data[self.col_name] = data['Date']

        # convert to specified output unit
        inputunit = 'm'
        data[self.col_name] = convert_SI(data[self.col_name], inputunit, self.unit)
        # filter out obs by error
        if 'sigZTD' in data.keys():
            data['sigZTD'] = convert_SI(data['sigZTD'], inputunit, self.unit)
            self.obs_errlimit = convert_SI(self.obs_errlimit, inputunit, self.unit)
            data = data[data['sigZTD'] <= self.obs_errlimit]
        else:
            logger.warning('Key "sigZTD" not found in dataset, cannot filter out obs by error')

        return data

    def create_DF(self) -> None:
        """Create dataframe."""
        # Open file
        self.df = self._reader()

        # Filter dataframe
        # drop all nans
        self.df.dropna(how='any', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        # convert to datetime object

        # time-interval filter
        if self.timeinterval:
            self.timeinterval = [dt.datetime.strptime(val, '%Y-%m-%d') for val in self.timeinterval.split()]
            self.df = self.df[(self.df['Date'] >= self.timeinterval[0]) & (self.df['Date'] <= self.timeinterval[-1])]

        # seasonal filter
        if self.seasonalinterval:
            self.seasonalinterval = self.seasonalinterval.split()
            # get day of year
            self.seasonalinterval = [
                dt.datetime.strptime('2001-' + self.seasonalinterval[0], '%Y-%m-%d').timetuple().tm_yday,
                dt.datetime.strptime('2001-' + self.seasonalinterval[-1], '%Y-%m-%d').timetuple().tm_yday,
            ]
            # track input order and wrap around year if necessary
            # e.g. month/day: 03/01 to 06/01
            if self.seasonalinterval[0] < self.seasonalinterval[1]:
                # non leap-year
                filtered_self = self.df[
                    (not self.df['Date'].dt.is_leap_year)
                    & (self.df['Date'].dt.dayofyear >= self.seasonalinterval[0])
                    & (self.df['Date'].dt.dayofyear <= self.seasonalinterval[-1])
                ]
                # leap-year
                self.seasonalinterval = [i + 1 if i > 59 else i for i in self.seasonalinterval]
                filtered_self_ly = self.df[
                    (self.df['Date'].dt.is_leap_year)
                    & (self.df['Date'].dt.dayofyear >= self.seasonalinterval[0])
                    & (self.df['Date'].dt.dayofyear <= self.seasonalinterval[-1])
                ]
                self.df = pd.concat([filtered_self, filtered_self_ly], ignore_index=True)
                del filtered_self
            # e.g. month/day: 12/01 to 03/01
            if self.seasonalinterval[0] > self.seasonalinterval[1]:
                # non leap-year
                filtered_self = self.df[
                    (not self.df['Date'].dt.is_leap_year)
                    & (self.df['Date'].dt.dayofyear >= self.seasonalinterval[-1])
                    & (self.df['Date'].dt.dayofyear <= self.seasonalinterval[0])
                ]
                # leap-year
                self.seasonalinterval = [i + 1 if i > 59 else i for i in self.seasonalinterval]
                filtered_self_ly = self.df[
                    (self.df['Date'].dt.is_leap_year)
                    & (self.df['Date'].dt.dayofyear >= self.seasonalinterval[-1])
                    & (self.df['Date'].dt.dayofyear <= self.seasonalinterval[0])
                ]
                self.df = pd.concat([filtered_self, filtered_self_ly], ignore_index=True)
                del filtered_self

        # estimate central longitude lines if '--time_lines' specified
        if self.time_lines and 'Datetime' in self.df.keys():
            self.df['Date_hr'] = self.df['Datetime'].dt.hour.astype(float).astype('Int32')
            # get list of unique times
            all_hrs = sorted(set(self.df['Date_hr']))

            # get central longitude bands associated with each time
            central_points = []
            # if single time, avoid loop
            if len(all_hrs) == 1:
                central_points.append(([0, max(self.df['Lon'])], [0, min(self.df['Lon'])]))
            else:
                for i in enumerate(all_hrs):
                    # last entry
                    if i[0] == len(all_hrs) - 1:
                        lons = self.df[self.df['Date_hr'] > all_hrs[i[0] - 1]]
                    # first entry
                    elif i[0] == 0:
                        lons = self.df[self.df['Date_hr'] < all_hrs[i[0] + 1]]
                    else:
                        lons = self.df[
                            (self.df['Date_hr'] > all_hrs[i[0] - 1]) & (self.df['Date_hr'] < all_hrs[i[0] + 1])
                        ]
                    central_points.append(([0, max(lons['Lon'])], [0, min(lons['Lon'])]))
            # get central longitudes
            self.time_lines = [midpoint(i[0], i[1]) for i in central_points]

        # Get bbox, buffered by grid spacing.
        # Check if bbox input is valid list.
        if self.bbox is not None:
            try:
                self.bbox = [float(val) for val in self.bbox.split()]
            except:
                raise Exception(
                    'Cannot understand the --bounding_box argument. String input is incorrect or path does not exist.'
                )
        self.plotbbox, self.grid_dim, self.gridpoints = self._get_extent()

        # generate list of grid-polygons
        append_poly = []
        for i in self.gridpoints:
            bbox = [
                i[1] - (self.spacing / 2),
                i[1] + (self.spacing / 2),
                i[0] - (self.spacing / 2),
                i[0] + (self.spacing / 2),
            ]
            append_poly.append(
                Polygon(
                    np.column_stack(
                        (
                            np.array([bbox[2], bbox[3], bbox[3], bbox[2], bbox[2]]),
                            np.array([bbox[0], bbox[0], bbox[1], bbox[1], bbox[0]]),
                        )
                    )
                )
            )  # Pass lons/lats to create polygon

        # Check for grid cell intersection with each station
        idtogrid_dict = {}
        self.unique_points = self.df.groupby(['ID', 'Lon', 'Lat']).size()
        self.unique_points = [
            self.unique_points.index.get_level_values('ID').tolist(),
            self.unique_points.index.get_level_values('Lon').tolist(),
            self.unique_points.index.get_level_values('Lat').tolist(),
        ]
        # Initiate R-tree of gridded array domain
        self.polygon_tree = STRtree(append_poly)
        for stat_ID in self.unique_points[0]:
            grd_index = self._check_stationgrid_intersection(stat_ID)
            idtogrid_dict[stat_ID] = grd_index

        # map gridnode dictionary to dataframe
        self.df['gridnode'] = self.df['ID'].map(idtogrid_dict)
        self.df = self.df[self.df['gridnode'].astype(str) != 'NaN']
        del self.unique_points, self.polygon_tree, idtogrid_dict, append_poly
        # sort by grid and date
        self.df.sort_values(['gridnode', 'Date'])

        # If specified, pass station locations to superimpose on gridplots
        if self.stationsongrids:
            unique_points = self.df.groupby(['Lon', 'Lat']).size()
            self.stationsongrids = [
                unique_points.index.get_level_values('Lon').tolist(),
                unique_points.index.get_level_values('Lat').tolist(),
            ]

        # If specified, setup gridded array(s)
        if self.grid_heatmap:
            self.grid_heatmap = (
                np.array(
                    [
                        np.nan
                        if i[0] not in self.df['gridnode'].values[:]
                        else int(len(np.unique(self.df['ID'][self.df['gridnode'] == i[0]])))
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_heatmap' + '.tif')
                save_gridfile(
                    self.grid_heatmap,
                    'grid_heatmap',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%1i',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='int16',
                    noData=0,
                )

        if self.grid_delay_mean:
            # Take mean of station-wise means per gridcell
            unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)[self.col_name].mean()
            unique_points = unique_points.groupby(['gridnode'])[self.col_name].mean()
            unique_points.dropna(how='any', inplace=True)
            self.grid_delay_mean = (
                np.array(
                    [
                        np.nan
                        if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                        else unique_points[i[0]]
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_delay_mean' + '.tif')
                save_gridfile(
                    self.grid_delay_mean,
                    'grid_delay_mean',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%.2f',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='float32',
                )

        if self.grid_delay_median:
            # Take mean of station-wise medians per gridcell
            unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)[self.col_name].median()
            unique_points = unique_points.groupby(['gridnode'])[self.col_name].mean()
            unique_points.dropna(how='any', inplace=True)
            self.grid_delay_median = (
                np.array(
                    [
                        np.nan
                        if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                        else unique_points[i[0]]
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_delay_median' + '.tif')
                save_gridfile(
                    self.grid_delay_median,
                    'grid_delay_median',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%.2f',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='float32',
                )

        if self.grid_delay_stdev:
            # Take mean of station-wise stdev per gridcell
            unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)[self.col_name].std()
            unique_points = unique_points.groupby(['gridnode'])[self.col_name].mean()
            unique_points.dropna(how='any', inplace=True)
            self.grid_delay_stdev = (
                np.array(
                    [
                        np.nan
                        if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                        else unique_points[i[0]]
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_delay_stdev' + '.tif')
                save_gridfile(
                    self.grid_delay_stdev,
                    'grid_delay_stdev',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%.2f',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='float32',
                )

        if self.grid_delay_absolute_mean:
            # Take mean of all data per gridcell
            unique_points = self.df.groupby(['gridnode'])[self.col_name].mean()
            unique_points.dropna(how='any', inplace=True)
            self.grid_delay_absolute_mean = (
                np.array(
                    [
                        np.nan
                        if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                        else unique_points[i[0]]
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_delay_absolute_mean' + '.tif')
                save_gridfile(
                    self.grid_delay_absolute_mean,
                    'grid_delay_absolute_mean',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%.2f',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='float32',
                )

        if self.grid_delay_absolute_median:
            # Take median of all data per gridcell
            unique_points = self.df.groupby(['gridnode'])[self.col_name].median()
            unique_points.dropna(how='any', inplace=True)
            self.grid_delay_absolute_median = (
                np.array(
                    [
                        np.nan
                        if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                        else unique_points[i[0]]
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_delay_absolute_median' + '.tif')
                save_gridfile(
                    self.grid_delay_absolute_median,
                    'grid_delay_absolute_median',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%.2f',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='float32',
                )

        if self.grid_delay_absolute_stdev:
            # Take stdev of all data per gridcell
            unique_points = self.df.groupby(['gridnode'])[self.col_name].std()
            unique_points.dropna(how='any', inplace=True)
            self.grid_delay_absolute_stdev = (
                np.array(
                    [
                        np.nan
                        if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                        else unique_points[i[0]]
                        for i in enumerate(self.gridpoints)
                    ]
                )
                .reshape(self.grid_dim)
                .T
            )
            # If specified, save gridded array(s)
            if self.grid_to_raster:
                gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_delay_absolute_stdev' + '.tif')
                save_gridfile(
                    self.grid_delay_absolute_stdev,
                    'grid_delay_absolute_stdev',
                    gridfile_name,
                    self.plotbbox,
                    self.spacing,
                    self.unit,
                    colorbarfmt='%.2f',
                    stationsongrids=self.stationsongrids,
                    time_lines=self.time_lines,
                    dtype='float32',
                )

        # If specified, compute phase/amplitude fits
        if self.station_seasonal_phase or self.grid_seasonal_phase or self.grid_seasonal_absolute_phase:
            # Sort by coordinates
            unique_points = self.df.sort_values(['ID', 'Date'])
            unique_points['Date'] = [i.timestamp() for i in unique_points['Date']]
            # Setup variables
            self.ampfit = []
            self.phsfit = []
            self.periodfit = []
            self.ampfit_c = []
            self.phsfit_c = []
            self.periodfit_c = []
            self.seasonalfit_rmse = []
            args = []
            for i in sorted(list(set(unique_points['ID']))):
                # pass all values corresponding to station (ID, data = y, time = x)
                args.append(
                    (
                        i,
                        unique_points[unique_points['ID'] == i]['Date'].to_list(),
                        unique_points[unique_points['ID'] == i][self.col_name].to_list(),
                        self.min_span[0],
                        self.min_span[1],
                        self.period_limit,
                    )
                )
            # Parallelize iteration through all grid-cells and time slices
            with mp.Pool(self.numCPUs) as multipool:
                for i, j, k, l, m, n, o in multipool.starmap(self._amplitude_and_phase, args):
                    self.ampfit.extend(i)
                    self.phsfit.extend(j)
                    self.periodfit.extend(k)
                    self.ampfit_c.extend(l)
                    self.phsfit_c.extend(m)
                    self.periodfit_c.extend(n)
                    self.seasonalfit_rmse.extend(o)
            # map phase/amplitude fits dictionary to dataframe
            self.phsfit = {k: v for d in self.phsfit for k, v in d.items()}
            self.ampfit = {k: v for d in self.ampfit for k, v in d.items()}
            self.periodfit = {k: v for d in self.periodfit for k, v in d.items()}
            self.df['phsfit'] = self.df['ID'].map(self.phsfit)
            # check if there are any valid data values
            if self.df['phsfit'].isnull().values.all(axis=0):
                raise Exception(
                    f'No valid data values, adjust --min_span inputs for time span in years {self.min_span[0]} and/or fractional obs. {self.min_span[1]}'
                )
            self.df['ampfit'] = self.df['ID'].map(self.ampfit)
            self.df['periodfit'] = self.df['ID'].map(self.periodfit)
            self.phsfit_c = {k: v for d in self.phsfit_c for k, v in d.items()}
            self.ampfit_c = {k: v for d in self.ampfit_c for k, v in d.items()}
            self.periodfit_c = {k: v for d in self.periodfit_c for k, v in d.items()}
            self.seasonalfit_rmse = {k: v for d in self.seasonalfit_rmse for k, v in d.items()}
            self.df['phsfit_c'] = self.df['ID'].map(self.phsfit_c)
            self.df['ampfit_c'] = self.df['ID'].map(self.ampfit_c)
            self.df['periodfit_c'] = self.df['ID'].map(self.periodfit_c)
            self.df['seasonalfit_rmse'] = self.df['ID'].map(self.seasonalfit_rmse)
            # drop nan
            self.df.dropna(how='any', inplace=True)
            # If grid plots specified
            if self.grid_seasonal_phase:
                # Pass mean phase of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)['phsfit'].mean()
                unique_points = unique_points.groupby(['gridnode'])['phsfit'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_phase = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_seasonal_phase' + '.tif')
                    save_gridfile(
                        self.grid_seasonal_phase,
                        'grid_seasonal_phase',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'days',
                        colorbarfmt='%.1i',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass mean amplitude of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)['ampfit'].mean()
                unique_points = unique_points.groupby(['gridnode'])['ampfit'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_amplitude = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_seasonal_amplitude' + '.tif')
                    save_gridfile(
                        self.grid_seasonal_amplitude,
                        'grid_seasonal_amplitude',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        self.unit,
                        colorbarfmt='%.3f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass mean period of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)['periodfit'].mean()
                unique_points = unique_points.groupby(['gridnode'])['periodfit'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_period = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_seasonal_period' + '.tif')
                    save_gridfile(
                        self.grid_seasonal_period,
                        'grid_seasonal_period',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'years',
                        colorbarfmt='%.2f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                ########################################################################################################################
                # Pass mean phase stdev of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)['phsfit_c'].mean()
                unique_points = unique_points.groupby(['gridnode'])['phsfit_c'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_phase_stdev = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_phase_stdev' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_phase_stdev,
                        'grid_seasonal_phase_stdev',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'days',
                        colorbarfmt='%.1i',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass mean amplitude stdev of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)['ampfit_c'].mean()
                unique_points = unique_points.groupby(['gridnode'])['ampfit_c'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_amplitude_stdev = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_amplitude_stdev' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_amplitude_stdev,
                        'grid_seasonal_amplitude_stdev',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        self.unit,
                        colorbarfmt='%.3f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass mean period stdev of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)['periodfit_c'].mean()
                unique_points = unique_points.groupby(['gridnode'])['periodfit_c'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_period_stdev = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_period_stdev' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_period_stdev,
                        'grid_seasonal_period_stdev',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'years',
                        colorbarfmt='%.2e',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass mean seasonal fit RMSE of station-wise means per gridcell
                unique_points = self.df.groupby(['ID', 'Lon', 'Lat', 'gridnode'], as_index=False)[
                    'seasonalfit_rmse'
                ].mean()
                unique_points = unique_points.groupby(['gridnode'])['seasonalfit_rmse'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_fit_rmse = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(self.workdir, self.col_name + '_' + 'grid_seasonal_fit_rmse' + '.tif')
                    save_gridfile(
                        self.grid_seasonal_fit_rmse,
                        'grid_seasonal_fit_rmse',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        self.unit,
                        colorbarfmt='%.3f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
            ########################################################################################################################
            if self.grid_seasonal_absolute_phase:
                # Pass absolute mean phase of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['phsfit'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_phase = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_phase' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_phase,
                        'grid_seasonal_absolute_phase',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'days',
                        colorbarfmt='%.1i',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass absolute mean amplitude of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['ampfit'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_amplitude = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_amplitude' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_amplitude,
                        'grid_seasonal_absolute_amplitude',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        self.unit,
                        colorbarfmt='%.3f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass absolute mean period of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['periodfit'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_period = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_period' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_period,
                        'grid_seasonal_absolute_period',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'years',
                        colorbarfmt='%.2f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                ########################################################################################################################
                # Pass absolute mean phase stdev of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['phsfit_c'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_phase_stdev = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_phase_stdev' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_phase_stdev,
                        'grid_seasonal_absolute_phase_stdev',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'days',
                        colorbarfmt='%.1i',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass absolute mean amplitude stdev of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['ampfit_c'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_amplitude_stdev = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_amplitude_stdev' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_amplitude_stdev,
                        'grid_seasonal_absolute_amplitude_stdev',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        self.unit,
                        colorbarfmt='%.3f',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )
                # Pass absolute mean period stdev of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['periodfit_c'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_period_stdev = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_period_stdev' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_period_stdev,
                        'grid_seasonal_absolute_period_stdev',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        'years',
                        colorbarfmt='%.2e',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )

                # Pass absolute mean seasonal fit RMSE of all data per gridcell
                unique_points = self.df.groupby(['gridnode'])['seasonalfit_rmse'].mean()
                unique_points.dropna(how='any', inplace=True)
                self.grid_seasonal_absolute_fit_rmse = (
                    np.array(
                        [
                            np.nan
                            if i[0] not in unique_points.index.get_level_values('gridnode').tolist()
                            else unique_points[i[0]]
                            for i in enumerate(self.gridpoints)
                        ]
                    )
                    .reshape(self.grid_dim)
                    .T
                )
                # If specified, save gridded array(s)
                if self.grid_to_raster:
                    gridfile_name = os.path.join(
                        self.workdir, self.col_name + '_' + 'grid_seasonal_absolute_fit_rmse' + '.tif'
                    )
                    save_gridfile(
                        self.grid_seasonal_absolute_fit_rmse,
                        'grid_seasonal_absolute_fit_rmse',
                        gridfile_name,
                        self.plotbbox,
                        self.spacing,
                        self.unit,
                        colorbarfmt='%.2e',
                        stationsongrids=self.stationsongrids,
                        time_lines=self.time_lines,
                        dtype='float32',
                    )

    def _amplitude_and_phase(self, station, tt, yy, min_span=2, min_frac=0.6, period_limit=0.0):
        """
        Fit sin to the input time sequence, and return fitting parameters:
            "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc".
        Minimum time span in years (min_span), minimum fractional observations in span (min_frac),
            and period limit (period_limit) enforced for statistical analysis.
        Source: https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy.
        """
        ampfit = {}
        phsfit = {}
        periodfit = {}
        ampfit_c = {}
        phsfit_c = {}
        periodfit_c = {}
        seasonalfit_rmse = {}
        ampfit[station] = np.nan
        phsfit[station] = np.nan
        periodfit[station] = np.nan
        ampfit_c[station] = np.nan
        phsfit_c[station] = np.nan
        periodfit_c[station] = np.nan
        seasonalfit_rmse[station] = np.nan
        # Fit with custom fit function with fixed period, if specified
        if period_limit != 0.0:
            # convert from years to radians/seconds
            w = (1 / period_limit) * (1 / 31556952) * (2.0 * np.pi)

            def custom_sine_function_base(t, A, p, c):
                return self._sine_function_base(t, A, w, p, c)
        else:

            def custom_sine_function_base(t, A, w, p, c):
                return self._sine_function_base(t, A, w, p, c)

        # If station TS does not span specified time period, pass NaNs
        time_span_yrs = (max(tt) - min(tt)) / 31556952
        if time_span_yrs >= min_span and len(list(set(tt))) / (time_span_yrs * 365.25) >= min_frac:
            tt = np.array(tt)
            yy = np.array(yy)
            ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
            Fyy = abs(np.fft.fft(yy))
            guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero period "peak", which is related to offset
            guess_amp = np.std(yy) * 2.0**0.5
            guess_offset = np.mean(yy)
            guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
            # Adjust frequency guess to reflect fixed period, if specified
            if period_limit != 0.0:
                guess = np.array([guess_amp, 0.0, guess_offset])
            # Catch warning where covariance cannot be estimated
            # I.e. OptimizeWarning: Covariance of the parameters could not be estimated
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                try:
                    optimize_warning = False
                    try:
                        # Note, may have to adjust max number of iterations (maxfev) higher to avoid crashes
                        popt, pcov = optimize.curve_fit(custom_sine_function_base, tt, yy, p0=guess, maxfev=int(1e6))
                    # If sparse input such that fittitng is not possible, pass NaNs
                    except TypeError:
                        (
                            self.ampfit.append(np.nan),
                            self.phsfit.append(np.nan),
                            self.periodfit.append(np.nan),
                            self.ampfit_c.append(np.nan),
                            self.phsfit_c.append(np.nan),
                            self.periodfit_c.append(np.nan),
                            self.seasonalfit_rmse.append(np.nan),
                        )
                        return (
                            self.ampfit,
                            self.phsfit,
                            self.periodfit,
                            self.ampfit_c,
                            self.phsfit_c,
                            self.periodfit_c,
                            self.seasonalfit_rmse,
                        )
                except OptimizeWarning:
                    optimize_warning = True
                    warnings.simplefilter('ignore', OptimizeWarning)
                    popt, pcov = optimize.curve_fit(custom_sine_function_base, tt, yy, p0=guess, maxfev=int(1e6))
                    debug_figure_path = os.path.join(self.workdir, 'phaseamp_per_station', f'station{station}.png')
                    print(
                        f'OptimizeWarning: Covariance for station {station} could not be estimated. '
                        f'Refer to debug figure here {debug_figure_path}'
                    )
                    pass
            # Adjust expected output to reflect fixed period, if specified
            if period_limit != 0.0:
                A, p, c = popt
            else:
                A, w, p, c = popt
            # convert from radians/seconds to years
            f = (w / (2.0 * np.pi)) * (31556952)
            f = 1 / f

            def fitfunc(t):
                return A * np.sin(w * t + p) + c

            # Outputs = "amp": A, "angular frequency": w, "phase": p, "offset": c, "freq": f, "period": 1./f,
            #        "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)
            # Pass amplitude (specified units) and phase (days) and stdev
            ampfit[station] = abs(A)
            # Convert phase from rad to days, apply half wavelength shift if Amp is negative
            if A < 0:
                p += 3.14159
            phsfit[station] = (365.25 / 2) * np.sin(p)
            periodfit[station] = f
            # Catch warning where output is so small that it gets rounded to 0
            # I.e. RuntimeWarning: invalid value encountered in double_scalars
            with np.errstate(invalid='raise'):
                try:
                    # pass covariance for each parameter
                    ampfit_c[station] = pcov[0, 0] ** 0.5
                    periodfit_c[station] = pcov[1, 1] ** 0.5
                    phsfit_c[station] = pcov[2, 2] ** 0.5
                    # pass RMSE of fit
                    seasonalfit_rmse[station] = yy - custom_sine_function_base(tt, *popt)
                    seasonalfit_rmse[station] = (
                        np.sum(seasonalfit_rmse[station] ** 2) / (seasonalfit_rmse[station].size - 2)
                    ) ** 0.5
                except FloatingPointError:
                    pass
            if self.phaseamp_per_station or optimize_warning:
                # Debug plotting for each station
                # convert time (datetime seconds) to absolute years for plotting
                tt_plot = copy.deepcopy(tt)
                tt_plot -= min(tt_plot)
                tt_plot /= 31556952
                plt.plot(tt_plot, yy, 'ok', label='input')
                plt.xlabel('time (years)')
                plt.ylabel(f'data ({self.unit})')
                num_testpoints = len(tt) * 10
                if num_testpoints > 1000:
                    num_testpoints = 1000
                tt2 = np.linspace(min(tt), max(tt), num_testpoints)
                # convert time to years for plotting
                tt2_plot = copy.deepcopy(tt2)
                tt2_plot -= min(tt2_plot)
                tt2_plot /= 31556952
                plt.plot(tt2_plot, fitfunc(tt2), 'r-', label='fit', linewidth=2)
                plt.legend(loc='best')
                if not os.path.exists(os.path.join(self.workdir, 'phaseamp_per_station')):
                    os.mkdir(os.path.join(self.workdir, 'phaseamp_per_station'))
                plt.savefig(
                    os.path.join(self.workdir, 'phaseamp_per_station', f'station{station}.png'),
                    format='png',
                    bbox_inches='tight',
                )
                plt.close()
                optimize_warning = False

        self.ampfit.append(ampfit)
        self.phsfit.append(phsfit)
        self.periodfit.append(periodfit)
        self.ampfit_c.append(ampfit_c)
        self.phsfit_c.append(phsfit_c)
        self.periodfit_c.append(periodfit_c)
        self.seasonalfit_rmse.append(seasonalfit_rmse)

        return (
            self.ampfit,
            self.phsfit,
            self.periodfit,
            self.ampfit_c,
            self.phsfit_c,
            self.periodfit_c,
            self.seasonalfit_rmse,
        )

    def _sine_function_base(self, t, A, w, p, c):
        """Base function for modeling sinusoidal amplitude/phase fits."""
        return A * np.sin(w * t + p) + c

    def __call__(
        self,
        gridarr,
        plottype,
        workdir='./',
        drawgridlines=False,
        colorbarfmt='%.2f',
        stationsongrids=None,
        resValue=5,
        plotFormat='pdf',
        userTitle=None,
    ):
        """Visualize a suite of statistics w.r.t. stations. Pass either a list of points or a gridded array as the first argument. Alternatively, you may superimpose your gridded array with a supplementary list of points by passing the latter through the stationsongrids argument."""
        from cartopy import crs as ccrs
        from cartopy import feature as cfeature
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
        from matplotlib import ticker as mticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # If specified workdir doesn't exist, create it
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        # Pass cbounds
        cbounds = self.cbounds
        # Initiate no-data array to mask data
        nodat_arr = [0, np.nan, np.inf]
        if self.unit in ['minute', 'hour', 'day', 'year']:
            colorbarfmt = '%.1i'
            nodat_arr = [np.nan, np.inf]

        fig, axes = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        # by default set background to white
        axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='white'), zorder=0)
        axes.set_extent(self.plotbbox, ccrs.PlateCarree())
        # add coastlines
        axes.coastlines(linewidth=0.2, color='gray', zorder=4)
        cmap = copy.copy(mpl.cm.get_cmap(self.usr_colormap))
        # cmap.set_bad('black', 0.)
        # extract all colors from the hot map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist)
        axes.set_xlabel('Longitude', weight='bold', zorder=2)
        axes.set_ylabel('Latitude', weight='bold', zorder=2)

        # set ticks
        axes.set_xticks(np.linspace(self.plotbbox[0], self.plotbbox[1], 5), crs=ccrs.PlateCarree())
        axes.set_yticks(np.linspace(self.plotbbox[2], self.plotbbox[3], 5), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='')
        lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='')
        axes.xaxis.set_major_formatter(lon_formatter)
        axes.yaxis.set_major_formatter(lat_formatter)

        # draw central longitude lines corresponding to respective datetimes
        if self.time_lines:
            tl = axes.grid(axis='x', linewidth=1.5, color='blue', alpha=0.5, linestyle='-', zorder=3)

        # If individual stations passed
        if isinstance(gridarr, list):
            # spatial distribution of stations
            if plottype == 'station_distribution':
                im = axes.scatter(
                    gridarr[0], gridarr[1], zorder=1, s=0.5, marker='.', color='b', transform=ccrs.PlateCarree()
                )

            # passing 3rd column as z-value
            if len(gridarr) > 2:
                # set land/water background to light gray/blue respectively so station point data can be seen
                axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='#A9A9A9'), zorder=0)
                axes.add_feature(
                    cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#ADD8E6'), zorder=0
                )
                # set masked values as nans
                zvalues = gridarr[2]
                for i in nodat_arr:
                    zvalues = np.ma.masked_where(zvalues == i, zvalues)
                zvalues = np.ma.filled(zvalues, np.nan)
                # define the bins and normalize
                if cbounds is None:
                    # avoid "ufunc 'isnan'" error by casting array as float
                    cbounds = [
                        np.nanpercentile(zvalues.astype('float'), self.colorpercentile[0]),
                        np.nanpercentile(zvalues.astype('float'), self.colorpercentile[1]),
                    ]
                    # if upper/lower bounds identical, overwrite lower bound as 75% of upper bound to avoid plotting ValueError
                    if cbounds[0] == cbounds[1]:
                        cbounds[0] *= 0.75
                        cbounds.sort()
                    # adjust precision for colorbar if necessary
                    if (abs(np.nanmax(zvalues) - np.nanmin(zvalues)) < 1 and (np.nanmean(zvalues)) < 1) or abs(
                        np.nanmax(zvalues) - np.nanmin(zvalues)
                    ) > 500:
                        colorbarfmt = '%.2e'

                colorbounds = np.linspace(cbounds[0], cbounds[1], 256)
                colorbounds = np.unique(colorbounds)
                norm = mpl.colors.BoundaryNorm(colorbounds, cmap.N)
                colorbounds_ticks = np.linspace(cbounds[0], cbounds[1], 10)

                # plot data and initiate colorbar
                im = axes.scatter(
                    gridarr[0],
                    gridarr[1],
                    c=zvalues,
                    cmap=cmap,
                    norm=norm,
                    zorder=1,
                    s=0.5,
                    marker='.',
                    transform=ccrs.PlateCarree(),
                )
                # initiate colorbar and control height of colorbar
                divider = make_axes_locatable(axes)
                cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
                cbar_ax = fig.colorbar(
                    im,
                    spacing='proportional',
                    ticks=colorbounds_ticks,
                    boundaries=colorbounds,
                    format=colorbarfmt,
                    pad=0.1,
                    cax=cax,
                )
                cbar_ax.ax.minorticks_off()

        # If gridded area passed
        else:
            # set masked values as nans
            for i in nodat_arr:
                gridarr = np.ma.masked_where(gridarr == i, gridarr)
            gridarr = np.ma.filled(gridarr, np.nan)
            # set land/water background to light gray/blue respectively so grid cells can be seen
            axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='#A9A9A9'), zorder=0)
            axes.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#ADD8E6'), zorder=0)
            # define the bins and normalize
            if cbounds is None:
                cbounds = [
                    np.nanpercentile(gridarr, self.colorpercentile[0]),
                    np.nanpercentile(gridarr, self.colorpercentile[1]),
                ]
                # if upper/lower bounds identical, overwrite lower bound as 75% of upper bound to avoid plotting ValueError
                if cbounds[0] == cbounds[1]:
                    cbounds[0] *= 0.75
                    cbounds.sort()
                # plot data and initiate colorbar
                if (abs(np.nanmax(gridarr) - np.nanmin(gridarr)) < 1 and abs(np.nanmean(gridarr)) < 1) or abs(
                    np.nanmax(gridarr) - np.nanmin(gridarr)
                ) > 500:
                    colorbarfmt = '%.2e'

            colorbounds = np.linspace(cbounds[0], cbounds[1], 256)
            colorbounds = np.unique(colorbounds)
            norm = mpl.colors.BoundaryNorm(colorbounds, cmap.N)
            colorbounds_ticks = np.linspace(cbounds[0], cbounds[1], 10)

            # plot data
            im = axes.imshow(
                gridarr,
                cmap=cmap,
                norm=norm,
                extent=self.plotbbox,
                zorder=1,
                origin='upper',
                transform=ccrs.PlateCarree(),
            )
            # initiate colorbar and control height of colorbar
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
            cbar_ax = fig.colorbar(
                im,
                spacing='proportional',
                ticks=colorbounds_ticks,
                boundaries=colorbounds,
                format=colorbarfmt,
                pad=0.1,
                cax=cax,
            )
            cbar_ax.ax.minorticks_off()

            # superimpose your gridded array with a supplementary list of point, if specified
            if self.stationsongrids:
                axes.scatter(
                    self.stationsongrids[0],
                    self.stationsongrids[1],
                    zorder=2,
                    s=0.5,
                    marker='.',
                    color='b',
                    transform=ccrs.PlateCarree(),
                )

            # draw gridlines, if specified
            if drawgridlines:
                gl = axes.gridlines(
                    crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.5, linestyle='-', zorder=3
                )
                gl.xlocator = mticker.FixedLocator(
                    np.arange(self.plotbbox[0], self.plotbbox[1] + self.spacing, self.spacing).tolist()
                )
                gl.ylocator = mticker.FixedLocator(
                    np.arange(self.plotbbox[2], self.plotbbox[3] + self.spacing, self.spacing).tolist()
                )

        # Add labels to colorbar, if necessary
        if 'cbar_ax' in locals():
            # experimental variogram fit sill heatmap
            if plottype == 'grid_variance':
                cbar_ax.set_label(
                    ' '.join(plottype.replace('grid_', '').split('_')).title() + f' ({self.unit}\u00b2)',
                    rotation=-90,
                    labelpad=10,
                )
            # specify appropriate units for mean/median/std/amplitude/experimental variogram fit heatmap
            elif (
                plottype == 'grid_delay_mean'
                or plottype == 'grid_delay_median'
                or plottype == 'grid_delay_stdev'
                or plottype == 'grid_seasonal_amplitude'
                or plottype == 'grid_range'
                or plottype == 'station_delay_mean'
                or plottype == 'station_delay_median'
                or plottype == 'station_delay_stdev'
                or plottype == 'station_seasonal_amplitude'
                or plottype == 'grid_delay_absolute_mean'
                or plottype == 'grid_delay_absolute_median'
                or plottype == 'grid_delay_absolute_stdev'
                or plottype == 'grid_seasonal_absolute_amplitude'
                or plottype == 'grid_seasonal_amplitude_stdev'
                or plottype == 'grid_seasonal_absolute_amplitude_stdev'
                or plottype == 'grid_seasonal_fit_rmse'
                or plottype == 'grid_seasonal_absolute_fit_rmse'
                or plottype == 'grid_variogram_rmse'
            ):
                # update label if sigZTD
                if 'sig' in self.col_name:
                    cbar_ax.set_label(
                        'sig ZTD '
                        + ' '.join(plottype.replace('grid_', '').replace('delay_', '').split('_')).title()
                        + f' ({self.unit})',
                        rotation=-90,
                        labelpad=10,
                    )
                else:
                    cbar_ax.set_label(
                        ' '.join(plottype.replace('grid_', '').split('_')).title() + f' ({self.unit})',
                        rotation=-90,
                        labelpad=10,
                    )
            # specify appropriate units for phase heatmap (days)
            elif (
                plottype == 'station_seasonal_phase'
                or plottype == 'grid_seasonal_phase'
                or plottype == 'grid_seasonal_absolute_phase'
                or plottype == 'grid_seasonal_absolute_phase_stdev'
                or plottype == 'grid_seasonal_phase_stdev'
            ):
                cbar_ax.set_label(
                    ' '.join(plottype.replace('grid_', '').split('_')).title() + ' (days)',
                    rotation=-90,
                    labelpad=10,
                )
            # specify appropriate units for period heatmap (years)
            elif (
                plottype == 'station_delay_period'
                or plottype == 'grid_seasonal_period'
                or plottype == 'grid_seasonal_absolute_period'
                or plottype == 'grid_seasonal_absolute_period_stdev'
                or plottype == 'grid_seasonal_period_stdev'
            ):
                cbar_ax.set_label(
                    ' '.join(plottype.replace('grid_', '').split('_')).title() + ' (years)',
                    rotation=-90,
                    labelpad=10,
                )
            # gridmap of station density has no units
            else:
                cbar_ax.set_label(' '.join(plottype.replace('grid_', '').split('_')).title(), rotation=-90, labelpad=10)

        # Add title to plots, if specified
        if userTitle:
            axes.set_title(userTitle, zorder=2)

        # save/close figure
        # cbar_ax.ax.locator_params(nbins=10)
        # for label in cbar_ax.ax.xaxis.get_ticklabels()[::25]:
        # label.set_visible(False)
        plt.savefig(
            os.path.join(workdir, self.col_name + '_' + plottype + '.' + plotFormat),
            format=plotFormat,
            bbox_inches='tight',
        )
        plt.close()


def stats_analyses(
    fname,
    col_name,
    unit,
    workdir,
    numCPUs,
    verbose,
    bbox,
    spacing,
    timeinterval,
    seasonalinterval,
    obs_errlimit,
    figdpi,
    user_title,
    plot_fmt,
    cbounds,
    colorpercentile,
    usr_colormap,
    densitythreshold,
    stationsongrids,
    drawgridlines,
    time_lines,
    plotall,
    station_distribution,
    station_delay_mean,
    station_delay_median,
    station_delay_stdev,
    station_seasonal_phase,
    phaseamp_per_station,
    grid_heatmap,
    grid_delay_mean,
    grid_delay_median,
    grid_delay_stdev,
    grid_seasonal_phase,
    grid_delay_absolute_mean,
    grid_delay_absolute_median,
    grid_delay_absolute_stdev,
    grid_seasonal_absolute_phase,
    grid_to_raster,
    min_span,
    period_limit,
    variogramplot,
    binnedvariogram,
    variogram_per_timeslice,
    variogram_errlimit,
) -> None:
    """
    Main workflow for generating a suite of plots to illustrate spatiotemporal distribution
    and/or character of zenith delays.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Control DPI for output figures
    mpl.rcParams['savefig.dpi'] = figdpi

    # If user requests to generate all plots.
    if plotall:
        logger.info('"-plotall" == True. All plots will be made.')
        station_distribution = True
        station_delay_mean = True
        station_delay_median = True
        station_delay_stdev = True
        station_seasonal_phase = True
        grid_heatmap = True
        grid_delay_mean = True
        grid_delay_median = True
        grid_delay_stdev = True
        grid_seasonal_phase = True
        grid_delay_absolute_mean = True
        grid_delay_absolute_median = True
        grid_delay_absolute_stdev = True
        grid_seasonal_absolute_phase = True
        variogramplot = True

    logger.info('***Stats Function:***')
    # prep dataframe object for plotting/variogram analysis based off of user specifications
    df_stats = RaiderStats(
        fname,
        col_name,
        unit,
        workdir,
        bbox,
        spacing,
        timeinterval,
        seasonalinterval,
        obs_errlimit,
        time_lines,
        stationsongrids,
        station_seasonal_phase,
        cbounds,
        colorpercentile,
        usr_colormap,
        grid_heatmap,
        grid_delay_mean,
        grid_delay_median,
        grid_delay_stdev,
        grid_seasonal_phase,
        grid_delay_absolute_mean,
        grid_delay_absolute_median,
        grid_delay_absolute_stdev,
        grid_seasonal_absolute_phase,
        grid_to_raster,
        min_span,
        period_limit,
        numCPUs,
        phaseamp_per_station,
    )

    # Station plots
    # Plot each individual station
    if station_distribution:
        logger.info('- Plot spatial distribution of stations.')
        unique_points = df_stats.df.groupby(['Lon', 'Lat']).size()
        df_stats(
            [
                unique_points.index.get_level_values('Lon').tolist(),
                unique_points.index.get_level_values('Lat').tolist(),
            ],
            'station_distribution',
            workdir=os.path.join(workdir, 'figures'),
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean delay per station
    if station_delay_mean:
        logger.info('- Plot mean delay for each station.')
        unique_points = df_stats.df.groupby(['Lon', 'Lat'])[col_name].mean()
        unique_points.dropna(how='any', inplace=True)
        df_stats(
            [
                unique_points.index.get_level_values('Lon').tolist(),
                unique_points.index.get_level_values('Lat').tolist(),
                unique_points.values,
            ],
            'station_delay_mean',
            workdir=os.path.join(workdir, 'figures'),
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot median delay per station
    if station_delay_median:
        logger.info('- Plot median delay for each station.')
        unique_points = df_stats.df.groupby(['Lon', 'Lat'])[col_name].median()
        unique_points.dropna(how='any', inplace=True)
        df_stats(
            [
                unique_points.index.get_level_values('Lon').tolist(),
                unique_points.index.get_level_values('Lat').tolist(),
                unique_points.values,
            ],
            'station_delay_median',
            workdir=os.path.join(workdir, 'figures'),
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot delay stdev per station
    if station_delay_stdev:
        logger.info('- Plot delay stdev for each station.')
        unique_points = df_stats.df.groupby(['Lon', 'Lat'])[col_name].std()
        unique_points.dropna(how='any', inplace=True)
        df_stats(
            [
                unique_points.index.get_level_values('Lon').tolist(),
                unique_points.index.get_level_values('Lat').tolist(),
                unique_points.values,
            ],
            'station_delay_stdev',
            workdir=os.path.join(workdir, 'figures'),
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot delay phase/amplitude per station
    if station_seasonal_phase:
        logger.info('- Plot delay phase/amplitude for each station.')
        # phase
        unique_points_phase = df_stats.df.groupby(['Lon', 'Lat'])['phsfit'].mean()
        unique_points_phase.dropna(how='any', inplace=True)
        df_stats(
            [
                unique_points_phase.index.get_level_values('Lon').tolist(),
                unique_points_phase.index.get_level_values('Lat').tolist(),
                unique_points_phase.values,
            ],
            'station_seasonal_phase',
            workdir=os.path.join(workdir, 'figures'),
            colorbarfmt='%.1i',
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
        # amplitude
        unique_points_amplitude = df_stats.df.groupby(['Lon', 'Lat'])['ampfit'].mean()
        unique_points_amplitude.dropna(how='any', inplace=True)
        df_stats(
            [
                unique_points_amplitude.index.get_level_values('Lon').tolist(),
                unique_points_amplitude.index.get_level_values('Lat').tolist(),
                unique_points_amplitude.values,
            ],
            'station_seasonal_amplitude',
            workdir=os.path.join(workdir, 'figures'),
            colorbarfmt='%.3f',
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
        # period
        unique_points_period = df_stats.df.groupby(['Lon', 'Lat'])['periodfit'].mean()
        df_stats(
            [
                unique_points_period.index.get_level_values('Lon').tolist(),
                unique_points_period.index.get_level_values('Lat').tolist(),
                unique_points_period.values,
            ],
            'station_delay_period',
            workdir=os.path.join(workdir, 'figures'),
            colorbarfmt='%.2f',
            plotFormat=plot_fmt,
            userTitle=user_title,
        )

    # Gridded station plots
    # Plot density of stations for each gridcell
    if isinstance(df_stats.grid_heatmap, np.ndarray):
        logger.info('- Plot density of stations per gridcell.')
        df_stats(
            df_stats.grid_heatmap,
            'grid_heatmap',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.1i',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of station-wise mean delay across each gridcell
    if isinstance(df_stats.grid_delay_mean, np.ndarray):
        logger.info('- Plot mean of station-wise mean delay across each gridcell.')
        df_stats(
            df_stats.grid_delay_mean,
            'grid_delay_mean',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of station-wise median delay across each gridcell
    if isinstance(df_stats.grid_delay_median, np.ndarray):
        logger.info('- Plot mean of station-wise median delay across each gridcell.')
        df_stats(
            df_stats.grid_delay_median,
            'grid_delay_median',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of station-wise stdev delay across each gridcell
    if isinstance(df_stats.grid_delay_stdev, np.ndarray):
        logger.info('- Plot mean of station-wise stdev delay across each gridcell.')
        df_stats(
            df_stats.grid_delay_stdev,
            'grid_delay_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of station-wise delay phase across each gridcell
    if isinstance(df_stats.grid_seasonal_phase, np.ndarray):
        logger.info('- Plot mean of station-wise delay phase across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_phase,
            'grid_seasonal_phase',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.1i',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of station-wise delay amplitude across each gridcell
    if isinstance(df_stats.grid_seasonal_amplitude, np.ndarray):
        logger.info('- Plot mean of station-wise delay amplitude across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_amplitude,
            'grid_seasonal_amplitude',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.3f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of station-wise delay period across each gridcell
    if isinstance(df_stats.grid_seasonal_period, np.ndarray):
        logger.info('- Plot mean of station-wise delay period across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_period,
            'grid_seasonal_period',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean stdev of station-wise delay phase across each gridcell
    if isinstance(df_stats.grid_seasonal_phase_stdev, np.ndarray):
        logger.info('- Plot mean stdev of station-wise delay phase across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_phase_stdev,
            'grid_seasonal_phase_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.1i',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean stdev of station-wise delay amplitude across each gridcell
    if isinstance(df_stats.grid_seasonal_amplitude_stdev, np.ndarray):
        logger.info('- Plot mean stdev of station-wise delay amplitude across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_amplitude_stdev,
            'grid_seasonal_amplitude_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.3f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean stdev of station-wise delay period across each gridcell
    if isinstance(df_stats.grid_seasonal_period_stdev, np.ndarray):
        logger.info('- Plot mean stdev of station-wise delay period across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_period_stdev,
            'grid_seasonal_period_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2e',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot mean of seasonal fit RMSE across each gridcell
    if isinstance(df_stats.grid_seasonal_fit_rmse, np.ndarray):
        logger.info('- Plot mean of seasonal fit RMSE across each gridcell.')
        df_stats(
            df_stats.grid_seasonal_fit_rmse,
            'grid_seasonal_fit_rmse',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.3f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute mean delay for each gridcell
    if isinstance(df_stats.grid_delay_absolute_mean, np.ndarray):
        logger.info('- Plot absolute mean delay per gridcell.')
        df_stats(
            df_stats.grid_delay_absolute_mean,
            'grid_delay_absolute_mean',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute median delay for each gridcell
    if isinstance(df_stats.grid_delay_absolute_median, np.ndarray):
        logger.info('- Plot absolute median delay per gridcell.')
        df_stats(
            df_stats.grid_delay_absolute_median,
            'grid_delay_absolute_median',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute stdev delay for each gridcell
    if isinstance(df_stats.grid_delay_absolute_stdev, np.ndarray):
        logger.info('- Plot absolute delay stdev per gridcell.')
        df_stats(
            df_stats.grid_delay_absolute_stdev,
            'grid_delay_absolute_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute delay phase for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_phase, np.ndarray):
        logger.info('- Plot absolute delay phase per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_phase,
            'grid_seasonal_absolute_phase',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.1i',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute delay amplitude for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_amplitude, np.ndarray):
        logger.info('- Plot absolute delay amplitude per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_amplitude,
            'grid_seasonal_absolute_amplitude',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.3f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute delay period for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_period, np.ndarray):
        logger.info('- Plot absolute delay period per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_period,
            'grid_seasonal_absolute_period',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute delay phase stdev for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_phase_stdev, np.ndarray):
        logger.info('- Plot absolute delay phase stdev per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_phase_stdev,
            'grid_seasonal_absolute_phase_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.1i',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute delay amplitude stdev for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_amplitude_stdev, np.ndarray):
        logger.info('- Plot absolute delay amplitude stdev per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_amplitude_stdev,
            'grid_seasonal_absolute_amplitude_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.3f',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute delay period stdev for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_period_stdev, np.ndarray):
        logger.info('- Plot absolute delay period stdev per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_period_stdev,
            'grid_seasonal_absolute_period_stdev',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2e',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    # Plot absolute mean seasonal fit RMSE for each gridcell
    if isinstance(df_stats.grid_seasonal_absolute_fit_rmse, np.ndarray):
        logger.info('- Plot absolute mean seasonal fit RMSE per gridcell.')
        df_stats(
            df_stats.grid_seasonal_absolute_fit_rmse,
            'grid_seasonal_absolute_fit_rmse',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2e',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )

    # Perform variogram analysis
    if (
        variogramplot
        and not isinstance(df_stats.grid_range, np.ndarray)
        and not isinstance(df_stats.grid_variance, np.ndarray)
        and not isinstance(df_stats.grid_variogram_rmse, np.ndarray)
    ):
        logger.info('***Variogram Analysis Function:***')
        if unit in ['minute', 'hour', 'day', 'year']:
            unit = 'm'
            df_stats.unit = 'm'
            logger.warning(f'Output unit {unit} specified for Variogram analysis. Reverted to meters')
        make_variograms = VariogramAnalysis(
            df_stats.df,
            df_stats.gridpoints,
            col_name,
            unit,
            workdir,
            df_stats.seasonalinterval,
            densitythreshold,
            binnedvariogram,
            numCPUs,
            variogram_per_timeslice,
            variogram_errlimit,
        )
        TOT_grids, TOT_res_robust_arr, TOT_res_robust_rmse = make_variograms.create_variograms()
        # get range
        df_stats.grid_range = (
            np.array(
                [
                    np.nan if i[0] not in TOT_grids else float(TOT_res_robust_arr[TOT_grids.index(i[0])][0])
                    for i in enumerate(df_stats.gridpoints)
                ]
            )
            .reshape(df_stats.grid_dim)
            .T
        )
        # convert range to specified output unit
        df_stats.grid_range = convert_SI(df_stats.grid_range, 'm', unit)
        # get sill
        df_stats.grid_variance = (
            np.array(
                [
                    np.nan if i[0] not in TOT_grids else float(TOT_res_robust_arr[TOT_grids.index(i[0])][1])
                    for i in enumerate(df_stats.gridpoints)
                ]
            )
            .reshape(df_stats.grid_dim)
            .T
        )
        # convert sill to specified output unit
        df_stats.grid_range = convert_SI(df_stats.grid_range, 'm^2', unit.split('^2')[0] + '^2')
        # get variogram rmse
        df_stats.grid_variogram_rmse = (
            np.array(
                [
                    np.nan if i[0] not in TOT_grids else float(TOT_res_robust_rmse[TOT_grids.index(i[0])])
                    for i in enumerate(df_stats.gridpoints)
                ]
            )
            .reshape(df_stats.grid_dim)
            .T
        )
        # convert range to specified output unit
        df_stats.grid_variogram_rmse = convert_SI(df_stats.grid_variogram_rmse, 'm', unit)
        # If specified, save gridded array(s)
        if grid_to_raster:
            # write range
            gridfile_name = os.path.join(workdir, col_name + '_' + 'grid_range' + '.tif')
            save_gridfile(
                df_stats.grid_range,
                'grid_range',
                gridfile_name,
                df_stats.plotbbox,
                df_stats.spacing,
                df_stats.unit,
                colorbarfmt='%1i',
                stationsongrids=df_stats.stationsongrids,
                dtype='float32',
            )
            # write sill
            gridfile_name = os.path.join(workdir, col_name + '_' + 'grid_variance' + '.tif')
            save_gridfile(
                df_stats.grid_variance,
                'grid_variance',
                gridfile_name,
                df_stats.plotbbox,
                df_stats.spacing,
                df_stats.unit + '^2',
                colorbarfmt='%.3e',
                stationsongrids=df_stats.stationsongrids,
                dtype='float32',
            )
            # write variogram rmse
            gridfile_name = os.path.join(workdir, col_name + '_' + 'grid_variogram_rmse' + '.tif')
            save_gridfile(
                df_stats.grid_variogram_rmse,
                'grid_variogram_rmse',
                gridfile_name,
                df_stats.plotbbox,
                df_stats.spacing,
                df_stats.unit,
                colorbarfmt='%.2e',
                stationsongrids=df_stats.stationsongrids,
                dtype='float32',
            )

    if isinstance(df_stats.grid_range, np.ndarray):
        # plot range heatmap
        logger.info('- Plot variogram range per gridcell.')
        df_stats(
            df_stats.grid_range,
            'grid_range',
            workdir=os.path.join(workdir, 'figures'),
            colorbarfmt='%1i',
            drawgridlines=drawgridlines,
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    if isinstance(df_stats.grid_variance, np.ndarray):
        # plot sill heatmap
        logger.info('- Plot variogram sill per gridcell.')
        df_stats(
            df_stats.grid_variance,
            'grid_variance',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.3e',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )
    if isinstance(df_stats.grid_variogram_rmse, np.ndarray):
        # plot variogram rmse heatmap
        logger.info('- Plot variogram RMSE per gridcell.')
        df_stats(
            df_stats.grid_variogram_rmse,
            'grid_variogram_rmse',
            workdir=os.path.join(workdir, 'figures'),
            drawgridlines=drawgridlines,
            colorbarfmt='%.2e',
            stationsongrids=stationsongrids,
            plotFormat=plot_fmt,
            userTitle=user_title,
        )


def main() -> None:
    inps = cmd_line_parse()

    stats_analyses(
        inps.fname,
        inps.col_name,
        inps.unit,
        inps.workdir,
        inps.cpus,
        inps.verbose,
        inps.bounding_box,
        inps.spacing,
        inps.timeinterval,
        inps.seasonalinterval,
        inps.obs_errlimit,
        inps.figdpi,
        inps.user_title,
        inps.plot_fmt,
        inps.cbounds,
        inps.colorpercentile,
        inps.usr_colormap,
        inps.densitythreshold,
        inps.stationsongrids,
        inps.drawgridlines,
        inps.time_lines,
        inps.plotall,
        inps.station_distribution,
        inps.station_delay_mean,
        inps.station_delay_median,
        inps.station_delay_stdev,
        inps.station_seasonal_phase,
        inps.phaseamp_per_station,
        inps.grid_heatmap,
        inps.grid_delay_mean,
        inps.grid_delay_median,
        inps.grid_delay_stdev,
        inps.grid_seasonal_phase,
        inps.grid_delay_absolute_mean,
        inps.grid_delay_absolute_median,
        inps.grid_delay_absolute_stdev,
        inps.grid_seasonal_absolute_phase,
        inps.grid_to_raster,
        inps.min_span,
        inps.period_limit,
        inps.variogramplot,
        inps.binnedvariogram,
        inps.variogram_per_timeslice,
        inps.variogram_errlimit,
    )
