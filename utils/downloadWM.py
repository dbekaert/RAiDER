"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _STEP meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity.
"""


from osgeo import gdal
gdal.UseExceptions()

# standard imports
import dask 
import itertools
import numpy as np
import os
import pyproj
import tempfile
import queue
import threading

# local imports
import constants as const
import demdownload
import losreader
import util


def tropo_delay(los = None, lat = None, lon = None, 
                heights = None, 
                weather = None, 
                zref = 15000, 
                out = None, 
                time = None,
                outformat='ENVI', 
                parallel=True,
                verbose = False):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    _tropo_delay_with_values. Then we'll write the output to the output
    file.
    """
    if out is None:
        out = os.getcwd()

    # Make weather
    weather_type = weather['type']
    weather_files = weather['files']
    weather_fmt = weather['name']

    # Lat, lon
    if lat is None:
        # They'll get set later with weather
        lats = lons = None
        latproj = lonproj = None
    else:
        lats, latproj = util.gdal_open(lat, returnProj = True)
        lons, lonproj = util.gdal_open(lon, returnProj = True)

    if weather_type == 'wrf':
        import wrf
        weather = wrf.WRF()
        weather.load(*weather_files)

        # Let lats and lons to weather model nodes if necessary
        #TODO: Need to fix the case where lats are None, because
        # the weather model need not be in latlong projection
        if lats is None:
            lats, lons = wrf.wm_nodes(*weather_files)
    elif weather_type == 'pickle':
        weather = util.pickle_load('weatherObj.dat')
    else:
        weather_model = weather_type
        if weather_files is None:
            if lats is None:
                raise ValueError(
                    'Unable to infer lats and lons if you also want me to '
                    'download the weather model')
            if verbose:
                f = os.path.join(out, 'weather_model.dat')
                weather_model.fetch(lats, lons, time, f)
                weather_model.load(f)
                weather = weather_model # Need to maintain backwards compatibility at the moment
                print(weather)
                p = weather.plot()


    return hydro_file_name, wet_file_name
