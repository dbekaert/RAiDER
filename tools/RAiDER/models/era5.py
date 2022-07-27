from pathlib import Path

import datetime as dt
from pyproj import CRS
from RAiDER.models.ecmwf import ECMWF
from RAiDER.logger import logger


class ERA5(ECMWF):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        super().__init__()

        self._humidityType = 'q'
        self._expver = '0001'
        self._classname = 'ea'
        self._dataset = 'era5'
        self._Name = 'ERA-5'
        self._proj = CRS.from_epsg(4326)

        # Tuple of min/max years where data is available.
        self._valid_range = (dt.datetime(1950, 1, 1), "Present")
        # Availability lag time in days
        self._lag_time = dt.timedelta(days=30)

        # Default, need to change to ml
        self.setLevelType('pl')

    def _fetch(self, lats, lons, times, out: Path, n_extra=2):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, n_extra)

        # execute the search at ECMWF
        try:
            self._get_from_cds(lat_min, lat_max, lon_min, lon_max, times, out)
        except Exception as e:
            logger.warning(e)
            raise RuntimeError('Could not access or download from the CDS API')

    def load_weather(self, filename: Path, *args, **kwargs):
        '''Load either pressure or model level data'''
        if self._model_level_type == 'pl':
            self._load_pressure_level(filename, *args, **kwargs)
        elif self._model_level_type == 'ml':
            self._load_model_level(filename, *args, **kwargs)
        else:
            raise RuntimeError(
                '{} is not a valid model type'.format(self._model_level_type)
            )
