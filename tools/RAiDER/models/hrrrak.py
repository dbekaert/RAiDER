import datetime

from pyproj import CRS
from shapely.geometry import Polygon

from RAiDER.models.hrrr import HRRR

class HRRRAK(HRRR):
    def __init__(self):
        # initialize a weather model
        super().__init__()
        self._classname = 'hrrrak'
        self._dataset = 'hrrrak'
        self._Name = "HRRR-AK"
        self._valid_range = (datetime.datetime(2018, 7, 13), "Present")
        
        # Projection information
        # This will get updated based on the downloaded file
        self._proj = CRS.from_string(
            '+proj=stere +ellps=sphere +a=6371229.0 +b=6371229.0 +lat_0=90 +lon_0=225.0 ' +
            '+x_0=0.0 +y_0=0.0 +lat_ts=60.0 +no_defs +type=crs'
        )

        # This uses 0-360 longitudes instead of -180-180
        self._valid_bounds =  Polygon(((195, 40), (157, 55), (260, 77), (232, 52)))


    def _fetch(self,  out):
        '''
        Fetch weather model data from HRRR
        '''
        self._download_hrrr_file(self._time, out, model='hrrrak')