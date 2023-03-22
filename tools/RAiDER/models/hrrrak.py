import datetime

from RAiDER.models.hrrr import HRRR

class HRRRAK(HRRR):
    def __init__(self):
        # initialize a weather model
        super().__init__()
        self._classname = 'hrrrak'
        self._dataset = 'hrrrak'
        self._Name = "HRRR-AK"
        self._valid_range = (datetime.datetime(2018, 7, 13), "Present")

    def _fetch(self,  out):
        '''
        Fetch weather model data from HRRR
        '''
        self._download_hrrr_file(self._time, out, model='hrrrak')