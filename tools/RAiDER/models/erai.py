import datetime

from RAiDER.models.ecmwf import ECMWF
from RAiDER.models.model_levels import A_ERAI, B_ERAI


class ERAI(ECMWF):
    # A and B parameters to calculate pressures for model levels,
    #  extracted from an ECMWF ERA-Interim GRIB file and then hardcoded here
    def __init__(self):
        ECMWF.__init__(self)
        self._classname = 'ei'
        self._expver = '0001'
        self._dataset = 'interim'
        self._Name = 'ERA-I'
        self.setLevelType('ml')

        # Tuple of min/max years where data is available.
        self._valid_range = (
            datetime.datetime(1979, 1, 1),
            datetime.datetime(2019, 8, 31)
        )

        self._lag_time = datetime.timedelta(days=30)  # Availability lag time in days
        self._a = A_ERAI
        self._b = B_ERAI

    def __pressure_levels__(self):
        raise RuntimeError('ERA-I does not use pressure levels, you need to use model levels')
