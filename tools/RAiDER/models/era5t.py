import datetime

from RAiDER.models.era5 import ERA5


class ERA5T(ERA5):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        ERA5.__init__(self)

        self._expver = '0005'
        self._dataset = 'era5t'
        self._Name = 'ERA-5T'

        self._valid_range = (datetime.datetime(1950, 1, 1), "Present")  # Tuple of min/max years where data is available.
        # Availability lag time in days; actually about 12 hours but unstable on ECMWF side
            # https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
            # see data update frequency
        self._lag_time = datetime.timedelta(days=1)
