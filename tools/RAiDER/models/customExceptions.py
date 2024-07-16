class DatetimeFailed(Exception):
    def __init__(self, model, time) -> None:
        msg = f"Weather model {model} failed to download for datetime {time}"
        super().__init__(msg)


class DatetimeNotAvailable(Exception):
    def __init__(self, model, time) -> None:
        msg = f"Weather model {model} was not found for datetime {time}"
        super().__init__(msg)


class DatetimeOutsideRange(Exception):
    def __init__(self, model, time) -> None:
        msg = f"Time {time} is outside the available date range for weather model {model}"
        super().__init__(msg)


class ExistingWeatherModelTooSmall(Exception):
    def __init__(self) -> None:
        msg = 'The weather model passed does not cover all of the input ' \
                'points; you may need to download a larger area.'
        super().__init__(msg)


class TryToKeepGoingError(Exception):
    def __init__(self, date=None) -> None:
        if date is not None:
            msg = 'The weather model does not exist for date {date}, so I will try to use the closest available date.'
        else:
            msg = 'I will try to keep going'
        super().__init__(msg)
    
class CriticalError(Exception):
    def __init__(self) -> None:
        msg = 'I have experienced a critical error, please take a look at the log files'
        super().__init__(msg)

class WrongNumberOfFiles(Exception):
    def __init__(self, Nexp, Navail) -> None:
        msg = 'The number of files downloaded does not match the requested, '
        f'I expected {Nexp} and got {Navail}, aborting'
        super().__init__(msg)
    

class NoWeatherModelData(Exception):
    def __init__(self, custom_msg=None) -> None:
        if custom_msg is None:
            msg = 'No weather model files were available to download, aborting'
        else:
            msg = custom_msg
        super().__init__(msg)


class NoStationDataFoundError(Exception):
    def __init__(self, station_list=None, years=None) -> None:
        if (station_list is None) and (years is None):
            msg = 'No GNSS station data was found'
        elif (years is None):
            msg = f'No data was found for GNSS stations {station_list}'
        elif station_list is None:
            msg = f'No data was found for years {years}'
        else:
            msg = f'No data was found for GNSS stations {station_list} and years {years}'

        super().__init__(msg)
