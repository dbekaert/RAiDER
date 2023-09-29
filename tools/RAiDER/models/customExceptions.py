class DatetimeFailed(Exception):
    def __init__(self, model, time):
        msg = f"Weather model {model} failed to download for datetime {time}"
        super().__init__(msg)


class DatetimeNotAvailable(Exception):
    def __init__(self, model, time):
        msg = f"Weather model {model} was not found for datetime {time}"
        super().__init__(msg)


class DatetimeOutsideRange(Exception):
    def __init__(self, model, time):
        msg = f"Time {time} is outside the available date range for weather model {model}"
        super().__init__(msg)


class ExistingWeatherModelTooSmall(Exception):
    def __init__(self):
        msg = 'The weather model passed does not cover all of the input ' \
                'points; you may need to download a larger area.'
        super().__init__(msg)


class TryToKeepGoingError(Exception):
    def __init__(self, date=None):
        if date is not None:
            msg = 'The weather model does not exist for date {date}, so I will try to use the closest available date.'
        else:
            msg = 'I will try to keep going'
        super().__init__(msg)
    
class CriticalError(Exception):
    def __init__(self):
        msg = 'I have experienced a critical error, please take a look at the log files'
        super().__init__(msg)

class WrongNumberOfFiles(Exception):
    def __init__(self, Nexp, Navail):
        msg = 'The number of files downloaded does not match the requested, '
        'I expected {} and got {}, aborting'.format(Nexp, Navail)
        super().__init__(msg)
    

class NoWeatherModelData(Exception):
    def __init__(self, custom_msg=None):
        if custom_msg is None:
            msg = 'No weather model files were available to download, aborting'
        else:
            msg = custom_msg
        super().__init__(msg)

