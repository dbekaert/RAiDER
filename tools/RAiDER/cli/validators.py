import itertools
from argparse import Action, ArgumentError, ArgumentTypeError
from datetime import date, time, timedelta
from time import strptime


class MappingType(object):
    """
    A type that maps arguments to constants.

    # Example
    ```
    mapping = MappingType(foo=42, bar="baz").default(None)
    assert mapping("foo") == 42
    assert mapping("bar") == "baz"
    assert mapping("hello") is None
    ```
    """
    UNSET = object()

    def __init__(self, **kwargs):
        self.mapping = kwargs
        self._default = self.UNSET

    def default(self, default):
        """Set a default value if no mapping is found"""
        self._default = default
        return self

    def __call__(self, arg):
        if arg in self.mapping:
            return self.mapping[arg]

        if self._default is self.UNSET:
            raise KeyError(
                "Invalid choice '{}', must be one of {}".format(
                    arg, list(self.mapping.keys())
                )
            )

        return self._default


class IntegerType(object):
    """
    A type that converts arguments to integers.

    # Example
    ```
    integer = IntegerType(0, 100)
    assert integer("0") == 0
    assert integer("100") == 100
    integer("-10")  # Raises exception
    ```
    """

    def __init__(self, lo=None, hi=None):
        self.lo = lo
        self.hi = hi

    def __call__(self, arg):
        integer = int(arg)

        if self.lo is not None and integer < self.lo:
            raise ArgumentTypeError("Must be greater than {}".format(self.lo))
        if self.hi is not None and integer > self.hi:
            raise ArgumentTypeError("Must be less than {}".format(self.hi))

        return integer


class IntegerMappingType(MappingType, IntegerType):
    """
    An integer type that converts non-integer types through a mapping.

    # Example
    ```
    integer = IntegerMappingType(0, 100, random=42)
    assert integer("0") == 0
    assert integer("100") == 100
    assert integer("random") == 42
    ```
    """

    def __init__(self, lo=None, hi=None, mapping={}, **kwargs):
        IntegerType.__init__(self, lo, hi)
        kwargs.update(mapping)
        MappingType.__init__(self, **kwargs)

    def __call__(self, arg):
        try:
            return IntegerType.__call__(self, arg)
        except ValueError:
            return MappingType.__call__(self, arg)


class DateListAction(Action):
    """An Action that parses and stores a list of dates"""

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None
    ):
        if type is not date_type:
            raise ValueError("type must be `date_type`!")

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 2 or not values:
            raise ArgumentError(self, "Only 1 or 2 dates may be supplied")

        if len(values) == 2:
            start, end = values
            values = [start + timedelta(days=k)
                      for k in range((end - start).days + 1)]

        setattr(namespace, self.dest, values)


class BBoxAction(Action):
    """An Action that parses and stores a valid bounding box"""

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None
    ):
        if nargs != 4:
            raise ValueError("nargs must be 4!")

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar
        )

    def __call__(self, parser, namespace, values, option_string=None):
        S, N, W, E = values

        if N <= S or E <= W:
            raise ArgumentError(self, 'Bounding box has no size; make sure you use "S N W E"')

        for sn in (S, N):
            if sn < -90 or sn > 90:
                raise ArgumentError(self, 'Lats are out of S/N bounds')

        for we in (W, E):
            if we < -180 or we > 180:
                raise ArgumentError(self, 'Lons are out of W/E bounds')

        setattr(namespace, self.dest, values)


class LOSAction(Action):
    '''
    An Action that checks for and parses a valid method for computing line-of-sight
    vectors, and stores the result as a generator object
    '''

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None
    ):

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            los = Zenith
        else:
            try:
                generator = parseAsRaster(values)
            except AttributeError:
                pass

            try:
                generator = parseAsStateVectorFile(values)
            except AttributeError:
                pass

           #TODO: Add in ARIA file reader

            raise ValueError('The format of the line-of-sight file(s) is not recognized')
          
        setattr(namespace, self.dest, values)

def date_type(arg):
    """
    Parse a date from a string in pseudo-ISO 8601 format.
    """
    year_formats = (
        '%Y-%m-%d',
        '%Y%m%d'
    )

    for yf in year_formats:
        try:
            return date(*strptime(arg, yf)[0:3])
        except ValueError:
            pass

    raise ArgumentTypeError(
        'Unable to coerce {} to a date. Try %Y-%m-%d'.format(arg)
    )


def time_type(arg):
    '''
    Parse an input time (required to be ISO 8601)
    '''
    time_formats = (
        '',
        'T%H:%M:%S.%f',
        'T%H%M%S.%f',
        '%H%M%S.%f',
        'T%H:%M:%S',
        '%H:%M:%S',
        'T%H%M%S',
        '%H%M%S',
        'T%H:%M',
        'T%H%M',
        '%H:%M',
        'T%H',
    )
    timezone_formats = (
        '',
        'Z',
        '%z',
    )
    all_formats = map(
        ''.join,
        itertools.product(time_formats, timezone_formats)
    )

    for tf in all_formats:
        try:
            return time(*strptime(arg, tf)[3:6])
        except ValueError:
            pass

    raise ArgumentTypeError(
        'Unable to coerce {} to a time. Try T%H:%M:%S'.format(arg)
    )
