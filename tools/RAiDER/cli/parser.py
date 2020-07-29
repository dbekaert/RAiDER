import os

from RAiDER.cli.validators import BBoxAction, IntegerMappingType


def add_cpus(parser):
    parser.add_argument(
        '--cpus',
        help='The number of cpus to be used for multiprocessing or "all" for '
             'all available cpus.',
        type=IntegerMappingType(0, all=os.cpu_count()),
        default=8,
    )


def add_verbose(parser):
    parser.add_argument(
        '--verbose', '-v',
        help='Run in verbose mode',
        action='count',
        default=0
    )


def add_out(parser):
    parser.add_argument(
        '--out',
        help='Output directory',
        default='.'
    )


def add_bbox(parser):
    parser.add_argument(
        '--bbox', '-b',
        help="Bounding box",
        nargs=4,
        type=float,
        action=BBoxAction,
        metavar=('S', 'N', 'W', 'E')
    )
