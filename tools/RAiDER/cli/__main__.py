import argparse
import sys
from importlib.metadata import entry_points
from pathlib import Path

import RAiDER.cli.conf as conf


def main() -> None:
    parser = argparse.ArgumentParser(prefix_chars='+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '++process',
        choices=['calcDelays', 'downloadGNSS', 'calcDelaysGUNW'],
        default='calcDelays',
        help='Select the entrypoint to use',
    )
    parser.add_argument(
        '++logger_path',
        required=False,
        default=None,
        help='Set the path to write the log files',
    )

    args, unknowns = parser.parse_known_args()

    # Needed for a global logging path
    logger_path = Path(args.logger_path) if args.logger_path else None
    conf.setLoggerPath(logger_path)

    sys.argv = [args.process, *unknowns]

    try:
        # python >=3.10 interface
        process_entry_point = next(iter(entry_points(group='console_scripts', name=f'{args.process}.py')), None)

    except TypeError:
        # python 3.8 and 3.9 interface
        scripts = entry_points()['console_scripts']
        process_entry_point = [ep for ep in scripts if ep.name == f'{args.process}.py'][0]

    process_entry_point.load()()


if __name__ == '__main__':
    main()
