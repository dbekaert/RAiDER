import argparse
import sys
from importlib.metadata import entry_points

from RAiDER.cli.raider import calcDelays, downloadGNSS, calcDelaysGUNW


def main():
    parser = argparse.ArgumentParser(
        prefix_chars='+',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '++process', choices=['calcDelays', 'downloadGNSS', 'calcDelaysGUNW'],
                     default='calcDelays',
                     help='Select the entrypoint to use'
    )
    args, unknowns = parser.parse_known_args()
    sys.argv = [args.process, *unknowns]

    try:
        # python >=3.10 interface
        process_entry_point = entry_points(group='console_scripts', name=f'{args.process}.py')[0]
    except TypeError:
        # python 3.8 and 3.9 interface
        scripts = entry_points()['console_scripts']
        process_entry_point = [ep for ep in scripts if ep.name == f'{args.process}.py'][0]

    process_entry_point.load()()


if __name__ == '__main__':
    main()
