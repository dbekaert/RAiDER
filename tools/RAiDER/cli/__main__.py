import argparse
import os
from importlib.metadata import entry_points

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
    os.sys.argv = [args.process, *unknowns]

    process_entry_point = entry_points(group='console_scripts', name=f'{args.process}.py')[0]
    process_entry_point.load()()


if __name__ == '__main__':
    main()