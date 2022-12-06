import argparse
import os
import shutil
import sys
import yaml
import re, glob

import RAiDER
from RAiDER.constants import _ZREF, _CUBE_SPACING_IN_M
from RAiDER.logger import logger, logging
from RAiDER.cli.validators import (enforce_time, enforce_bbox, parse_dates,
                            get_query_region, get_heights, get_los, enforce_wm)

from RAiDER.checkArgs import checkArgs
from RAiDER.delay import main as main_delay



class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DEFAULT_DICT = dict(
        look_dir='right',
        date_start=None,
        date_end=None,
        date_step=None,
        date_list=None,
        time=None,
        end_time=None,
        weather_model=None,
        lat_file=None,
        lon_file=None,
        station_file=None,
        bounding_box=None,
        geocoded_file=None,
        dem=None,
        use_dem_latlon=False,
        height_levels=None,
        height_file_rdr=None,
        ray_trace=False,
        zref=_ZREF,
        cube_spacing_in_m=_CUBE_SPACING_IN_M,  # TODO - Where are these parsed?
        los_file=None,
        los_convention='isce',
        los_cube=None,
        orbit_file=None,
        verbose=True,
        raster_format='GTiff',
        output_directory=os.getcwd(),
        weather_model_directory=os.path.join(
            os.getcwd(),
            'weather_files'
        ),
        output_projection='EPSG:4236',
    )

def read_template_file(fname):
    """
    Read the template file into a dictionary structure.
    Parameters: fname      - str, full path to the template file
                delimiter  - str, string to separate the key and value
                skip_chars - list of str, skip certain charaters in values
    Returns:    template   - dict, file content
    Examples:   template = read_template('raider.yaml')

    Modified from MintPy's 'read_template'
    """
    with open(fname, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError('Something is wrong with the yaml file {}'.format(fname))

    # Drop any values not specified
    params = drop_nans(params)

    # Need to ensure that all the groups exist, even if they are not specified by the user
    group_keys = ['date_group', 'time_group', 'aoi_group', 'height_group', 'los_group', 'runtime_group']
    for key in group_keys:
        if not key in params.keys():
            params[key] = {}

    # Parse the user-provided arguments
    template = DEFAULT_DICT
    for key, value in params.items():
        if key == 'runtime_group':
            for k, v in value.items():
                if v is not None:
                    template[k] = v
        if key == 'weather_model':
            template[key]= enforce_wm(value)
        if key == 'time_group':
            template.update(enforce_time(AttributeDict(value)))
        if key == 'date_group':
            template['date_list'] = parse_dates(AttributeDict(value))
        if key == 'aoi_group':
            ## in case a DEM is passed and should be used
            dct_temp = {**AttributeDict(value),
                        **AttributeDict(params['height_group'])}
            template['aoi'] = get_query_region(AttributeDict(dct_temp))

        if key == 'los_group':
            template['los'] = get_los(AttributeDict(value))
        if key == 'look_dir':
            if value.lower() not in ['right', 'left']:
                raise ValueError(f"Unknown look direction {value}")
            template['look_dir'] = value.lower()

    # Have to guarantee that certain variables exist prior to looking at heights
    for key, value in params.items():
        if key == 'height_group':
            template.update(
                get_heights(
                    AttributeDict(value),
                    template['output_directory'],
                    template['station_file'],
                    template['bounding_box'],
                )
            )
    return AttributeDict(template)


def drop_nans(d):
    for key in list(d.keys()):
        if d[key] is None:
            del d[key]
        elif isinstance(d[key], dict):
            for k in list(d[key].keys()):
                if d[key][k] is None:
                    del d[key][k]
    return d


##########################################################################
def main(inps):
    # Read the template file
    params = read_template_file(inps.customTemplateFile)

    # Argument checking
    params = checkArgs(params)

    params['download_only'] = inps.download_only

    if not params.verbose:
        logger.setLevel(logging.INFO)


    for t, w, f in zip(
        params['date_list'],
        params['wetFilenames'],
        params['hydroFilenames']
    ):
        try:
            (_, _) = main_delay(t, w, f, params)
        except RuntimeError:
            logger.exception("Date %s failed", t)
            continue
