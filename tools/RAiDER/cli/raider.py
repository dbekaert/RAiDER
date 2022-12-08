import argparse
import os
import shutil
import sys
import yaml
import re, glob

import RAiDER
from RAiDER.checkArgs import checkArgs
from RAiDER.cli.validators import (
    enforce_time, parse_dates, get_query_region, get_heights, get_los, enforce_wm
)
from RAiDER.constants import _ZREF, _CUBE_SPACING_IN_M
from RAiDER.delay import tropo_delay
from RAiDER.logger import logger, logging
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import writeDelays


HELP_MESSAGE = """
Command line options for RAiDER processing. Default options can be found by running
raider.py --generate_config

Download a weather model and calculate tropospheric delays
"""

SHORT_MESSAGE = """
Program to calculate troposphere total delays using a weather model
"""

EXAMPLES = """
Usage examples:
raider.py -g
raider.py customTemplatefile.cfg
"""

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


def create_parser():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = HELP_MESSAGE,
        epilog = EXAMPLES,
    )

    p.add_argument(
        'customTemplateFile', nargs='?',
        help='custom template with option settings.\n' +
        "ignored if the default smallbaselineApp.cfg is input."
    )

    p.add_argument(
        '-g', '--generate_template',
        dest='generate_template',
        action='store_true',
        help='generate default template (if it does not exist) and exit.'
    )

    
    p.add_argument(
        '--download-only',
        action='store_true',
        help='only download a weather model.'
    )

    return p


def parseCMD(iargs=None):
    """
    Parse command-line arguments and pass to delay.py
    """

    p = create_parser()
    args = p.parse_args(args=iargs)

    args.argv = iargs if iargs else sys.argv[1:]

    # default input file
    template_file = os.path.join(
        os.path.dirname(
            RAiDER.__file__
        ),
        'cli', 'raider.yaml'
    )
    if '-g' in args.argv:
        dst = os.path.join(os.getcwd(), 'raider.yaml')
        shutil.copyfile(
                template_file,
                dst,
            )

        logger.info('Wrote %s', dst)
        sys.exit(0)

    # check: existence of input template files
    if (not args.customTemplateFile
            and not os.path.isfile(os.path.basename(template_file))
            and not args.generate_template):
        p.print_usage()
        print(EXAMPLES)

        msg = "No template file found! It requires that either:"
        msg += "\n  a custom template file, OR the default template "
        msg += "\n  file 'raider.yaml' exists in current directory."
        raise SystemExit(f'ERROR: {msg}')

    if  args.customTemplateFile:
        # check the existence
        if not os.path.isfile(args.customTemplateFile):
            raise FileNotFoundError(args.customTemplateFile)

        args.customTemplateFile = os.path.abspath(args.customTemplateFile)

    return args


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
def main(iargs=None):
    # parse
    inps = parseCMD(iargs)

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

        los = params['los']
        aoi = params['aoi']
        model = params['weather_model']

        if los.ray_trace():
            ll_bounds = aoi.add_buffer(buffer=1) # add a buffer for raytracing
        else:
            ll_bounds = aoi.bounds()

        ###########################################################
        # weather model calculation
        logger.debug('Starting to run the weather model calculation')
        logger.debug('Time: {}'.format(t.strftime('%Y%m%d')))
        logger.debug('Beginning weather model pre-processing')
        try:
            weather_model_file = prepareWeatherModel(
                model, t,
                ll_bounds=ll_bounds, # SNWE
                wmLoc=params['weather_model_directory'],
                zref=params['zref'],
                makePlots=params['verbose'],
            )
        except RuntimeError:
            logger.exception("Date %s failed", t)
            continue
    
        # Now process the delays
        try:
            wet_delay, hydro_delay = tropo_delay(
                t, weather_model_file, aoi, los, 
                params['height_levels'], 
                params['output_projection'],
                params['look_dir'],
                params['cube_spacing_in_m']
            )
        except RuntimeError:
            logger.exception("Date %s failed", t)
            continue
        
        ###########################################################
        # Write the delays to file
        # Different options depending on the inputs

        if not los.ray_trace() and not los.is_Zenith():
            out_filename = w.replace("_ztd", "_std")
        else:
            out_filename = w.replace("_ztd", "_ray")
        
        if hydro_delay is None:
            # means that a dataset was returned
            ds = wet_delay
            ext = os.path.splitext(out_filename)
            if ext not in ['.nc', '.h5']:
                out_filename = f'{os.path.splitext(out_filename)[0]}.nc'
            
            out_filename = out_filename.replace("wet", "tropo")

            if out_filename.endswith(".nc"):
                ds.to_netcdf(out_filename, mode="w")
            elif out_filename.endswith(".h5"):
                ds.to_netcdf(out_filename, engine="h5netcdf", invalid_netcdf=True)

        else:
            if aoi.type() == 'station_file':
                w = f'{os.path.splitext(w)[0]}.csv'

            if aoi.type() in ['station_file', 'radar_rasters']:
                writeDelays(aoi, wet_delay, hydro_delay, w, f, outformat=params['raster_format'])
    

