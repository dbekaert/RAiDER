import argparse
import copy
import multiprocessing
import numpy as np

from textwrap import dedent

from RAiDER.checkArgs import checkArgs
from RAiDER.cli.parser import add_bbox, add_out, add_verbose
from RAiDER.cli.validators import DateListAction, date_type, time_type
from RAiDER.constants import _ZREF
from RAiDER.delay import tropo_delay
from RAiDER.logger import logger, logging
from RAiDER.models.allowed import ALLOWED_MODELS
from RAiDER.processWM import weather_model_debug


# this and the next function are modified from MintPy
def get_the_latest_default_template_file(work_dir):
    """Get the latest version of default template file.
    If an obsolete file exists in the working directory, the existing option values are kept.
    """
    # get the latest and current versions of the default template files
    lfile = os.path.join(os.path.dirname(mintpy.__file__), 'defaults/smallbaselineApp.cfg')
    cfile = os.path.join(work_dir, 'smallbaselineApp.cfg')

    if not os.path.isfile(cfile):
        print(f'copy default template file {lfile} to work directory')
        shutil.copy2(lfile, work_dir)
    else:
        #cfile is obsolete if any key is missing
        ldict = readfile.read_template(lfile)
        cdict = readfile.read_template(cfile)
        if any([key not in cdict.keys() for key in ldict.keys()]):
            print('obsolete default template detected, update to the latest version.')
            shutil.copy2(lfile, work_dir)

            #keep the existing option value from obsolete template file
            ut.update_template_file(cfile, cdict)

    return cfile


def _tropo_delay(args):

    args_copy = copy.deepcopy(args)

    if 0 < len(args['times']) < 2:
        args_copy['times'] = args['times'][0]
        try:
            (_, _) = tropo_delay(args_copy)
        except RuntimeError:
            logger.exception("Date %s failed", args_copy['times'])
    else:
        for tim, wetFilename, hydroFilename in zip(args['times'], args['wetFilenames'], args['hydroFilenames']):
            try:
                args_copy['times'] = tim
                args_copy['wetFilenames'] = wetFilename
                args_copy['hydroFilenames'] = hydroFilename
                (_, _) = tropo_delay(args_copy)
            except RuntimeError:
                logger.exception("Date %s failed", tim)
                continue

