import copy
import logging
import sys

from RAiDER.cli.raiderDelay import parseCMD, read_template_file
from RAiDER.checkArgs import checkArgs
from RAiDER.delay import tropo_delay
from RAiDER.logger import logger


##########################################################################
def main(iargs=None):
    # parse
    inps = parseCMD(iargs)

    # Read the template file
    params = read_template_file(inps.customTemplateFile)

    # Argument checking
    params = checkArgs(params)

    if params.verbose:
        logger.setLevel(logging.DEBUG)

    # run
    step_list       = inps.runSteps
    params.runSteps = step_list


    if 'download_gnss' in step_list:
        from RAiDER.downloadGNSSDelays import main
        params['gps_repo'] = 'UNR' # only UNR supported; used to be exposed
        params['out']      = f'{params["gps_repo"]}_products' # output directory
        params['download'] = False
        params['cpus']     = 4
        params['bounding_box'] = params['aoi'].bounds()

        main(params)


    #TODO: separate out the weather model calculation as a separate step
    if 'calculate_delays' in step_list:
        for t, w, f in zip(
            params['date_list'],
            params['wetFilenames'],
            params['hydroFilenames']
        ):
            try:
                (_, _) = tropo_delay(t, w, f, params)
            except RuntimeError:
                logger.exception("Date %s failed", t)
                continue
