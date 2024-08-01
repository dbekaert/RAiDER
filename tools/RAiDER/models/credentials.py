'''
API credential information and help url for downloading weather model data
    saved in a hidden file in home directory

api filename      weather models          UID           KEY         URL
_________________________________________________________________________________
cdsapirc          ERA5, ERA5T             uid           key         https://cds.climate.copernicus.eu/api/v2
ecmwfapirc        HRES                    email         key         https://api.ecmwf.int/v1
netrc             GMAO, MERRA2            username      password    urs.earthdata.nasa.gov
<N/A>             HRRR [public access]    <N/A>         <N/A>       <N/A>
'''

import os
from pathlib import Path
from platform import system
from typing import Dict, Optional, Tuple

from RAiDER.logger import logger


# Filename for the rc file for each model
RC_FILENAMES: Dict[str, Optional[str]] = {
    'ERA5': 'cdsapirc',
    'ERA5T': 'cdsapirc',
    'HRES': 'ecmwfapirc',
    'GMAO': 'netrc',
    'MERRA2': 'netrc',
    'HRRR':  None
}

APIS = {
    'cdsapirc': {
        'template': (
            'url: {host}\n'
            'key: {uid}:{key}\n'
        ),
        'help_url': 'https://cds.climate.copernicus.eu/api-how-to',
        'default_host': 'https://cds.climate.copernicus.eu/api/v2'
    },
    'ecmwfapirc': {
        'template': (
            '{{\n'
            '    "url"   : "{host}",\n'
            '    "key"   : "{key}",\n'
            '    "email" : "{uid}"\n'
            '}}\n'
        ),
        'help_url': 'https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key',
        'default_host': 'https://api.ecmwf.int/v1'
    },
    'netrc': {
        'template': (
            'machine {host}\n'
            '	login {uid}\n'
            '	password {key}\n'
        ),
        'help_url': 'https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget',
        'default_host': 'urs.earthdata.nasa.gov'
    }
}


# Get the environment variables for a given weather model API
def _get_envs(model: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if model in ('ERA5', 'ERA5T'):
        uid = os.getenv('RAIDER_ECMWF_ERA5_UID')
        key = os.getenv('RAIDER_ECMWF_ERA5_API_KEY')
        host = APIS['cdsapirc']['default_host']
    elif model == 'HRES':
        uid = os.getenv('RAIDER_HRES_EMAIL')
        key = os.getenv('RAIDER_HRES_API_KEY')
        host = os.getenv('RAIDER_HRES_URL',
                            APIS['ecmwfapirc']['default_host'])
    elif model in ('GMAO', 'MERRA2'):
        # same as in DockerizedTopsApp
        uid = os.getenv('EARTHDATA_USERNAME')
        key = os.getenv('EARTHDATA_PASSWORD')
        host = APIS['netrc']['default_host']
    else:  # for HRRR
        uid, key, host = None, None, None
    return uid, key, host


def check_api(model: str,
              uid: Optional[str] = None,
              key: Optional[str] = None,
              output_dir: str = '~/',
              update_rc_file: bool = False) -> None:
    # Weather model API RC filename
    # Typically stored in home dir as a hidden file
    rc_filename = RC_FILENAMES[model]

    # If the given API does not require an rc file, then there is nothing to do
    # (e.g., HRRR)
    if rc_filename is None:
        return

    # Get the target rc file's path
    hidden_ext = '_' if system() == "Windows" else '.'
    rc_path = Path(output_dir) / (hidden_ext + rc_filename)
    rc_path = rc_path.expanduser()

    # If the RC file doesn't exist, then create it.
    # But if it does exist, only update it if the user requests.
    if rc_path.exists() and not update_rc_file:
        return

    # Get credentials from env vars if uid and key are not passed in
    if uid is None and key is None:
        uid, key, url = _get_envs(model)
    else:
        url = APIS[rc_filename]['default_host']

    # Check for invalid inputs
    if uid is None or key is None:
        help_url = APIS[rc_filename]['help_url']
        if uid is None and key is not None:
            raise ValueError(
                f'ERROR: {model} API UID not provided in RAiDER arguments and '
                'not present in environment variables.\n'
                f'See info for this model\'s API at \033[1m{help_url}\033[0m'
            )
        elif uid is not None and key is None:
            raise ValueError(
                f'ERROR: {model} API key not provided in RAiDER arguments and '
                'not present in environment variables.\n'
                f'See info for this model\'s API at \033[1m{help_url}\033[0m'
            )
        else:
            raise ValueError(
                f'ERROR: {model} API credentials not provided in RAiDER '
                'arguments and not present in environment variables.\n'
                f'See info for this model\'s API at \033[1m{help_url}\033[0m'
            )

    # Create file with the API credentials
    if update_rc_file:
        logger.info(f'Updating {model} API credentials in {rc_path}')
    else:
        logger.warning(f'{model} API credentials not found in {rc_path}; creating')
    rc_type = RC_FILENAMES[model]
    if rc_type in ('cdsapirc', 'ecmwfapirc'):
        # These RC files only ever contain one set of credentials, so
        # they can just be overwritten when updating.
        template = APIS[rc_filename]['template']
        entry = template.format(host=url, uid=uid, key=key)
        rc_path.write_text(entry)
    elif rc_type == 'netrc':
        # This type of RC file may contain more than one set of credentials,
        # so extra care needs to be taken to make sure we only touch the
        # one that belongs to this URL.
        import netrc
        rc_path.touch()
        netrc_credentials = netrc.netrc(rc_path.name)
        netrc_credentials.hosts[url] = (uid, '', key)
        rc_path.write_text(str(netrc_credentials))
    rc_path.chmod(0o000600)


def setup_from_env():
    for model in RC_FILENAMES.keys():
        check_api(model)
