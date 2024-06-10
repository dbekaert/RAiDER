'''
API credential information and help url for downloading weather model data
    saved in a hidden file in home directory

api filename      weather models          UID           KEY         URL
_________________________________________________________________________________
cdsapirc          ERA5, ERA5T             uid           key         https://cds.climate.copernicus.eu/api/v2
ecmwfapirc        HRES                    email         key         https://api.ecmwf.int/v1
netrc             GMAO, MERRA2            username      password    urs.earthdata.nasa.gov
<NAN>             HRRR [public access]    <NAN>         <NAN>
'''

import os
from pathlib import Path
from platform import system
from typing import Dict, Optional, Tuple

# Filename for the rc file for each model
RC_FILENAMES: Dict[str, Optional[str]] = {
    'ERA5': 'cdsapirc',
    'ERA5T': 'cdsapirc',
    'HRES': 'ecmwfapirc',
    'GMAO': 'netrc',
    'HRRR':  None
}

API_URLS = {
    'cdsapirc': 'https://cds.climate.copernicus.eu/api/v2',
    'ecmwfapirc': 'https://api.ecmwf.int/v1',
    'netrc': 'urs.earthdata.nasa.gov'
}

API_CREDENTIALS_DICT = {
    'cdsapirc': {
        'api': """\
                                \nurl: {host}\
                                \nkey: {uid}:{key}
        """,
        'help_url': 'https://cds.climate.copernicus.eu/api-how-to'
    },
    'ecmwfapirc': {
        'api': """{{\
                                 \n"url"   : "{host}",\
                                 \n"key"   : "{key}",\
                                 \n"email" : "{uid}"\
                                 \n}}
        """,
        'help_url': 'https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key'
    },
    'netrc': {
        'api': """\
                                \nmachine {host}\
                                \n        login {uid}\
                                \n        password {key}\
        """,
        'help_url': 'https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget'
    }
}

# system environmental variables for API credentials
'''
ENV variables in cdsapi and ecmwapir

cdsapi ['cdsapirc'] : CDSAPI_KEY [UID:KEY], 'CDSAPI_URL'
ecmwfapir [ecmwfapirc] : 'ECMWF_API_KEY', 'ECMWF_API_EMAIL','ECMWF_API_URL'  

'''

# Check if the user has the environment variables for a given weather model API
def _check_envs(model: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    match model:
        case 'ERA5' | 'ERA5T':
            uid = os.getenv('RAIDER_ECMWF_ERA5_UID')
            key = os.getenv('RAIDER_ECMWF_ERA5_API_KEY')
            host = API_URLS['cdsapirc']
        case 'HRES':
            uid = os.getenv('RAIDER_HRES_EMAIL')
            key = os.getenv('RAIDER_HRES_API_KEY')
            host = os.getenv('RAIDER_HRES_URL')
            if host is None:
                host = API_URLS['ecmwfapirc']
        case 'GMAO':
            # same as in DockerizedTopsApp
            uid = os.getenv('EARTHDATA_USERNAME')
            key = os.getenv('EARTHDATA_PASSWORD')
            host = API_URLS['netrc']
        case _:  # for HRRR
            uid, key, host = None, None, None

    return uid, key, host


# Check if the user has the rc file necessary for a given weather model API
def check_api(model: str,
              uid: Optional[str] = None,
              key: Optional[str] = None,
              output_dir: str = '~/',
              update_rc_file: bool = False) -> None:

    # Weather model API RC filename
    # Typically stored in home dir as a hidden file
    rc_filename = RC_FILENAMES[model]

    # If the given API does not require an rc file, return (nothing to do)
    if rc_filename is None:
        return

    # Get credentials from env vars if uid/key is not passed in
    if uid is None and key is None:
        uid, key, url = _check_envs(model)
    else:
        url = API_URLS[rc_filename]

    # Get hidden ext for Windows
    hidden_ext = '_' if system() == "Windows" else '.'

    # Get the target rc file's path
    rc_path = Path(output_dir) / (hidden_ext + rc_filename)
    rc_path = rc_path.expanduser()

    # if update flag is on, overwrite existing file
    if update_rc_file:
        rc_path.unlink(missing_ok=True)

    # Check if API_RC file already exists
    if rc_path.exists():
        return
    # if it does not exist, put uid/key inserted, create it
    elif not rc_path.exists() and uid and key:
        # Create file with inputs, do it only once
        print(f'Writing {rc_path} locally!')
        rc_path.write_text(
            API_CREDENTIALS_DICT[rc_filename]['api'].format(
                uid=uid,
                key=key,
                host=url
            )
        )
        rc_path.chmod(0o000600)
    # Raise ERROR message
    else:
        help_url = API_CREDENTIALS_DICT[rc_filename]['help_url']

        # Raise ERROR in case only username or password is present
        if uid is None and key is not None:
            raise ValueError('ERROR: API uid not inserted'
                             ' or does not exist in ENVIRONMENTALS!')
        elif uid is not None and key is None:
            raise ValueError('ERROR: API key not inserted'
                             ' or does not exist in ENVIRONMENTALS!')
        else:
            # Raise ERROR if both UID/KEY are none
            raise ValueError(
                f'{rc_path}, API ENVIRONMENTALS'
                ' and API UID and KEY, do not exist !!'
                f'\nGet API info from ' + '\033[1m' f'{help_url}' + '\033[0m, and add it!')


def setup_from_env():
    for model in RC_FILENAMES.keys():
        check_api(model)
