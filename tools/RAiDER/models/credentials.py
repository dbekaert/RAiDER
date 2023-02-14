'''
API credential information and help url for downloading weather model data 
    saved in a hidden file in home directory 

api filename      weather models          UID           KEY         URL
_________________________________________________________________________________
cdsapirc          ERA5, ERA5T             uid           key         https://cds.climate.copernicus.eu/api/v2
ecmwfapirc        ERAI, HRES              email         key         https://api.ecmwf.int/v1
netrc             GMAO, MERRA2            username      password    urs.earthdata.nasa.gov 
<NAN>             HRRR [public access]    <NAN>         <NAN> 
'''

import os
from pathlib import Path
from platform import system

# Filename for the hidden file per model
API_FILENAME = {'ERA5'  : 'cdsapirc',
                'ERA5T' : 'cdsapirc',
                'ERAI'  : 'ecmwfapirc',
                'HRES'  : 'ecmwfapirc',
                'GMAO'  : 'netrc',
                'HRRR'  :  None
                }

# API urls
API_URLS = {'cdsapirc'   : 'https://cds.climate.copernicus.eu/api/v2',
            'ecmwfapirc' : 'https://api.ecmwf.int/v1',
            'netrc' : 'urs.earthdata.nasa.gov'}

# api credentials dict
API_CREDENTIALS_DICT = {
        'cdsapirc' :   {'api' : """\
                                \nurl: {host}\
                                \nkey: {uid}:{key}
                                """,
                        'help_url' : 'https://cds.climate.copernicus.eu/api-how-to'
                        },
        'ecmwfapirc' : {'api' : """{{\
                                 \n"url"   : "{host}",\
                                 \n"key"   : "{key}",\
                                 \n"email" : "{uid}"\
                                 \n}}
                                """,
                        'help_url' : 'https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key'
                        },
        'netrc' :       {'api' : """\
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

# Check if API enviroments exists
def _check_envs(model):
    if model in ('ERA5', 'ERA5T'):
        uid = os.getenv('RAIDER_ECMWF_ERA5_UID')
        key = os.getenv('RAIDER_ECMWF_ERA5_API_KEY')
        host = API_URLS['cdsapirc']

    elif model in ('HRES'):
        uid = os.getenv('RAIDER_HRES_EMAIL') 
        key = os.getenv('RAIDER_HRES_API_KEY')
        host = os.getenv('RAIDER_HRES_URL')
        if host is None:
            host = API_URLS['ecmwfapirc']

    elif model in ('GMAO'):
        uid = os.getenv('EARTHDATA_USERNAME') # same as in DockerizedTopsApp
        key = os.getenv('EARTHDATA_PASSWORD')
        host = API_URLS['netrc']

    else: # for HRRR
        uid, key, host = None, None, None

    return uid, key, host

# Check and write MODEL API_RC_FILE for downloading weather model data
def check_api(model: str,
              UID: str = None,
              KEY: str = None,
              output_dir : str = '~/',
              update_flag: bool = False) -> None:

    # Weather model API filename
    # typically stored in home dir as hidden file
    api_filename = API_FILENAME[model]

    # Get API credential from os.env if UID/KEY are not inserted
    if UID is None and KEY is None:
        UID, KEY, URL = _check_envs(model)
    else:
        URL = API_URLS[api_filename]

    # Get hidden ext for Windows
    hidden_ext = '_' if system()=="Windows" else '.'

    # skip below if model is HRRR as it does not need API
    if api_filename:    
        # Check if the credential api file exists
        api_filename_path = Path(output_dir) / (hidden_ext + api_filename)
        api_filename_path = api_filename_path.expanduser()

        # if update flag is on, overwrite existing file 
        if update_flag is True:
            api_filename_path.unlink(missing_ok=True)
        
        # Check if API_RC file already exists
        if not api_filename_path.exists() and UID and KEY:
            # Create file with inputs, do it only once
            print(f'Writing {api_filename_path} locally!')
            api_filename_path.write_text(API_CREDENTIALS_DICT[api_filename]['api'].format(uid=UID,
                                                                                          key=KEY,
                                                                                          host=URL))
            api_filename_path.chmod(0o000600)

        else:
            # Raise ERROR message
            help_url = API_CREDENTIALS_DICT[api_filename]['help_url']

            # Raise ERROR in case only UID or KEY is inserted
            if UID is not None and KEY is None:
                raise ValueError(f'ERROR: API UID not inserted'
                                    f' or does not exist in ENVIRONMENTALS!')
            elif UID is None and KEY is not None:
                raise ValueError(f'ERROR: API KEY not inserted'
                                    f' or does not exist in ENVIRONMENTALS!')
            else:
                #Raise ERROR is both UID/KEY are none
                raise ValueError(
                        f'{api_filename_path}, API ENVIRONMENTALS'
                        f' and API UID and KEY, do not exist !!'
                        f'\nGet API info from ' + '\033[1m' f'{help_url}' + '\033[0m, and add it!')
