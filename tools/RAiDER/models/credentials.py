'''
API credential information and help url for downloading weather model data 
    saved in a hidden file in home directory 

api filename      weather models          uid           key           host
_________________________________________________________________________________
cdsapirc          ERA5, ERA5T             uid           key         https://cds.climate.copernicus.eu/api/v2
ecmwfapirc        ERAI, HRES              email         key         https://api.ecmwf.int/v1
netrc             GMAO, MERRA2            username      password    urs.earthdata.nasa.gov 
<NAN>             HRRR [public access]    <NAN>         <NAN> 
'''

import os

# Filename for the hidden file per model
API_FILENAME = {'ERA5'  : 'cdsapirc',
                'ERA5T' : 'cdsapirc',
                'ERAI'  : 'ecmwfapirc',
                'HRES'  : 'ecmwfapirc',
                'GMAO'  : 'netrc',
                'MERRA2': 'netrc',
                'HRRR'  :  None
                }

# system environmental variables for API credentials
'''
cdsapi : already has as default set to look for CDSAPI_URL, CDSAPI_KEY
         https://github.com/ecmwf/cdsapi/blob/master/cdsapi/api.py LINE:253, 254
         however, if hidden file CDSAPI_RC exists, it overwrites CDSAPI_URL, CDSAPI_KEY

         CDSAPI_KEY = UID:KEY

ecmwfapir : same as above, looks for the envs however they are not selected as default input (default: anonymous)
            https://github.com/ecmwf/ecmwf-api-client/blob/master/ecmwfapi/api.py
'''

API_ENVIRONMENTS= {
    'cdsapirc'   : {'uid' : 'CDSAPI_KEY', #Not Needed
                    'key' : 'CDSAPI_KEY',
                    'host': 'CDSAPI_URL'},

    'ecmwfapirc' : {'uid' : 'ECMWF_API_EMAIL',
                    'key' : 'ECMWF_API_KEY',
                    'host': 'ECMWF_API_URL'},
                    
    'netrc'      : {'uid' : 'EARTHDATA_USERNAME',
                    'key' : 'EARTHDATA_PASSWORD',
                    'host': 'EARTHDATA_URL'} #Added this one in case of url change
                    }

# API urls
API_URLS = {'cdsapirc'   : 'https://cds.climate.copernicus.eu/api/v2',
            'ecmwfapirc' : 'https://api.ecmwf.int/v1',
            'netrc' : 'urs.earthdata.nasa.gov'}

# api credentials dict
API_CREDENTIALS_DICT = {
        'cdsapirc' :   {'api' : """\
                                url: {host}\
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

# Check if API enviroments exists
def _check_envs(api):
    # First check if API host env exist, if not use the default ones
    try:
        host = os.getenv(API_ENVIRONMENTS[api]['host'])
    except (RuntimeError):
        print('Unable to get API url env!')
    else:
        host = API_URLS[api] 

    # Second check for API env credentials
    try:
        if api == 'cdsapirc':
            uid, key  = os.getenv(API_ENVIRONMENTS[api]['key']).split(':')
        else:
            uid  = os.getenv(API_ENVIRONMENTS[api]['uid'])
            key = os.getenv(API_ENVIRONMENTS[api]['key']) 
        return uid, key, host
    except:
        return None, None, host

# Check and write MODEL API_RC_FILE for downloading weather model data
def check_api(model: str,
              UID: str = None,
              KEY: str = None,
              update_flag: bool = False) -> None:

    # Weather model credential filename
    # typically stored in home dir as hidden file
    api_filename = API_FILENAME[model]

    # Get API credential from os.env if they exist
    UID, KEY, URL = _check_envs(api_filename)

    # skip below if model is HRRR as it does not need API
    if api_filename:    
        # Check if the credential api file exists
        api_filename_path = os.path.expanduser('~/.') + f'{api_filename}'

        # if update flag is on, delete existing file and update it
        if update_flag:
            if os.path.exists(api_filename_path):
                os.remove(api_filename_path) 
            
        if not os.path.exists(api_filename_path): 
            # Credential API file does not exist, create it
            # Need api information
            if UID is None or KEY is None:
                url = API_CREDENTIALS_DICT[api_filename]['help_url']
                
                #Raise ERROR
                msg = f'{api_filename_path}, API ENVIRONMENTALS and weather model'
                msg += 'credentials API UID and KEY, do not exist !!'
                msg += '\nGet API info from ' + '\033[1m' f'{url}' + '\033[0m, and add it!'
                raise ValueError(msg)
                
            # Create file with inputs, do it only once
            print(f'Writing {api_filename_path} locally!')
            with open(api_filename_path, 'w') as f:
                f.write(API_CREDENTIALS_DICT[api_filename]['api'].format(uid=UID,
                                                                         key=KEY,
                                                                         host=URL))
            os.system(f'chmod 0600 {api_filename_path}')
