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

# system environmental variables
API_ENVIRONMENTS= {
    'cdsapirc'   : {'uid' : 'CDS_UID',
                    'key' : 'CDS_KEY'},

    'ecmwfapirc' : {'uid' : 'ECMWF_EMAIL',
                    'key' : 'ECMWF_KEY'},
                    
    'netrc'      : {'uid' : 'EARTHDATA_USERNAME',
                    'key' : 'EARTHDATA_PASSWORD'}
                    }

# filename for the hidden file per model
API_FILENAME = {'ERA5'  : 'cdsapirc',
                'ERA5T' : 'cdsapirc',
                'ERAI'  : 'ecmwfapirc',
                'HRES'  : 'ecmwfapirc',
                'GMAO'  : 'netrc',
                'MERRA2': 'netrc',
                'HRRR'  :  None
               }

# api credentials dict
MODEL_API_DICT = {
        'cdsapirc' :   {'api' : """\
                                url: https://cds.climate.copernicus.eu/api/v2\
                                \nkey: {uid}:{key}
                                """,
                        'url' : 'https://cds.climate.copernicus.eu/api-how-to'
                        },
        'ecmwfapirc' : {'api' : """{{\
                                 \n"url"   : "https://api.ecmwf.int/v1",\
                                 \n"key"   : "{key}",\
                                 \n"email" : "{uid}"\
                                 \n}}
                                """,
                        'url' : 'https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key'
                        },
        'netrc' :       {'api' : """\
                                \nmachine urs.earthdata.nasa.gov\
                                \n        login {uid}\
                                \n        password {key}\
                                """,
                       'url': 'https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget'
                       }
                }

# Check if enviroments exists
def _check_envs(api):
    try:
        uid = os.getenv(API_ENVIRONMENTS[api]['uid'])
        key = os.getenv(API_ENVIRONMENTS[api]['key']) 
        return uid, key
    except:
        return None, None

# Function to check and write MODEL API credential for downloading weather model data
def check_api(model: str,
              UID: str = None,
              KEY: str = None,
              prompt_flag: bool = True,
              update_flag: bool = False) -> None:
    import os

    # Weather model credential filename
    # typically stored in home dir as hidden file
    api_filename = API_FILENAME[model]

    # Get API credential from os.env if they exist
    UID, KEY = _check_envs(api_filename)

    # skip below if model is HRRR as it does not need API
    if api_filename:    
        # Check if the credential api file exists
        api_filename_path = os.path.expanduser('~/.')+ f'{api_filename}'

        # if update flag is on, delete existing file and update it
        if update_flag:
            if os.path.exists(api_filename_path):
                os.remove(api_filename_path) 
            
        if not os.path.exists(api_filename_path): 
            # Credential API file does not exist, create it
            # Need api information
            if UID is None or KEY is None:
                url = MODEL_API_DICT[api_filename]['url']

                #if prompt flag selected, ask user to input the API uid & key
                if prompt_flag:
                    print(f'GET MODEL API credentials, link: {url}')
                    UID = input('Please type your UID [uid, email, username]:')
                    KEY = input('Please type your KEY [key, password]')
                    if UID == '' or KEY == '':
                        raise ValueError('ERROR : UID and/or KEY are empty, define them !!')

                # Raise ERROR
                else:
                    msg = f'{api_filename_path} and weather model credential API UID and KEY,'
                    msg += ' do not exist !!'
                    msg += '\nGet API info from ' + '\033[1m' f'{url}' + '\033[0m, and added it!'
                    raise ValueError(msg)
                
            # Create file with inputs, Needed only once
            print(f'Writing {api_filename_path} locally!')
            with open(api_filename_path, 'w') as f:
                f.write(MODEL_API_DICT[api_filename]['api'].format(uid=UID,
                                                                    key=KEY))
            os.system(f'chmod 0600 {api_filename_path}')
