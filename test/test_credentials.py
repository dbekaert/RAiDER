import os
from platform import system
from RAiDER.models import credentials

# Test checking/creating ECMWF_RC CAPI file
def test_ecmwfApi_createFile():
    import ecmwfapi
    
    #Check extension for hidden files
    hidden_ext = '_' if system()=="Windows" else '.'

    # Test creation of ~/.ecmwfapirc file
    ecmwf_file = os.path.expanduser('./') + hidden_ext + credentials.API_FILENAME['ERAI']
    credentials.check_api('ERAI', 'dummy', 'dummy', './', update_flag=True)
    assert os.path.exists(ecmwf_file) == True,f'{ecmwf_file} does not exist' 

    # Get existing ECMWF_API_RC env if exist
    default_ecmwf_file = os.getenv("ECMWF_API_RC_FILE")
    if default_ecmwf_file is None:
        default_ecmwf_file = ecmwfapi.api.DEFAULT_RCFILE_PATH

    #Set it to current dir to avoid overwriting ~/.ecmwfapirc file 
    os.environ["ECMWF_API_RC_FILE"] = ecmwf_file
    key, url, uid = ecmwfapi.api.get_apikey_values()
    
    # Return to default_ecmwf_file and remove local API file
    os.environ["ECMWF_API_RC_FILE"] = default_ecmwf_file 
    os.remove(ecmwf_file)

    #Check if API is written correctly
    assert uid ==  'dummy', f'{ecmwf_file}: UID was not written correctly'
    assert key == 'dummy', f'{ecmwf_file}: KEY was not written correctly'


# Test checking/creating Copernicus Climate Data Store
#   CDS_RC CAPI file 
def test_cdsApi_createFile():
    import cdsapi

    #Check extension for hidden files
    hidden_ext = '_' if system()=="Windows" else '.'
    
    # Test creation of .cdsapirc file in current dir
    cds_file = os.path.expanduser('./') + hidden_ext + credentials.API_FILENAME['ERA5']
    credentials.check_api('ERA5', 'dummy', 'dummy', './', update_flag=True)
    assert os.path.exists(cds_file) == True,f'{cds_file} does not exist' 

    # Check the content
    cds_credentials = cdsapi.api.read_config(cds_file)
    uid, key = cds_credentials['key'].split(':')

    # Remove local API file
    os.remove(cds_file)
    
    assert uid ==  'dummy', f'{cds_file}: UID was not written correctly'
    assert key == 'dummy', f'{cds_file}: KEY was not written correctly'

# Test checking/creating EARTHDATA_RC API file 
def test_netrcApi_createFile():
    import netrc
    
    #Check extension for hidden files
    hidden_ext = '_' if system()=="Windows" else '.'

    # Test creation of ~/.cdsapirc file
    netrc_file = os.path.expanduser('./') + hidden_ext + credentials.API_FILENAME['GMAO']
    credentials.check_api('GMAO', 'dummy', 'dummy', './', update_flag=True)
    assert os.path.exists(netrc_file) == True,f'{netrc_file} does not exist' 

    # Check the content
    host = 'urs.earthdata.nasa.gov'
    netrc_credentials = netrc.netrc(netrc_file)
    uid, _, key = netrc_credentials.authenticators(host)

    # Remove local API file
    os.remove(netrc_file)
    
    assert uid ==  'dummy', f'{netrc_file}: UID was not written correctly'
    assert key == 'dummy', f'{netrc_file}: KEY was not written correctly'