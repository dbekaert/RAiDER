# Setup for accessing weather models

## For accessing ECMWF-type weather models (e.g. ECMWF, ERA5, ERA5-T)
1. Create an account on the Copernicus servers here:  https://cds.climate.copernicus.eu/user
2. Confirm your email, etc. 
3. Install the public API key and client (https://cds.climate.copernicus.eu/api-how-to): 
   a. Copy the URL and API key from the webpage into a file in your home directory name ~/.cdsapirc 
   b. Install the CDS API using pip: pip install cdsapi. 
4. You must accept the license for each product you wish to download: https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
5. See the test_cdsapi.py script for details of the API. You can test that you can connect to the servers by running the test suite (described below). 
