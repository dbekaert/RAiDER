# Setup for accessing weather model data

## For accessing ECMWF-type weather model data (e.g. ECMWF, ERA5, ERA5-T)

1. Create an account on the Copernicus servers [here](https://cds.climate.copernicus.eu/user)

2. Confirm your email, etc. 

3. Install the public API key and client as instructed [here](https://cds.climate.copernicus.eu/api-how-to): 

   a. Copy the URL and API key from the webpage into a file in your home directory name ~/.cdsapirc 
      
         url: https://cds.climate.copernicus.eu/api/v2
         key: <KEY>
      
      _Note the `<KEY>` represents the API key obtained upon the registration of CDS API, and should be replaced with the user's own information._
      
   b. Install the CDS API using pip: 
   
         pip install cdsapi
   
   ___Note: this step has been included in the conda install of RAiDER, thus can be omitted if one uses the recommended conda install of RAiDER___
   
4. You must accept the [license](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products) for each product you wish to download.

## For accessing MERRA2-type weather model data with the use of OpenDAP  (e.g. MERRA-2, GMAO)

1. Create an account on the NASA's Earthdata website [here](https://urs.earthdata.nasa.gov)

2. Confirm your email, etc. 

3. Copy the login username and password to a file in your home directory name ~/.netrc 
         
         machine urs.earthdata.nasa.gov
                 login <USERNAME>
                 password <PASSWORD>
                 
   _Note the `<USERNAME>` and `<PASSWORD>` represent the actual username and password, and should be replaced with the user's own information correspondingly._
   
4. Add the application `NASA GESDISC DATA ARCHIVE` by clicking on the `Applications->Authorized Apps` on the menu after logging into your Earthdata profile, and then scrolling down to the application `NASA GESDISC DATA ARCHIVE` to approve it. _This seems not required for GMAO for now, but recommended to do so for all OpenDAP-based weather models._

5. Install the OpenDAP using pip: 

         pip install pydap
      
   ___Note: this step has been included in the conda install of RAiDER, thus can be omitted if one uses the recommended conda install of RAiDER___
   
   ___Note: PyDAP v3.2.1 is required for now because the latest v3.2.2 (as of now) has a known [bug](https://colab.research.google.com/drive/1f_ss1Oa3VzgAOd_p8sgekdnLVE5NW6s5) in accessing and slicing the GMAO data. This bug is expected to be fixed in newer versions of PyDAP.___
