from RAiDER.models.weatherModel import WeatherModel


class customModelReader(WeatherModel):
    def __init__(self):
        WeatherModel.__init__(self)
        self._humidityType = 'q'  # can be "q" (specific humidity) or "rh" (relative humidity)
        self._model_level_type = 'pl'  # Default, pressure levels are "pl", and model levels are "ml"
        self._classname = 'abcd'  # name of the custom weather model
        self._dataset = 'abcd'  # same name as above

        # Tuple of min/max years where data is available.
        #  valid range of the dataset. Users need to specify the start date and end date (can be "present")
        self._valid_range = (datetime.datetime(2016, 7, 15), "Present")
        #  Availability lag time. Can be specified in hours "hours=3" or in days "days=3"
        self._lag_time = datetime.timedelta(hours=3)

        # model constants (these three constants are borrowed from ECMWF model and currently
        # set to be default for all other models, which may need to be double checked.)
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # horizontal grid spacing
        self._lat_res = 3. / 111  # grid spacing in latitude
        self._lon_res = 3. / 111  # grid spacing in longitude
        self._x_res = 3.  # x-direction grid spacing in the native weather model projection
        #  (if the projection is in lat/lon, it is the same as "self._lon_res")
        self._y_res = 3.  # y-direction grid spacing in the weather model native projection
        #  (if the projection is in lat/lon, it is the same as "self._lat_res")

        self._Name = 'ABCD'  # name of the custom weather model (better to be capitalized)

        # Projections in RAiDER are defined using pyproj (python wrapper around Proj)
        # If the projection is defined with EPSG code, one can use "self._proj = CRS.from_epsg(4326)"
        # to replace the following lines to get "self._proj".
        # Below we show the example of HRRR model with the parameters of its Lambert Conformal Conic projection
        lon0 = 262.5
        lat0 = 38.5
        lat1 = 38.5
        lat2 = 38.5
        x0 = 0
        y0 = 0
        earth_radius = 6371229
        p1 = CRS('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} +x_0={x0} +y_0={y0} +a={a} +b={a} +units=m +no_defs'.format(lat1=lat1, lat2=lat2, lat0=lat0, lon0=lon0, x0=x0, y0=y0, a=earth_radius))
        self._proj = p1

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch weather model data from the custom weather model "ABCD"
        Inputs (no need to change in the custom weather model reader): 
        lats - latitude 
        lons - longitude 
        time - datatime object (year,month,day,hour,minute,second)
        out - name of downloaded dataset file from the custom weather model server
        Nextra - buffer of latitude/longitude for determining the bounding box 
        '''

        # bounding box plus a buffer using the helper function from the WeatherModel base class
        # This part can be kept without modification.
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)

        # Auxilliary function:
        # download dataset of the custom weather model "ABCD" from a server and then save it to a file named out.
        # This function needs to be writen by the users. For download from the weather model server, the weather model
        # name, time and bounding box may be needed to retrieve the dataset; for cases where no file is actually
        # downloaded, e.g. the GMAO and MERRA-2 models using OpenDAP, this function can be omitted leaving the data
        # retrieval to the following "load_weather" function.
        self._files = self._download_abcd_file(out, 'abcd', time, self._bounds)

    def load_weather(self, filename):
        '''
        Load weather model variables from the downloaded file named filename
        Inputs: 
        filename - filename of the downloaded weather model file 
        '''

        # Auxilliary function:
        # read individual variables (in 3-D cube format with exactly the same dimension) from downloaded file
        # This function needs to be writen by the users. For downloaded file from the weather model server,
        # this function extracts the individual variables from the saved file named filename;
        # for cases where no file is actually downloaded, e.g. the GMAO and MERRA-2 models using OpenDAP,
        # this function retrieves the individual variables directly from the weblink of the weather model.
        lats, lons, xs, ys, t, q, p, hgt = self._makeDataCubes(filename)

        # extra steps that may be needed to calculate topographic height and pressure level if not provided
        # directly by the weather model through the above auxilliary function "self._makeDataCubes"

        # if surface pressure (in logarithm) is provided as "p" along with the surface geopotential "z" (needs to be
        # added to the auxilliary function "self._makeDataCubes"), one can use the following line to convert to
        # geopotential, pressure level and geopotential height; otherwise commented out
        z, p, hgt = self._calculategeoh(z, p)

        # if the geopotential is provided as "z" (needs to be added to the auxilliary function "self._makeDataCubes"),
        # one can use the following line to convert to geopotential height; otherwise, commented out
        hgt = z / self._g0

        # if geopotential height is provided/calculated as "hgt", one can use the following line to convert to
        # topographic height, which is then automatically assigned to "self._zs"; otherwise commented out
        self._get_heights(lats, hgt)

        # if topographic height is provided as "hgt", use the following line directly; otherwise commented out
        self._zs = hgt

        ###########

        ######### output of the weather model reader for delay calculations (all in 3-D data cubes) ########

        # _t: temperture
        # _q: either relative or specific humidity
        # _p: must be pressure level
        # _xs: x-direction grid dimension of the native weather model coordinates (if in lat/lon, _xs = _lons)
        # _ys: y-direction grid dimension of the native weather model coordinates (if in lat/lon, _ys = _lats)
        # _zs: must be topographic height
        # _lats: latitude
        # _lons: longitude
        self._t = t
        self._q = q
        self._p = p
        self._xs = xs
        self._ys = ys
        self._lats = lats
        self._lons = lons

    def _download_abcd_file(self, out, model_name, date_time, bounding_box):
        '''
        Auxilliary function:
        Download weather model data from a server
        Inputs: 
        out - filename for saving the retrieved data file 
        model_name - name of the custom weather model 
        date_time - datatime object (year,month,day,hour,minute,second)
        bounding_box - lat/lon bounding box for the region of interest
        Output: 
        out - returned filename from input
        '''
        pass

    def _makeDataCubes(self, filename):
        '''
        Auxilliary function:
        Read 3-D data cubes from downloaded file or directly from weather model weblink (in which case, there is no 
        need to download and save any file; rather, the weblink needs to be hardcoded in the custom reader, e.g. GMAO)
        Input: 
        filename - filename of the downloaded weather model file from the server
        Outputs: 
        lats - latitude (3-D data cube)
        lons - longitude (3-D data cube)
        xs - x-direction grid dimension of the native weather model coordinates (3-D data cube; if in lat/lon, _xs = _lons)
        ys - y-direction grid dimension of the native weather model coordinates (3-D data cube; if in lat/lon, _ys = _lats)
        t - temperature (3-D data cube)
        q - humidity (3-D data cube; could be relative humidity or specific humidity)
        p - pressure level (3-D data cube; could be pressure level (preferred) or surface pressure)
        hgt - height (3-D data cube; could be geopotential height or topographic height (preferred))
        '''
        pass
