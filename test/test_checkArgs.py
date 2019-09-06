# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
import pandas as pd
import unittest

import RAiDER.llreader
import RAiDER.util
import RAiDER.delay
import RAiDER.delayFcns

from RAiDER.constants import Zenith

class ArgTests(unittest.TestCase):

    station_file = 'test/scenario_9/station_file.txt'
    dt = '20180101T140000'
    model = 'ERA5'
    out = '.'
    parallel = False
    lats = np.array([
    41.8387,
    41.8531,
    39.6425,
    34.1574,
    38.1694,
    34.1042,
    34.0666,
    39.9036,
    39.1502,
    39.3525,
    38.6679,
    34.4133,
    36.9983,
    37.7630,
    37.3988,
    39.3459,
    38.7587,
    39.5202,
    41.7317,
    38.992,
    32.0277,
    37.5582])
    lats = lats.astype('float')
    stations = pd.DataFrame(data= np.array([
    ['34A1',41.8387,-119.6540,1907.6770],
    ['34A2',41.8531,-119.6074,1862.779],
    ['ANTV',39.6425,-120.2813,1681.0610],
    ['AOA1',34.1574,-118.8303,246.5610],
    ['ASTA',38.1694,-121.6845,-16.671],
    ['EWPP',34.1042,-117.5256,330.491],
    ['FCTF',34.0666,-118.4421,148.455],
    ['FIRE',39.9036,-119.0899,1426.6970],
    ['P186',39.1502,-123.5182,372.801],
    ['P187',39.3525,-123.6025,163.9910],
    ['P188',38.6679,-123.2296,208.88],
    ['UCSB',34.4133,-119.8438,-9.519],
    ['UCSC',36.9983,-122.0596,230.06],
    ['UCSF',37.7630,-122.4582,154.636],
    ['UFOS',37.3988,-117.109,1409.316],
    ['VIRC',39.3459,-119.6369,2013.983],
    ['WALK',38.7587,-118.764,1282.853],
    ['WWRF',39.5202,-119.7029,1325.5810],
    ['YBHB',41.7317,-122.7107,1065.683],
    ['YER1',38.992,-119.1623,1317.204],
    ['YUMX',32.0277,-115.1992,-28.4570],
    ['ZUMA',37.5582,-117.4902,1924.5520]]), 
      columns = ['ID', 'Lat', 'Lon', 'Hgt_m'])

    #lats = stations['Lat'].values
    #lons = stations['Lon'].values
    #hgts = stations['Hgt_m'].values

    # test error messaging
    def test_readLL(self):
        lat, lon, latproj, lonproj = RAiDER.llreader.readLL(self.station_file)
        import pdb; pdb.set_trace()
        self.assertTrue(np.allclose(lat, self.lats))
        

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

