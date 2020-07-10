# Unit and other tests
import datetime
import os
import pickle
import unittest

import numpy as np
import pandas as pd
from osgeo import gdal

from RAiDER.delay import computeDelay, interpolateDelay
from RAiDER.llreader import getHeights, readLL
from RAiDER.losreader import getLookVectors
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import modelName2Module, pickle_load


class TimeTests(unittest.TestCase):
    pass

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

