# Unit and other tests
import datetime
import gdal
import numpy as np
import os
import pandas as pd
import pickle
import unittest

from RAiDER.delay import computeDelay, interpolateDelay
from RAiDER.llreader import readLL, getHeights
from RAiDER.losreader import getLookVectors
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import pickle_load, modelName2Module

class TimeTests(unittest.TestCase):
    pass

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

