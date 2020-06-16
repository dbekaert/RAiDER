# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
import pandas as pd
import unittest




class hdf5_parallel_Tests(unittest.TestCase):

    stepSize = 15.0
    pnts_file = '../test/scenario_1/geom/query_points.h5'
    wm_file = '../test/scenario_1/weather_files/ERA5_2020-01-03 23:00:00_15.75N_18.25N_-103.25E_-99.75E.h5'
    interpType = 'rgi'
    verbose = True
    delayType = ['Zenith']
    
    from RAiDER.delayFcns import get_delays
    
    delays_wet_1, delays_hydro_1 = get_delays(stepSize, pnts_file, wm_file, interpType,
                                              verbose, delayType, cpu_num = 1)
    with open('get_delays_time_elapse.txt', 'r') as f:
        time_elapse_1 = float(f.readline())

    delays_wet_4, delays_hydro_4 = get_delays(stepSize, pnts_file, wm_file, interpType,
                                              verbose, delayType, cpu_num = 4)
    with open('get_delays_time_elapse.txt', 'r') as f:
        time_elapse_4 = float(f.readline())


    # test error messaging
    def test_get_delays_wet_accuracy(self):
        self.assertTrue(np.allclose(self.delays_wet_1, self.delays_wet_4))

    
    def test_get_delays_hydro_accuracy(self):
        self.assertTrue(np.allclose(self.delays_hydro_1, self.delays_hydro_4))

    
    def test_get_delays_runtime(self):
        print("Speedup by using 4 cpu threads vs single thread: ".format(self.time_elapse_1/self.time_elapse_4))


def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

