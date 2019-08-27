# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator as lndi
import pickle
import unittest

import RAiDER.llreader
import RAiDER.util
import RAiDER.delay
import RAiDER.delayFcns

from RAiDER.constants import Zenith

class WMTests(unittest.TestCase):

    picklefile = os.path.join('test', 'scenario_0', 'pickledWeatherModel.pik')
    with open(picklefile, 'rb') as f:
        wm = pickle.load(f)
    points = np.stack([wm._xs.flatten(), wm._ys.flatten(), wm._zs.flatten()], axis = -1)
    wrf = wm._wet_refractivity                                                                            
    hrf = wm._hydrostatic_refractivity
    zs = wm._zs[1,1,:]
    zref = 15000
    stepSize = 10

    # test error messaging
    def test_interpVector(self):
        f1 = lndi(self.points, self.wrf.flatten()) 
        f2 = lndi(self.points, self.hrf.flatten())  
        ray = np.stack([-100*np.ones(self.zref//100), 20*np.ones(self.zref//100), 
                         np.linspace(-100, self.zref, self.zref//100)]).T
        testwet = f1(ray)
        testhydro = f2(ray)
        dx = ray[1,2] - ray[0,2] 
        total = 1e-6*dx*np.sum(testwet + testhydro)
        total_true = 1e-6*(np.trapz(self.wrf[1,1,:], self.zs) + np.trapz(self.hrf[1,1,:], self.zs))

        self.assertTrue(np.abs(total-total_true) < 0.01)


def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

