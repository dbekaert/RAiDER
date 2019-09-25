# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
import pandas as pd
import unittest

import RAiDER.llreader
import RAiDER.losreader
import RAiDER.delay
import RAiDER.delayFcns

from RAiDER.constants import Zenith

class FcnTests(unittest.TestCase):

    zs = np.linspace(0, 15000, 1000)
    step = zs[1] - zs[0]
    pw = 100*(np.arctan(((15000-zs)-10000)/800)+np.pi/2) 
    trueInt = np.round(np.trapz(pw,x=zs)/1e6, 2)

    lats = np.array([15.0, 15.5, 16.0, 16.5, 17.5, -40, 60, 90])
    lons = np.array([-100.0, -100.4, -91.2, 45.0, 0., -100,-100, -100])
    hgts = np.array([0., 1000., 10000., 0., 0., 0., 0., 0.])
    zref = 15000
    
    zlvsTrue = np.array([[-2.51596889e+03, -1.42687686e+04,  3.88228568e+03],
       [-2.43535244e+03, -1.32691919e+04,  3.74133727e+03],
       [-1.00655730e+02, -4.80525438e+03,  1.37818678e+03],
       [ 1.01698190e+04,  1.01698190e+04,  4.26023017e+03],
       [ 1.43057543e+04,  0.00000000e+00,  4.51058699e+03],
       [-1.99533332e+03, -1.13160976e+04, -9.64181415e+03],
       [-1.30236133e+03, -7.38605815e+03,  1.29903811e+04],
       [-1.59493264e-13, -9.04531247e-13,  1.50000000e+04]])

    vec1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    vec2 = np.array([0, 0, 1])
    rayTrue = np.stack([np.zeros(1001), np.zeros(1001), np.arange(0, 100.1, 0.1)], axis= 1)
    trueLengths = zref - hgts

    testArr1 = np.array([[10, 9, 8], [9, 9, 9], [1, 1, 1], [0, 0, 0.1], [1, 1, 2], [0, 0, 1]])
    testSort = np.array([[0, 0, 0.1], [0, 0, 1],[1, 1, 1], [1, 1, 2],[9, 9, 9], [10, 9, 8]])

    # test error messaging
    def test_integrateZenith(self):
        estInt = RAiDER.delay._integrateZenith(self.zs, self.pw)
        self.assertTrue(math.isclose(np.round(estInt,2),self.trueInt))
    def test_int_fcn(self):
        estInt = RAiDER.delay.int_fcn(self.pw, self.step)
        self.assertTrue(math.isclose(np.round(estInt,2),self.trueInt))
    def test_getZenithLookVecs(self):
        lvs = RAiDER.losreader._getZenithLookVecs(self.lats, self.lons, self.hgts, self.zref)
        self.assertTrue(np.allclose(np.round(lvs, 4),self.zlvsTrue))
    def test_get_lengths(self):
        lengths = RAiDER.delayFcns._get_lengths(self.vec1)
        self.assertTrue(np.allclose(lengths,np.linalg.norm(self.vec1, axis=1)))
    def test_get_lengths_wrongOrientation(self):
        self.assertRaises(RuntimeError, RAiDER.delayFcns._get_lengths, self.vec1.T)
    def test_get_lengths_wrongSize(self):
        self.assertRaises(RuntimeError, RAiDER.delayFcns._get_lengths, self.vec1[:,:2])
    def test_compute_ray(self):
        ray = RAiDER.delayFcns._compute_ray(100, np.array([0, 0, 0]), self.vec2, 0.1)
        self.assertTrue(len(ray)==len(self.rayTrue))
        self.assertTrue(np.allclose(ray,self.rayTrue))
    def test_sortSP(self):
        xSort = RAiDER.delayFcns.sortSP(self.testArr1)
        self.assertTrue(np.allclose(xSort, self.testSort))
    def test_getUnitLVs(self):
        lv,lengths = RAiDER.delayFcns.getUnitLVs(np.array([0, 0, 1001.5]))
        self.assertTrue(np.allclose(lv,np.array([0, 0, 1])))
        self.assertTrue(lengths==1001.5)

        

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

