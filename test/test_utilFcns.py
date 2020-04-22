# Unit and other tests
import datetime
import gdal
import math
import numpy as np
import os
import pandas as pd
import unittest


class FcnTests(unittest.TestCase):

    theta1 = np.array([0,30,90,180])
    truev1 = np.array([0,0.5,1,0])

    # test error messaging
    def test_sind(self):
        from RAiDER.utilFcns import sind
        out =sind(self.theta1)
        self.assertTrue(np.allclose(out, self.truev1))

  


def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

