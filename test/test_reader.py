# Unit and other tests
import numpy as np
import unittest
import reader as rd

class RunTests(unittest.TestCase):

    temps = np.array([273.15, 240,300,1000,0,-100])
    svpTru = np.array([0,0,0,0,0,0])
    PIK = 'pickle.dat'

#    def test__run_svp_suite(self):
#       self.assertTrue(np.allclose(rd._find_svp(self.temps), self.svpTru))

#    def test_sane_interp_goodinputs(self):
#        with open(self.PIK, "rb") as f:
#             xs, ys, heights, projection, values_list, zmin  = pickle.load(f)
#        output = rd._sane_interpolate(xs, ys, heights, projection, values_list, zmin)
#        self.assertEqual(output, true_output)

if __name__=='__main__':
    unittest.main()

