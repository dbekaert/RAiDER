# Unit and other tests
import numpy as np
import unittest
import delay as d
import pickle as p

class RunTests(unittest.TestCase):

    with open('test/td_lookvecs_lengths_steps.pik', 'rb') as f:
        td = p.load(f)

    def test_get_lengths(self):
        self.assertTrue(np.allclose(d._get_lengths(self.td['look_vecs']), self.td['lengths']))

if __name__=='__main__':
    unittest.main()

