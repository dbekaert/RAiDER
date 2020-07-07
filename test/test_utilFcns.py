# Unit and other tests
import os
import unittest
from datetime import time

from osgeo import gdal
import numpy as np

from RAiDER.utilFcns import makeDelayFileNames


class FcnTests(unittest.TestCase):

    # test sind
    theta1 = np.array([0,30,90,180])
    truev1 = np.array([0,0.5,1,0])

    # test cosd
    theta2 = np.array([0,60,90,180])
    truev2 = np.array([1,0.5,0,-1])

    # test tand
    theta3 = np.array([0,30,45,60])
    truev3 = np.array([0,1/np.sqrt(3),1,np.sqrt(3)])

    # test gdal_open
    fname1 = '../test/test_geom/lat.rdr'
    shape1 = (45,226)

    # test writeResultsToHDF5
    lats = np.array([15.0, 15.5, 16.0, 16.5, 17.5, -40, 60, 90])
    lons = np.array([-100.0, -100.4, -91.2, 45.0, 0., -100,-100, -100])
    hgts = np.array([0., 1000., 10000., 0., 0., 0., 0., 0.])
    wet = np.zeros(lats.shape)
    hydro = np.ones(lats.shape)
    filename1 = 'dummy.hdf5'

    # test writeArrayToRaster
    array = np.transpose(np.array([np.arange(0,10)]))*np.arange(0,10)
    filename2 = 'dummy.out'

    # test error messaging
    def test_sind(self):
        from RAiDER.utilFcns import sind
        out =sind(self.theta1)
        self.assertTrue(np.allclose(out, self.truev1))

    def test_cosd(self):
        from RAiDER.utilFcns import cosd
        out =cosd(self.theta2)
        self.assertTrue(np.allclose(out, self.truev2))

    def test_tand(self):
        from RAiDER.utilFcns import tand
        out =tand(self.theta3)
        self.assertTrue(np.allclose(out, self.truev3))

    def test_gdal_open(self):
        from RAiDER.utilFcns import gdal_open
        out = gdal_open(self.fname1, False)
        self.assertTrue(np.allclose(out.shape, self.shape1))

    def test_writeResultsToHDF5(self):
        from RAiDER.utilFcns import writeResultsToHDF5
        writeResultsToHDF5(self.lats, self.lons, self.hgts, self.wet, self.hydro, self.filename1)
        import h5py
        with h5py.File(self.filename1, 'r') as f:
            lats = np.array(f['lat'])
            hydro = np.array(f['hydroDelay'])
            delayType = f.attrs['DelayType']
        self.assertTrue(np.allclose(lats, self.lats))
        self.assertTrue(np.allclose(hydro, self.hydro))
        self.assertEqual(delayType, 'Zenith')

    def test_writeArrayToRaster(self):
        from RAiDER.utilFcns import writeArrayToRaster
        writeArrayToRaster(self.array, self.filename2)
        ds = gdal.Open(self.filename2, gdal.GA_ReadOnly)
        b = ds.GetRasterBand(1)
        d = b.ReadAsArray()
        nodata = b.GetNoDataValue()
        self.assertTrue(np.allclose(d, self.array))
        self.assertEqual(nodata, 0)

    def test_makePoints1D_Python(self):
        from RAiDER.utilFcns import makePoints1D
        test_result = makePoints1D(1000., np.array([0, 0, 0]), np.array([0, 0, 1]), 5)
        true_ray = np.stack([np.zeros((200,)), np.zeros((200,)), np.arange(0,1000,5)], axis = -1).T
        self.assertTrue(np.allclose(test_result, true_ray))

    def test_makePoints3D_Python_dim(self):
        from RAiDER.utilFcns import makePoints3D
        sp = np.zeros((3, 3, 3, 3))
        sp[:,:,1,2] = 10
        sp[:,:,2,2] = 100
        slv = np.zeros((3,3,3,3))
        slv[0,:,:,2] = 1
        slv[1,:,:,1] = 1
        slv[2,:,:,0] = 1
        test_result = makePoints3D(100., sp, slv, 5)
        self.assertTrue(test_result.ndim==5)

    def test_makePoints3D_Python_values(self):
        from RAiDER.utilFcns import makePoints3D
        sp = np.zeros((3, 3, 3, 3))
        sp[:,:,1,2] = 10
        sp[:,:,2,2] = 100
        slv = np.zeros((3,3,3,3))
        slv[0,:,:,2] = 1
        slv[1,:,:,1] = 1
        slv[2,:,:,0] = 1
        test_result = makePoints3D(100., sp, slv, 5)
        df = np.loadtxt('test_result_makePoints3D.txt')
        shape = (3,3,3,3,20)
        true_rays = df.reshape(shape)
        self.assertTrue(np.allclose(test_result, true_rays))

    def test_makePoints1D_Cython(self):
        from RAiDER.makePoints import makePoints1D
        test_result = makePoints1D(1000., np.array([0., 0., 0.]), np.array([0., 0., 1.]), 5.)
        true_ray = np.stack([np.zeros((200,)), np.zeros((200,)), np.arange(0,1000,5)], axis = -1).T
        self.assertTrue(np.allclose(test_result, true_ray))

    def test_makePoints3D_Cython_dim(self):
        from RAiDER.makePoints import makePoints3D
        sp = np.zeros((3, 3, 3, 3))
        sp[:,:,1,2] = 10
        sp[:,:,2,2] = 100
        slv = np.zeros((3,3,3,3))
        slv[0,:,:,2] = 1
        slv[1,:,:,1] = 1
        slv[2,:,:,0] = 1
        test_result = makePoints3D(100., sp, slv, 5)
        self.assertTrue(test_result.ndim==5)

    def test_makePoints3D_Cython_values(self):
        from RAiDER.makePoints import makePoints3D
        sp = np.zeros((3, 3, 3, 3))
        sp[:,:,1,2] = 10
        sp[:,:,2,2] = 100
        slv = np.zeros((3,3,3,3))
        slv[0,:,:,2] = 1
        slv[1,:,:,1] = 1
        slv[2,:,:,0] = 1
        test_result = makePoints3D(100., sp, slv, 5)
        df = np.loadtxt('test_result_makePoints3D.txt')
        shape = (3,3,3,3,20)
        true_rays = df.reshape(shape)
        self.assertTrue(np.allclose(test_result, true_rays))

    def test_makeDelayFileNames(self):
        self.assertEqual(
            makeDelayFileNames(None, None, "h5", "name", "dir"),
            ("dir/name_wet_ztd.h5", "dir/name_hydro_ztd.h5")
        )
        self.assertEqual(
            makeDelayFileNames(None, (), "h5", "name", "dir"),
            ("dir/name_wet_std.h5", "dir/name_hydro_std.h5")
        )
        self.assertEqual(
            makeDelayFileNames(time(1, 2, 3), None, "h5", "model_name", "dir"),
            (
                "dir/model_name_wet_01_02_03_ztd.h5",
                "dir/model_name_hydro_01_02_03_ztd.h5"
            )
        )
        self.assertEqual(
            makeDelayFileNames(time(1, 2, 3), "los", "h5", "model_name", "dir"),
            (
                "dir/model_name_wet_01_02_03_std.h5",
                "dir/model_name_hydro_01_02_03_std.h5"
            )
        )


def test_cleanUp():
    import glob
    dummy_files = glob.glob('dummy' + '*')
    [os.remove(f) for f in dummy_files]


def main():
    unittest.main()


if __name__ == '__main__':

    unittest.main()
