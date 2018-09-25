# Unit and other tests
import datetime
import delay
import traceback
import erai

class TimeTests():

    latfile = 'test_geom/lat.rdr'
    lonfile = 'test_geom/lon.rdr'
    losfile = 'test_geom/los.rdr'
    demfile = 'test_geom/warpedDEM.dem'
    heights = ('dem', demfile)
#    heights = ('download', None)
    los = ('los', losfile)

    wmtype = 'dill' 
    wmtype = None 
    if wmtype is None:
        wm = {'type': erai.Model, 'files': None,
                'name': 'ERA-I'}
    elif wmtype =='dill':
        wm = {'type': 'dill', 'files': None,
              'name': 'ERA-I'}

    print('Weather model: {}'.format(wm))

    test_time = datetime.datetime(2018, 1, 1, 0, 48, 0)

    # test error messaging
    def time_tropo_smallArea(self):
        print(self.test_time)
        print(self.wm)
        print(self.losfile)
        delay.tropo_delay(los = self.los, 
                          lat = self.latfile, 
                          lon = self.lonfile, 
                          heights = self.heights, 
                          weather = self.wm, 
                          time = self.test_time, 
                          parallel=False, 
                          verbose = True)


if __name__=='__main__':

    errFileName = 'errors.txt'
    timeFileName = 'timing_result_smallArea.txt'

#    import cProfile
    import os

    # profile the code
#    pr = cProfile.Profile()
    test = TimeTests()

    # time the process
#    pr.enable()
    try:
        test.time_tropo_smallArea()
    except Exception as e:
        print('Something failed: ')
        print(e)
        print('See {}'.format(errFileName))
        with open(errFileName, 'w') as f:
            f.write(str(e))
            f.write(traceback.format_exc())
#    pr.disable()

    # write results to file
#    pr.dump_stats(timeFileName)

