import unittest
import cdsapi

class RunTests(unittest.TestCase):

    outname= 'download.nc'
    year = 2017
    month = 12
    day = 1
    tod = '23:00'
    area = [20, -100, 15, -90]

    def test_get_data(self):
        c = cdsapi.Client(verify=0)
        c.retrieve('reanalysis-era5-pressure-levels',
                  {'product_type':'reanalysis',
                    'format':'netcdf',
                    'pressure_level':['1','2','3','5','7','10','20','30','50','70','100','125','150','175','200','225','250','300','350','400','450','500','550','600','650','700','750','775','800','825','850','875','900','925','950','975','1000'],
                    'variable':['geopotential','relative_humidity','specific_humidity','temperature'],
                    'year':'{}'.format(self.year),
                    'month':'{}'.format(self.month),
                    'day':'{}'.format(self.day),
                    'time':'{}'.format(self.tod),
                    'area':self.area, # North, West, South, East. Default: global
                    },
                '{}'.format(self.outname))
        self.assertTrue(True)

if __name__=='__main__':
    unittest.main()


