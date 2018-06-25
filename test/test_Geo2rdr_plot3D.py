import Geo2rdr
import numpy as np
import isce
import isceobj
from iscesys.Component.ProductManager import ProductManager as PM
import matplotlib.pyplot as plt
from datetime import datetime
import time

def datetime2year(dt): 
    return dt.sec
    #year_part = dt - datetime(year=dt.year, month=1, day=1)
    #year_length = datetime(year=dt.year+1, month=1, day=1) - datetime(year=dt.year, month=1, day=1)
    #return dt.year + year_part/year_length


############################
#Read  the state vectors
pm=PM()
pm.configure()
xmlname = "/u/k-data/fattahi/Kurdistan/174/master/IW1.xml"

obj = pm.loadProduct(xmlname)

numSV = len(obj.orbit.stateVectors)

t = np.ones(numSV)
x = np.ones(numSV)
y = np.ones(numSV)
z = np.ones(numSV)
vx = np.ones(numSV)
vy = np.ones(numSV)
vz = np.ones(numSV)

for i,st in enumerate(obj.orbit.stateVectors):
    #tt = st.time
    #t[i] = datetime2year(tt)
    t[i] = st.time.second + st.time.minute*60.0
    x[i] = st.position[0]
    y[i] = st.position[1]
    z[i] = st.position[2]
    vx[i] = st.velocity[0]
    vy[i] = st.velocity[1]
    vz[i] = st.velocity[2]

print("Fifth Sensor position: ")
print(t[5], x[5], y[5], z[5])
############################
#Read  the DEM 
d2r = np.pi/180.
lat_first = 33.5*d2r
lon_first = 43.5*d2r
lat_step = -0.002*d2r
lon_step = 0.002*d2r
#heights = np.zeros((1000,1000))
heights = np.zeros((1,1))

###########################
#Instantiate Geo2rdr 
st_time = time.time()

geo2rdrObj = Geo2rdr.PyGeo2rdr()

# pass the 1D arrays of state vectors to geo2rdr object
geo2rdrObj.set_orbit(t, x, y, z, vx , vy , vz)

# set the geo coordinates: lat and lon of the start pixel, 
#                           lat and lon steps
#                           DEM heights

geo2rdrObj.set_geo_coordinate(lon_first, lat_first,
                              lon_step, lat_step,
                              heights)

# compute the radar coordinate for each geo coordinate
geo2rdrObj.geo2rdr()

# get back the line of sight unit vector
los_x, los_y, los_z = geo2rdrObj.get_los()


# get back the slant ranges
slant_range = geo2rdrObj.get_slant_range()


run_time = time.time() - st_time
print("geo2rdr took {0} seconds".format(run_time))

#plt.imshow(slant_range)
#plt.show()

print(slant_range)
print(los_x)
# To get the Sensor Position, one nededs to add the target position to the vector from target to Sensor
#Sensor = Target + slant_range*los
import pyproj
ecef = pyproj.Proj(proj='geocent')
lla = pyproj.Proj(proj='latlong')

r2d = 180.0/np.pi
Target =  pyproj.transform(lla, ecef, lon_first*r2d, lat_first*r2d, 0)
print("Target:")
print(Target[0])
Sensor = (Target[0] + los_x[0][0]*slant_range[0][0], Target[1] + los_y[0][0]*slant_range[0][0] ,Target[2] + los_z[0][0]*slant_range[0][0])

print("Sensor: ")
print(Sensor)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z)
ax.plot([Sensor[0]], [Sensor[1]], [Sensor[2]], "-o")
plt.show()


