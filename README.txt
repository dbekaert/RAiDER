To test out the code, cd into the main directory (git_raytracing), which should contain the test/ folder and the Geometry/ folder. Run 

cd test
vi/nano/emacs test_scenario.py (whichever editor you like to use)

Then choose one of the scenarios to test run. scenario_0 uses ERA-I and is a single point, scenario_5 uses WRF and is a small region, etc. 
You can then run 

cd ..
python -m unittest

and the code will run that scenario in debug mode, which should produce a 
pdf plot of the weather model at 500 and 15000 meters, along with the 
weather model in a pickled object called weatherObj.dat. If the test 
completes successfully there will be a set of files in the test/ folder 
containing the estimated delay in meters in ENVI-format files. You can run

gdal2isce_xml.py -i <filename>
mdx.py <filename>

to convert the files to ISCE-readable format and view them. 


##############################
######## Installation ########
##############################

Installing Geometry Module:

1- create a directory somewhre that you want to build the module
mkdir build
cd build

2- setup the following variables:

setenv GEOMETRY_DIR  /home/fattahi/tools/ray_tracing/raytracing_current/Geometry
setenv GEOMETRY_LIB_DIR /home/fattahi/tools/ray_tracing/build

append-path    PYTHONPATH  /home/fattahi/tools/ray_tracing/build
append-path    LD_LIBRARY_PATH  /home/fattahi/tools/ray_tracing/build


3- inside your build directory run the following command:

cmake .  /home/fattahi/tools/ray_tracing/raytracing_current/Geometry/cpp/ 
make
python3 /home/fattahi/tools/ray_tracing/raytracing_current/Geometry/cython/setup.py build_ext -b /home/fattahi/tools/ray_tracing/build

4- Now you should be able to call the geometry module from python e.g.:

import Geo2rdr

5- Look at an example in the test directory

