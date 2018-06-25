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

