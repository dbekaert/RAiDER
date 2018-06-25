import os
from distutils.core import setup
from distutils.extension import Extension
import numpy
from Cython.Build import cythonize
#source_dir = "/home/fattahi/tools/geometry/Geo2rdr/cpp/classes"
GEOMETRY_DIR =  os.environ['GEOMETRY_DIR'] #"/home/fattahi/tools/ray_tracing/raytracing_current/Geometry"
GEOMETRY_LIB_DIR = os.environ['GEOMETRY_LIB_DIR'] #"/home/fattahi/tools/ray_tracing/build" 
obj_files = ['Geometry']

source_files = [os.path.join(GEOMETRY_DIR,"cpp/classes", f, f+'.cc') for f in obj_files]
source_files = source_files + [os.path.join(GEOMETRY_DIR,'cython/Geo2rdr/Geo2rdr.pyx')]

cls_dirs = [os.path.join(GEOMETRY_DIR, "cpp/classes/Geometry"), 
            os.path.join(GEOMETRY_DIR, "cpp/classes/Orbit"),
            os.path.join(GEOMETRY_DIR, "cpp/classes/Utility")]

#inc_dirs = [os.path.join(GEOMETRY_DIR,"cpp/classes", f, f+'.cc') for f in obj_files]
# Pin the os.env variables for the compiler to be g++ (otherwise it calls gcc which throws warnings)
os.environ["CC"] = 'g++'
os.environ["CXX"] = 'g++'

extensions = [
     Extension(
       name="Geo2rdr",
       sources=source_files,
       include_dirs=[numpy.get_include()] + cls_dirs, 
       extra_compile_args=['-std=c++11'],
       extra_link_args=['-lm'],
       library_dirs=[GEOMETRY_LIB_DIR],
       libraries=['geometry'],
       language="c++")
]

setup(
    name="Geometry",
    version="0.0",
    ext_modules=cythonize(extensions, quiet=True, nthreads=8)
    )

