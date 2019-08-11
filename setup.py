#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from distutils.core import setup, Extension
import numpy
from Cython.Build import cythonize
import os

# Paths to code requiring compilation
print('Installing RAiDER')

## geometry extension
obj_files = ['geometry']
geometry_dir='tools/bindings/geometry'
geometry_lib_dir = 'build'

# geometry source files
geometry_source_files = [os.path.join(geometry_dir,"cpp/classes", f, f+'.cc') for f in obj_files]
geometry_source_files = geometry_source_files + [os.path.join(geometry_dir,'cython/Geo2rdr/Geo2rdr.pyx')]
# geometry classes
cls_dirs = [os.path.join(geometry_dir, "cpp/classes/Geometry"),
            os.path.join(geometry_dir, "cpp/classes/Orbit"),
            os.path.join(geometry_dir, "cpp/classes/Utility")]

# Pin the os.env variables for the compiler to be g++ (otherwise it calls gcc which throws warnings)
os.environ["CC"] = 'g++'
os.environ["CXX"] = 'g++'

"""
extensions = [Extension(name="RAiDER.demo", sources=geometry_source_files,
                        include_dirs=[numpy.get_include()] + cls_dirs,
                        extra_compile_args=['-std=c++11'],
                        extra_link_args=['-lm'],
                        library_dirs=[geometry_lib_dir],
                        libraries=['geometry'],
                        language="c++")]
"""
    
extensions = [Extension(name="RAiDER.demo", sources=geometry_source_files,
                        include_dirs=[numpy.get_include()] + cls_dirs,
                        language="c++")]

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       ext_modules = cythonize(extensions, nthreads=8),
       packages=['RAiDER'],
       package_dir={'RAiDER': 'tools/RAiDER'},
       scripts=['tools/bin/raiderDelay.py'])


