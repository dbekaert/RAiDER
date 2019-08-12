#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize
import os


def getVersion():
    '''
    Load the version from a text file
    '''
    with open('version.txt', 'r') as f:
       try:
          return f.read().split('=')[-1].replace('\'', '').strip()
       except IOError:
          return "0.0.0a1"


def srcFiles():
   # geometry extension
   GEOMETRY_DIR = "tools/bindings/geometry/"
   GEOMETRY_LIB_DIR = "build" 
   obj_files = ['geometry']

   # geometry source files
   geometry_source_files = [os.path.join(GEOMETRY_DIR,"cpp/classes", f, f+'.cc') for f in obj_files]
   geometry_source_files = geometry_source_files + [os.path.join(GEOMETRY_DIR,'cython/Geo2rdr/Geo2rdr.pyx')]

   return geometry_source_files


def clsDirs():
   # geometry classes
   GEOMETRY_DIR = "tools/bindings/geometry/"
   cls_dirs = [os.path.join(GEOMETRY_DIR, "cpp/classes/Geometry"), 
               os.path.join(GEOMETRY_DIR, "cpp/classes/Orbit"),
               os.path.join(GEOMETRY_DIR, "cpp/classes/Utility")]

   return cls_dirs


# Pin the os.env variables for the compiler to be g++ (otherwise it calls gcc which throws warnings)
os.environ["CC"] = 'g++'
os.environ["CXX"] = 'g++'

"""
extensions = [Extension(name="RAiDER.demo", sources=geometry_source_files,
                        include_dirs=[np.get_include()] + cls_dirs,
                        extra_compile_args=['-std=c++11'],
                        extra_link_args=['-lm'],
                        library_dirs=[geometry_lib_dir],
                        libraries=['geometry'],
                        language="c++")]
    
extensions = [Extension(name="RAiDER.extension", sources=geometry_source_files,
                        include_dirs=[np.get_include()] + cls_dirs,
                        language="c++")]
"""
extensions = [
     Extension(
       name="Geo2rdr",
       sources=srcFiles(),
       include_dirs=[np.get_include()] + clsDirs(), 
       extra_compile_args=['-std=c++11'],
       extra_link_args=['-lm'],
       library_dirs=[GEOMETRY_LIB_DIR],
       libraries=['geometry'],
       language="c++")
]

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       cmdclass={'build_ext': Cython.Build.build_ext},
       ext_modules = cythonize(extensions, quiet = True,nthreads=8),
       zip_safe=False,
       packages=['RAiDER', 'RAiDER.models', 'RAiDER.geometry'],
       package_dir={'RAiDER': 'tools/RAiDER','RAiDER.models': 'tools/RAiDER/models', 'RAiDER.geometry': 'tools/bindings/geometry/'},
       scripts=['tools/bin/raiderDelay.py'])


