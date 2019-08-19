#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert, Jeremy Maurer, and Piyush Agram
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
A note about setup and installation of this module:
This package uses GDAL and g++, both of which can be tricky to set up correctly.
GDAL in particular will often break after installing a new program
If you receive error messages such as the following: 
ImportError: ~/anaconda3/envs/RAiDER/lib/python3.7/site-packages/matplotlib/../../../libstdc++.so.6: version `CXXABI_1.3.9' not found (required by ~/anaconda3/envs/RAiDER/lib/python3.7/site-packages/matplotlib/ft2font.cpython-37m-x86_64-linux-gnu.so)
or
ImportError: libtiledb.so.1.6.0: cannot open shared object file: No such file or directory
or
***cmake: /u/kriek1/maurer/software/anaconda3/envs/RAiDER/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by cmake)***
try running the following commands within your RAiDER conda environment:
conda update --force-install libstdcxx-ng
conda update --force-install gdal libgdal
'''
import numpy as np
import os
import glob
import subprocess as subp

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# Parameter defs
CWD = os.getcwd()
GEOMETRY_DIR = os.path.join(CWD, "tools/bindings/geometry")
CPP_DIR = os.path.join(GEOMETRY_DIR, "cpp/classes")
CYTHON_DIR = os.path.join(GEOMETRY_DIR, "cython/Geo2rdr")

extensions = [
     Extension(
       name="Geo2rdr",
       sources= glob.glob(os.path.join(CPP_DIR, "*/*.cc")) + 
                glob.glob(os.path.join(CYTHON_DIR, "*.pyx")), 
       include_dirs=[np.get_include()] + 
                    [os.path.join(CPP_DIR, "Geometry"),
                     os.path.join(CPP_DIR, "Utility"), 
                     os.path.join(CPP_DIR, "Orbit")],
       extra_compile_args=['-std=c++11'],
       extra_link_args=['-lm'],
       language="c++"
     )
]

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       package_dir={'tools': 'tools',
                    'RAiDER': 'tools/RAiDER',
                    'RAiDER.models': 'tools/RAiDER/models'},
       packages=['tools', 'RAiDER', 'RAiDER.models'],
       ext_modules = cythonize(extensions, quiet = True),
       scripts=['tools/bin/raiderDelay.py'])

