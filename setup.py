#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert, Jeremy Maurer, and Piyush Agram
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

def getVersion():
    with open('version.txt', 'r') as f:
         version = f.read().split("=")[-1].replace("'",'').strip()
    return version


extensions = [
     Extension(
       name="RAiDER.Geo2rdr",
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
       version = getVersion(),
       description = 'This is the RAiDER package',
       package_dir={'tools': 'tools',
                    'RAiDER': 'tools/RAiDER',
                    'RAiDER.models': 'tools/RAiDER/models'},
       packages=['tools', 'RAiDER', 'RAiDER.models'],
       ext_modules = cythonize(extensions, quiet = True, compiler_directives={'language_level': 3}),
       scripts=['tools/bin/raiderDelay.py', 'tools/bin/raiderStats.py'])

