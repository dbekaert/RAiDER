#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert
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
import subprocess as subp

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# Parameter defs
GEOMETRY_DIR = "tools/bindings/geometry/"
CPP_SRC_DIR = GEOMETRY_DIR + '/cpp/'
GEOMETRY_LIB_DIR = "./build" 
NTHREADS = 8

# Pin the os.env variables for the compiler to be g++ (otherwise it calls gcc which throws warnings)
os.environ["CC"] = 'gcc'
os.environ["CXX"] = 'g++'
os.environ["GEOMETRY_DIR"] = GEOMETRY_DIR
os.environ["GEOMETRY_LIB_DIR"] = GEOMETRY_LIB_DIR


def getVersion():
    '''
    Load the version from a text file
    '''
    with open('version.txt', 'r') as f:
       try:
          return f.read().split('=')[-1].replace('\'', '').strip()
       except IOError:
          return "0.0.0a1"


def geomFiles(GEOMETRY_DIR):
   # geometry extension
   obj_files = ['Geometry']

   # geometry source files
   geometry_source_files = [os.path.join(GEOMETRY_DIR,"cpp/classes", f, f+'.cc') for f in obj_files]
   geometry_source_files = geometry_source_files + [os.path.join(GEOMETRY_DIR,'cython/Geo2rdr/Geo2rdr.pyx')]

   return geometry_source_files


def clsDirs(GEOMETRY_DIR):
   # geometry classes
   cls_dirs = [os.path.join(GEOMETRY_DIR, "cpp/classes/Geometry"), 
               os.path.join(GEOMETRY_DIR, "cpp/classes/Orbit"),
               os.path.join(GEOMETRY_DIR, "cpp/classes/Utility")]

   return cls_dirs


def makeCPP(geom_dir):
    '''
    Run cmake with appropriate args to compile the geometry module
    '''
    cwd = os.getcwd()
    mkdir('build')
    os.chdir('build')
    subp.call(['cmake', '.', '..' + os.sep + geom_dir + 'cpp'])
    subp.call(['make'])
    os.chdir('..')
    #cmake .  $src_dir/Geometry/cpp/ 
    #make
    #python3 $src_dir/Geometry/cython/setup.py build_ext -b $src_dir/build


def mkdir(dirName):
   import shutil
   try:
      shutil.rmtree(dirName)
   except:
      pass 
   os.mkdir(dirName)


extensions = [
     Extension(
       name="Geo2rdr",
       sources=geomFiles(GEOMETRY_DIR),
       include_dirs=[np.get_include()] + clsDirs(GEOMETRY_DIR), 
       extra_compile_args=['-std=c++11'],
       extra_link_args=['-lm'],
       library_dirs=[GEOMETRY_LIB_DIR],
       libraries=['geometry'],
       depends=["./tools/bindings/geometry/cpp/classes/Geometry/Geometry.h"],
       language="c++"
     )
]

makeCPP(GEOMETRY_DIR)

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       package_dir={'tools': 'tools',
                    'RAiDER': 'tools/RAiDER',
                    'geometry': 'build',
                    'RAiDER.models': 'tools/RAiDER/models'},
       packages=['tools', 'RAiDER', 'RAiDER.models', 'geometry'],
       #packages=[''],
       ext_modules = cythonize(extensions, quiet = True,nthreads=NTHREADS),
       scripts=['tools/bin/raiderDelay.py'])


