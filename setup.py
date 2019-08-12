#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os
import subprocess as subp

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# Parameter defs
GEOMETRY_DIR = "tools/bindings/geometry/"
CPP_SRC_DIR = GEOMETRY_DIR + '/cpp/'
GEOMETRY_LIB_DIR = "build" 
NTHREADS = 8

# Pin the os.env variables for the compiler to be g++ (otherwise it calls gcc which throws warnings)
os.environ["CC"] = 'gcc'
os.environ["CXX"] = 'g++'


def getVersion():
    '''
    Load the version from a text file
    '''
    with open('version.txt', 'r') as f:
       try:
          return f.read().split('=')[-1].replace('\'', '').strip()
       except IOError:
          return "0.0.0a1"

def srcFiles(GEOMETRY_DIR, GEOMETRY_LIB_DIR):
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
    #python3 $src_dir/Geometry/cython/setup.py build_ext -b /home/fattahi/tools/ray_tracing/build


def mkdir(dirName):
   import shutil
   try:
      shutil.rmtree(dirName)
   except:
      pass 
   os.mkdir(dirName)


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
       sources=srcFiles(GEOMETRY_DIR, GEOMETRY_LIB_DIR),
       include_dirs=[np.get_include()] + clsDirs(GEOMETRY_DIR), 
       extra_compile_args=['-std=c++11'],
       extra_link_args=['-lm'],
       library_dirs=[GEOMETRY_LIB_DIR],
       libraries=['geometry'],
       language="c++")
]

makeCPP(GEOMETRY_DIR)

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       package_dir={'tools': 'tools',
                    'RAiDER': 'tools/RAiDER',
                    'RAiDER.models': 'tools/RAiDER/models'},
       packages=['tools', 'RAiDER', 'RAiDER.models'],
       #packages=[''],
       ext_modules = cythonize(extensions, quiet = True,nthreads=NTHREADS),
       scripts=['tools/bin/raiderDelay.py'])


