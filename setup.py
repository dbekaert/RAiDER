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
import platform
import re
import subprocess
import sys
import sysconfig

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# Parameter defs
GEOMETRY_DIR = "tools/bindings/geometry/"
CPP_SRC_DIR = GEOMETRY_DIR + '/cpp/'
GEOMETRY_LIB_DIR = "build" 
NTHREADS = 8

# Pin the os.env variables for the compiler to be g++ (otherwise it calls gcc which throws warnings)
os.environ["CC"] = 'g++'
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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


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

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       packages=['RAiDER', 'RAiDER.models', 'RAiDER.geometry'],
       package_dir={'RAiDER': 'tools/RAiDER','RAiDER.models': 'tools/RAiDER/models', 'RAiDER.geometry': 'tools/bindings/geometry/'},
       cmdclass=dict(build_ext=CMakeBuild(CPP_SRC_DIR),
       #ext_modules = cythonize(extensions, quiet = True,nthreads=NTHREADS),
       ext_modules = [CMakeExtension(srcFiles(GEOMETRY_DIR, GEOMETRY_LIB_DIR)[0])],
       scripts=['tools/bin/raiderDelay.py'])


