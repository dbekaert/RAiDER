#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize
import os


def getVersion():
    '''
    Load the version from a text file
    '''
    with open('version.txt', 'r') as f:
        return f.read().split('=')[-1].replace('\'', '').strip()
    except IOError:
        return "0.0.0a1"


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
extensions = [Extension(name="RAiDER.extension", sources=geometry_source_files,
                        include_dirs=[np.get_include()] + cls_dirs]

setup (name = 'RAiDER',
       version = '1.0',
       description = 'This is the RAiDER package',
       cmdclass={'build_ext': Cython.Build.build_ext},
       ext_modules = cythonize(extensions, nthreads=8),
       zip_safe=False,
#       setup_requires=['numpy>=1.0', 'cython>=0.24.1'],
#       install_requires=['numpy>=1.0', 'nose>=0.11', 'cython>=0.24.1'],
       packages=['RAiDER', 'RAiDER.models', 'RAiDER.geometry'],
       package_dir={'RAiDER': 'tools/RAiDER','RAiDER.models': 'tools/RAiDER/models', 'RAiDER.geometry': 'tools/bindings/geometry/'},
       scripts=['tools/bin/raiderDelay.py'])

'''

setup(
    package_dir={'pycydemo': 'pycydemo'},
    packages=['pycydemo', 'pycydemo.tests'],
    ext_modules=[Extension(
        'pycydemo.extension',
        sources=['pycydemo/extension.pyx'],
        include_dirs=[np.get_include()],
    )],
    zip_safe=False,
    setup_requires=['numpy>=1.0', 'cython>=0.24.1'],
    install_requires=['numpy>=1.0', 'nose>=0.11', 'cython>=0.24.1'],
)

'''

