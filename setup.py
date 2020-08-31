#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert, Jeremy Maurer, and Piyush Agram
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import glob
import os
import re

import numpy as np
from setuptools import Extension, find_packages, setup

# Cythonize should be imported after setuptools. See:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
from Cython.Build import cythonize  # isort:skip

# Parameter defs
CWD = os.getcwd()
GEOMETRY_DIR = os.path.join(CWD, "tools", "bindings", "geometry")
CPP_DIR = os.path.join(GEOMETRY_DIR, "cpp", "classes")
CYTHON_DIR = os.path.join(GEOMETRY_DIR, "cython", "Geo2rdr")
UTIL_DIR = os.path.join(CWD, 'tools', 'bindings', 'utils')


def get_version():
    with open('version.txt', 'r') as f:
        m = re.match("""version=['"](.*)['"]""", f.read())

    assert m, "Malformed 'version.txt' file!"
    return m.group(1)


# Based on https://github.com/pybind/python_example/blob/master/setup.py
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


pybind_extensions = [
    Extension(
        'RAiDER.interpolate',
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/python_example/pull/53)
        sorted([
            'tools/bindings/interpolate/src/module.cpp',
            'tools/bindings/interpolate/src/interpolate.cpp'
        ]),
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
        ],
        extra_compile_args=['-std=c++17'],
        language='c++'
    ),
]


cython_extensions = [
    Extension(
        name="RAiDER.Geo2rdr",
        sources=[
            *glob.glob(os.path.join(CPP_DIR, "*/*.cc")),
            *glob.glob(os.path.join(CYTHON_DIR, "*.pyx"))
        ],
        include_dirs=[
            np.get_include(),
            os.path.join(CPP_DIR, "Geometry"),
            os.path.join(CPP_DIR, "Utility"),
            os.path.join(CPP_DIR, "Orbit")
        ],
        extra_compile_args=['-std=c++11'],
        extra_link_args=['-lm'],
        language="c++"
    ),
    Extension(
        name="RAiDER.makeRays",
        sources=[os.path.join(UTIL_DIR, "makeRays.pyx")],
        include_dirs=[np.get_include()]
    ),
]

setup(
    name='RAiDER',
    version=get_version(),
    description='This is the RAiDER package',
    package_dir={
        'tools': 'tools',
        '': 'tools'
    },
    packages=['tools'] + find_packages('tools'),
    ext_modules=cythonize(
        cython_extensions,
        quiet=True,
        compiler_directives={'language_level': 3}
    ) + pybind_extensions,
    scripts=[
        'tools/bin/raiderDelay.py',
        'tools/bin/raiderStats.py',
        'tools/bin/raiderDownloadGNSS.py'
    ],
    setup_requires=['pybind11>=2.5.0'],
    zip_safe=False,
)
