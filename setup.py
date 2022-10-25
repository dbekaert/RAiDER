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
from pathlib import Path

import numpy as np
from setuptools import Extension, find_packages, setup

# Cythonize should be imported after setuptools. See:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
from Cython.Build import cythonize  # isort:skip

# Parameter defs
CWD = os.getcwd()
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
        name="RAiDER.makePoints",
        sources=glob.glob(os.path.join(UTIL_DIR, "*.pyx")),
        include_dirs=[np.get_include()]
    ),
]

setup(
    name='RAiDER',
    version=get_version(),
    description='Raytracing Atmospheric Delay Estimation for RADAR',
    long_description=(Path(__file__).parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/dbekaert/RAiDER',

    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

    python_requires='~=3.8',
    setup_requires=['pybind11>=2.5.0'],
    install_requires=[],

    package_dir={'': 'tools'},
    packages=find_packages('tools'),

    ext_modules=cythonize(
        cython_extensions,
        quiet=True,
        compiler_directives={'language_level': 3}
    ) + pybind_extensions,

    entry_points={
        'console_scripts': [
            'RAiDER = RAiDER.__main__:main'
            'generateGACOSVRT.py = RAiDER.models.generateGACOSVRT:main',
            'prepARIA.py = RAiDER.prepFromAria:prepFromAria',
            'raiderCombine.py = RAiDER.gnss.processDelayFiles:parseCMD',
            'raiderDelay.py = RAiDER.runProgram:parseCMD',
            'raiderStats.py = RAiDER.statsPlot:main',
            'raiderDownloadGNSS.py = RAiDER.downloadGNSSDelays:main',
            'raiderWeatherModelDebug.py = RAiDER.runProgram:parseCMD_weather_model_debug',
        ]
    },

    zip_safe=False,
)
