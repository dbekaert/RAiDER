# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert, Jeremy Maurer, and Piyush Agram
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import re
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, find_packages, setup

# Cythonize should be imported after setuptools. See:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
from Cython.Build import cythonize  # isort:skip

# Parameter defs
UTIL_DIR = Path.cwd() / 'tools' / 'bindings' / 'utils'


def get_version():
    with open('version.txt', 'r') as f:
        m = re.match("""version=['"](.*)['"]""", f.read())

    assert m, "Malformed 'version.txt' file!"
    return m.group(1)


pybind_extensions = [
    Pybind11Extension(
        'RAiDER.interpolate',
        [
            'tools/bindings/interpolate/src/module.cpp',
            'tools/bindings/interpolate/src/interpolate.cpp'
        ],
    ),
]

cython_extensions = cythonize(
    [
        Extension(
            name="RAiDER.makePoints",
            sources=[str(f) for f in UTIL_DIR.glob("*.pyx")],
            include_dirs=[np.get_include()]
        ),
    ],
    quiet=True,
    compiler_directives={'language_level': 3}
)

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
    # install_requires=[],

    package_dir={'': 'tools'},
    packages=find_packages('tools'),

    ext_modules=cython_extensions + pybind_extensions,
    cmdclass={"build_ext": build_ext},

    entry_points={
        'console_scripts': [
            'generateGACOSVRT.py = RAiDER.models.generateGACOSVRT:main',
            'raiderDelay.py = RAiDER.runProgram:main',
            'prepARIA.py = RAiDER.prepFromAria:main',
            'raiderCombine.py = RAiDER.gnss.processDelayFiles:main',
            'raiderStats.py = RAiDER.statsPlot:main',
            'raiderDownloadGNSS.py = RAiDER.downloadGNSSDelays:main',
        ]
    },
    zip_safe=False,
)
