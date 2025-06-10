# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: David Bekaert, Jeremy Maurer, and Piyush Agram
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

# Cythonize should be imported after setuptools. See:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
from Cython.Build import cythonize  # isort:skip

# Parameter defs
UTIL_DIR = Path('tools') / 'bindings' / 'utils'

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
    ext_modules=cython_extensions + pybind_extensions,
    cmdclass={"build_ext": build_ext},
)
