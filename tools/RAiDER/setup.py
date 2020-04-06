
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="makePoints",
    ext_modules=cythonize("makePoints.pyx"),
                include_dirs=[numpy.get_include()]
)  

