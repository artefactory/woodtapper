from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='_QuantileSplitter',
    ext_modules=cythonize("_QuantileSplitter.pyx"),
    include_dirs=[numpy.get_include()]
)