from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "QuantileSplitter.QuantileSplitter",       # Full module path
        ["QuantileSplitter/_QuantileSplitter.pyx"], # Path to .pyx file
    )
]

setup(
    name='QuantileSplitter',
    packages=["QuantileSplitter"],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)