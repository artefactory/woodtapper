from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "extract_rules.Splitter.QuantileSplitter",  # Full module path
        ["extract_rules/Splitter/_QuantileSplitter.pyx"],  # Path to .pyx file
    )
]

setup(
    name="Splitter",
    packages=["Splitter"],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
