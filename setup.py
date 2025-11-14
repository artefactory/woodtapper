from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "woodtapper.extract_rules.Splitter.QuantileSplitter",  # Full module path
        [
            "woodtapper/extract_rules/Splitter/_QuantileSplitter.pyx"
        ],  # Path to .pyx file
    ),
    Extension(
        "woodtapper.example_sampling.utils.weights",  # Full module path
        ["woodtapper/example_sampling/utils/weights.pyx"],  # Path to .pyx file
    ),
]

setup(
    name="woodtapper",  # must be unique on PyPI
    author="Abdoulaye SAKHO",
    author_email="abdoulaye.sakho@artefact.com",
    description="A Python toolbox for interpretable and explainable tree ensembles.",
    long_description=open("README.md").read(),
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
