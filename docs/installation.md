To install the required packages in a virtual environment, run the following command:



# 🛠 Installation

## From PyPI:
```bash
pip install woodtapper
```
> **Compatibility note:**
> WoodTapper includes a compiled Cython component that depends on the binary interface of `scikit-learn`. If `scikit-learn` is already installed in your environment, the prebuilt PyPI wheel may be incompatible with it. WoodTapper is currently pinned to `scikit-learn` 1.6.1: versions 1.9 and newer are not compatible with this component. To use another compatible version, rebuild WoodTapper locally with:
>
> ```bash
> pip uninstall -y woodtapper
> pip install -U pip setuptools wheel
> pip install -U Cython pybind11
> pip install --no-binary=woodtapper --no-build-isolation woodtapper
> ```

## From source:
```bash
git clone https://github.com/artefactory/woodtapper.git
cd woodtapper
pip install -e .[dev,docs]
```
> **Warning:** If you are a Windows user, you need to have a C/C++ compiler before installing woodtapper.

## Dependencies

WoodTapper requires the following:

* Python (>=3.11,<3.13)
* Numpy (>=2.3.1)
* Scikit-learn (==1.6.1)

# Contributing
You are welcome to contribute to the project ! You can help in various ways:

* raise issues
* resolve issues already opened
* develop new features
* provide additional examples of use
* fix typos, improve code quality
* develop new tests

We recommend to first open an [issue](https://github.com/artefactory/woodtapper/issues) to discuss your ideas.
