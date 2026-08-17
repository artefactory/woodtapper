To install the required packages in a virtual environment, run the following command:



# 🛠 Installation

## From PyPi:
```bash
pip install woodtapper
```
> **Warning (scikit-learn already installed):**
> If you install `woodtapper` in an environment where `scikit-learn` is already present, the prebuilt PyPI wheel may not be compatible with your existing `scikit-learn` binary. In that case, reinstall `woodtapper` from source so it is compiled against the `scikit-learn` version in your environment:
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
* Scikit-learn (>=1.6.1)

## Contributing
You are welcome to contribute to the project ! You can help in various ways:

* raise issues
* resolve issues already opened
* develop new features
* provide additional examples of use
* fix typos, improve code quality
* develop new tests

We recommend to first open an [issue](https://github.com/artefactory/woodtapper/issues) to discuss your ideas.

## 📜 Citation

If you find the code useful, please consider citing us:

```bibtex
@article{Sakho2026,
doi = {10.21105/joss.10112},
url = {https://doi.org/10.21105/joss.10112},
year = {2026}, publisher = {The Open Journal},
volume = {11},
number = {121},
pages = {10112},
author = {Sakho, Abdoulaye and Aouad, Jad and Gauthier, Carl-Erik and Malherbe, Emmanuel and Scornet, Erwan},
title = {WoodTapper: a Python package for explaining decision tree ensembles},
journal = {Journal of Open Source Software} }
```
For SIRUS methodology, consider citing:
```bibtex
@article{benard2021sirus,
  title={Sirus: Stable and interpretable rule set for classification},
  author={Benard, Clement and Biau, Gerard and Da Veiga, Sebastien and Scornet, Erwan},
  year={2021}
}

### License

The use of this software is under the MIT license.
