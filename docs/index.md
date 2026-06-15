

# Welcome to the WoodTapper documentation!

<div align="center">
<img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed.png" width="300">
</div>

<div align="center">
User-friendly and scalable Python package for tapping decision tree ensembles
</div>

WoodTapper is supported by a peer reviewed publication:
> &nbsp; &nbsp;  *`WoodTapper`: a Python package for explaining decision tree ensembles*, Sakho et al. (2026) [📄](https://joss.theoj.org/papers/10.21105/joss.10112)

WoodTapper is a machine learning toolbox for investigating tree-based models.
In this documentation you will find examples to be quickly getting started as well as some more in-depth example.

## Installation

[Getting started.](./installation.md)

## Tutorials

[**The mathematical formulation of WoodTapper modules are available here.**](./0_tutorials.md)

Example tutorials are also available for each module:

  - [Tutorials for Rules Extraction](1_tutorials.md)
  - [Tutorials for Example-based Explainability](2_tutorials_example_exp.md)


## What's in there ?

Here is a quick overview of the different functionalities offered by WoodTapper. Further details are given in the rest of the documentation.

### Rules Extractors
- [Classification Rules Extractors](./references/classification_extractors.md)
- [Regression Rules Extractors](./references/regression_extractors.md)

### Example Explanation
- [Classification Example Explanation](./references/classification_explanation.md)
- [Regression Example Explanation](./references/regression_explanation.md)


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
```
