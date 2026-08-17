# WoodTapper Documentation

<div align="center">
<img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed.png" width="300">
</div>

<div align="center">

User-friendly and scalable Python package for tapping decision tree ensembles

</div>

## Overview

WoodTapper is a comprehensive Python package designed for interpretability and explainability of decision tree ensembles. It provides tools for extracting interpretable rules and generating example-based explanations from tree-based models.

The package is supported by peer-reviewed research:

> Sakho et al. (2026). *WoodTapper: a Python package for explaining decision tree ensembles*. **Journal of Open Source Software**, 11(121), 10112. [📄](https://joss.theoj.org/papers/10.21105/joss.10112) https://doi.org/10.21105/joss.10112

## Quick Start

- [Installation guide](./installation.md)
- [Mathematical foundations](./0_tutorials.md)

## Modules

### Rules Extraction

Extract interpretable decision rules from tree-based models:

- [Classification Rules](./references/classification_extractors.md)
- [Regression Rules](./references/regression_extractors.md)

**Tutorials:** [Rules Extraction](./1_tutorials.md)

### Example-Based Explainability

Generate instance-level explanations:

- [Classification Explanations](./references/classification_explanation.md)
- [Regression Explanations](./references/regression_explanation.md)

**Tutorials:** [Example-Based Explainability](./2_tutorials_example_exp.md)

## Citation

If you use WoodTapper in your research, please cite:

```bibtex
@article{Sakho2026,
  doi = {10.21105/joss.10112},
  url = {https://doi.org/10.21105/joss.10112},
  year = {2026},
  publisher = {The Open Journal},
  volume = {11},
  number = {121},
  pages = {10112},
  author = {Sakho, Abdoulaye and Aouad, Jad and Gauthier, Carl-Erik and Malherbe, Emmanuel and Scornet, Erwan},
  title = {WoodTapper: a Python package for explaining decision tree ensembles},
  journal = {Journal of Open Source Software}
}
```

For SIRUS methodology:

```bibtex
@article{benard2021sirus,
  title={SIRUS: Stable and interpretable rule set for classification},
  author={Benard, Clement and Biau, Gerard and Da Veiga, Sebastien and Scornet, Erwan},
  journal = {Machine Learning},
  year = {2021}
}
```
