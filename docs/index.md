# WoodTapper Documentation

<div align="center">
<img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed.png" width="300">
</div>

<div align="center">

User-friendly and scalable Python package for tapping decision tree ensembles

</div>

## Overview

WoodTapper is a comprehensive Python package designed for interpretability and explainability of decision tree ensembles. It provides tools for extracting interpretable rules and generating example-based explanations from tree-based models.

The package is supported by peer-reviewed and preprint research:

> Sakho et al. (2026). *WoodTapper: a Python package for explaining decision tree ensembles*. **Journal of Open Source Software**, 11(121), 10112. [📄](https://joss.theoj.org/papers/10.21105/joss.10112) https://doi.org/10.21105/joss.10112

> Shivola et al. (2026) *OT-FairBoost: Optimal Transport-Guided Gradient Boosting for Fairness Regularization on Tabular Data*, [📄](https://arxiv.org/pdf/2607.28014)

## Quick Start

- [Installation guide](./installation.md)
- [Mathematical foundations](./0_tutorials.md)

## Modules

### Rules Extraction

Extract interpretable decision rules from tree-based models. **Tutorials:** [Rules Extraction](./1_tutorials.md)

- [Classification Rules](./references/classification_extractors.md)
- [Regression Rules](./references/regression_extractors.md)



### Fairness

Gradient-Boosting with fairness-aware training. **Tutorials:** [Fairness](./3_tutorials_fairness.md)

- [Classification Fairness](./references/classification_fairness.md)
- [Regression Fairness](./references/regression_fairness.md)



### Example-Based Explainability

Generate instance-level explanations. **Tutorials:** [Example-Based Explainability](./2_tutorials_example_exp.md)

- [Classification Explanations](./references/classification_explanation.md)
- [Regression Explanations](./references/regression_explanation.md)


## License

WoodTapper is distributed under the MIT License.

Copyright (c) 2025 artefactory.

This permissive license allows you to use, copy, modify, merge, publish,
distribute, sublicense, and sell copies of the software, provided that the
copyright notice and permission notice are included in all copies or
substantial portions of the software.

## Acknowledgements
`WoodTapper` has been developed through a collaboration between the **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.


<div style="display: flex; justify-content: center; align-items: center; gap: 24px; flex-wrap: wrap;">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/" style="width: 180px; height: 70px; display: flex; align-items: center; justify-content: center;">
    <img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_arc.png" style="max-width: 180px; max-height: 70px; object-fit: contain;" />
  </a>
  <a href="https://www.lpsm.paris/" style="width: 180px; height: 70px; display: flex; align-items: center; justify-content: center;">
    <img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_LPSM.jpg" style="max-width: 180px; max-height: 70px; object-fit: contain;" />
  </a>
</div>

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

```bibtex
@article{shilova2026ot,
  title={OT-FairBoost: Optimal Transport-Guided Gradient Boosting for Fairness Regularization on Tabular Data},
  author={Shilova, Veronika and Sakho, Abdoulaye and Boumoussou, Younes and Risser, Laurent and Loubes, Jean-Michel and Malherbe, Emmanuel},
  journal={arXiv preprint arXiv:2607.28014},
  year={2026}
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
