<div align="center">


<picture>
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed.png">
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed_light.png" >
<img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed_light.png" width="300">

</picture>

*User-friendly Python toolbox for interpreting and manipulating decision tree ensembles from scikit-learn*

[![CI Status](https://github.com/artefactory/woodtapper/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/woodtapper/actions/workflows/ci.yaml?query=branch%3Amain)
[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/woodtapper/blob/main/.pre-commit-config.yaml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](#-documentation)

[![License](https://img.shields.io/github/license/artefactory/woodtapper)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/woodtapper?label=python)](https://pypi.org/project/woodtapper/)
[![PyPI Version](https://img.shields.io/pypi/v/woodtapper.svg)](https://pypi.org/project/woodtapper/)


</div>

## 🪵 Key Features
WoodTapper is a Python toolbox that provides:

- Rule extraction from tree-based ensembles: Generates a final estimator composed of a sequence of simple rule-based on features and thresholds.

- Example-based explanations: Connects predictions to a small set of representative samples, returning the most similar examples along with their target values.

Detailed information about the modules can be found [here.](https://artefactory.github.io/woodtapper/0_tutorials/)

WoodTapper is fully compatible with scikit-learn tree ensemble models.

## 🛠 Installation

**From PyPi**:
```bash
pip install woodtapper
```

**From this repository, within a pip/conda/mamba environment (python=3.12)**:
```bash
pip install -r requirements.txt
pip install -e '.[dev]'
```

## 🌿 WoodTapper RulesExtraction module
```python
from woodtapper.extract_rules import SirusClassifier
from woodtapper.extract_rules.visualization import show_rules

sirus = SirusClassifier(n_estimators=1000,max_depth=2,
                          quantile=10,p0=0.01, random_state=0)
sirus.fit(X_train,y_train)
y_pred_sirus = sirus.predict(X_test)
show_rules(Sirus,max_rules=10) # Show rules
```

## 🌱 WoodTapper ExampleExplanation module
```python
from woodtapper.example_sampling import RandomForestClassifierExplained

rf_explained = RandomForestClassifierExplained(n_estimators=100)
rf_explained.fit(X_train,y_train)
Xy_explain = rf_explained.explanation(X_test) # Get the 5 most similar samples (and target) for each test sample
```

## 🙏 Acknowledgements

This work was done through a partnership between the **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_arc.png" height="80" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.lpsm.paris/">
    <img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos//logo_LPSM.jpg" height="95" />
  </a>
</p>


## 📜 Citation

If you find the code useful, please consider citing us:

```bibtex
@misc{woodtapper,
  title        = {WoodTapper: a Python package for tapping decision tree ensembles},
  author       = {Sakho, Abdoulaye and AOUAD, Jad and Malherbe, Emmanuel and Scornet, Erwan},
  year         = {2025},
  howpublished = {\url{https://github.com/artefactory/woodtapper}},
}
```
For SIRUS methodology, consider citing:
```
@article{benard2021sirus,
  title={Sirus: Stable and interpretable rule set for classification},
  author={Benard, Clement and Biau, Gerard and Da Veiga, Sebastien and Scornet, Erwan},
  year={2021}
}
```
