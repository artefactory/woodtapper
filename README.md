<div align="center">


<picture>
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed.png">
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed_light.png" >
<img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed_light.png" width="300">

</picture>

*User-friendly Python toolbox for interpreting and manipuling decision tree ensembles from scikit-learn*

[![CI Status](https://github.com/artefactory/woodtapper/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/woodtapper/actions/workflows/ci.yaml?query=branch%3Amain)
[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/choice-learn/blob/main/.pre-commit-config.yaml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](#-documentation)

[![License](https://img.shields.io/github/license/artefactory/woodtapper)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/woodtapper?label=python)](https://pypi.org/project/woodtapper/)
[![PyPI Version](https://img.shields.io/pypi/v/woodtapper.svg)](https://pypi.org/project/woodtapper/)


</div>

WoodTapper is a Python toolbox for interpretable and manipulable ensemble of tree-based models, fully compatible with scikit-learn forests and boosting models. 

## 🪵 Key Features
- Rule extraction from tree-based ensembles.
- Example-based explanation module that links predictions to a small set of representative samples.


## 🛠 Installation

**From PyPi**:
```bash
pip install woodtapper
```

**From this repository, within a pip/conda/mamba environment (python=3.12)**:
````bash
pip install -r requirements.txt
pip install -e '.[dev]'
```

## 🌿 WoodTapper RulesExtraction module
```python
## RandomForestClassifier rules extraction
from woodtapper.extract_rules import SirusClassifier

SIRUS = SirusClassifier(n_estimators=1000,max_depth=2,
                          quantile=10,p0=0.01, random_state=0)
SIRUS.fit(X_train,y_train)
y_pred_sirus = SIRUS.predict(X_test)
```

## 🌱 WoodTapper ExampleExplanation module
```python
## RandomForestClassifier rules extraction
from woodtapper.example_sampling import RandomForestClassifierExplained

RFExplained = RandomForestClassifierExplained(n_estimators=100)
RFExplained.fit(X_train,y_train)
example_explanation = RFExplained.explanation(X_test) # Get the 5 most similar samples for each test sample
```

## 🙏 Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

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

If you find the code usefull, please consider citing us :

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
