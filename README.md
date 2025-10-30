<div align="center">


<picture>
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed.png">
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed_light.png" >
<img src="https://raw.githubusercontent.com/artefactory/woodtapper/main/data/logos/logo_woodpecker_compressed_light.png" width="300">

</picture>

*User-friendly and scalable Python package for tapping decision tree ensembles*

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
</div>

WoodTapper is a Python toolbox  for interpretable and explainable tree ensembles learning, fully compatible with the scikit-learn API. WoodTapper enables seamless integration of interpretable rule extraction into existing machine learning workflows. In addition, it introduces an example-based explanation module that links predictions to a small set of representative samples.


## 🛠 Installation

**From PyPi**:
```bash
pip install woodtapper
```

## 🌳 WoodTapper RulesEXtraction module
```python
## RandomForestClassifier rules extraction
from extract_rules.extractors import SirusClassifier

SIRUS = SirusClassifier(n_estimators=1000,max_depth=2,
                          quantile=10,p0=0.01, random_state=0)
SIRUS.fit(X_train,y_train)
y_pred_sirus = SIRUS.predict(X_test)
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
