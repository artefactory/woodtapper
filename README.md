<div align="center">

# WoodTapper

<p align="center"><img width="25%" src="data/logos/logo_woodpecker_compressed.png"  /></p>

*User-friendly and scalable Python package for tapping decision tree ensembles*

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
</div>

WoodTapper is a Python toolbox  for interpretable and explainable tree ensembles learning, fully compatible with the scikit-learn API. WoodTapper enables seamless integration of interpretable rule extraction into existing machine learning workflows. In addition, it introduces an example-based explanation module that links predictions to a small set of representative samples.


## 🛠 Installation
First you can clone the repository:
```bash
git clone git@github.com:artefactory/mgs-grf.git
```

And install the required packages into your environment (conda, mamba or pip):
```bash
pip install -r requirements.txt
```

Then, you **need** to run the following command from the repository root directory :
```
pip install -e .[dev]
```

## 🚀 How to use pySIRUS
```python
## RandomForestClassifier rules extraction
from extract_rules.extractors import SirusRFClassifier

SIRUS = SirusRFClassifier(n_estimators=1000,max_depth=2,
                          quantile=10,p0=0.01, random_state=0)
SIRUS.fit(X_train,y_train)
y_pred_sirus = SIRUS.predict(X_test)
```

## 🙏 Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="data/logos/logo_arc.png" height="80" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.lpsm.paris/">
    <img src="data/logos//logo_LPSM.jpg" height="95" />
  </a>
</p>


## 📜 Citation

If you find the code usefull, please consider citing us :
