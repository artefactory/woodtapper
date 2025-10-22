<div align="center">

# Stable and Interpretable RUle Set in Python (pySIRUS)


Abdoulaye SAKHO<sup>1, 2</sup>, Jad AOUAD<sup>1</sup>, Emmanuel MALHERBE<sup>1</sup></sup> and Erwan SCORNET<sup>2</sup> <br>
 <sup>1</sup> <sub> Artefact Research Center, </sub> <br> <sup>2</sup> <sub>*LPSM* - Sorbonne Université</sub>

Preprint. <br>
[[Full Paper]]() <br>

</div>

<p align="center"><img width="65%" src="data/logos/image.png"  /></p>

**Abstract:** *Repository for the devloppement of scikit-learn based methodology for stable rules extraction from a tree-based classifier.*


## 🛠 Installation
First you can clone the repository:
```bash
git clone git@github.com:artefactory/mgs-grf.git
```

And install the required packages into your environment (conda, mamba or pip):
```bash
pip install -r requirements.txt
```

> ⚠️ **Warning** Then, you **need** to run the following command from the repository root directory :
```
python setup.py build_ext --inplace
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
## 💾 Data sets
* [Pima](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* Phoneme : https://github.com/jbrownlee/Datasets/blob/master/phoneme.csv
* Adult : https://archive.ics.uci.edu/dataset/2/adult
* Houses sales : https://www.openml.org/d/44144

* California : https://www.openml.org/d/44090
* Titanic : https://www.kaggle.com/datasets/yasserh/titanic-dataset?resource=download
* Wine : https://archive.ics.uci.edu/dataset/186/wine+quality
* Haberman : https://archive.ics.uci.edu/dataset/43/haberman+s+survival
* Yeast : https://archive.ics.uci.edu/dataset/110/yeast

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
