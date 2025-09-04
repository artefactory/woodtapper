<div align="center">

# Stable and Interpretable RUle Set in Python (pySIRUS)


Abdoulaye SAKHO<sup>1, 2</sup>, Emmanuel MALHERBE<sup>1</sup></sup>, Erwan SCORNET<sup>2</sup> <br>
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

> ⚠️ **Warning** Then, you **need** to run the following command inside models folder : 
```
python setup.py build_ext --inplace
```
> Finally, you have to move the generated file file *models/build/_QuantileSplitter.cpython-311-x86_64-linux-gnu.so* into *models/_QuantileSplitter.cpython-311-x86_64-linux-gnu.so*

## 🚀 How to use pySIRUS

## 💾 Data sets

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


## ⭐ Citation

If you find the code usefull, please consider citing us :