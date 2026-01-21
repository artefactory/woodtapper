---
title: '`WoodTapper`: a Python package for explaining decision tree ensembles'
tags:
  - Python
  - Machine Learning
  - XAI
authors:
  - name: Abdoulaye SAKHO
    orcid: 0009-0002-0248-4881
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Jad AOUAD
    orcid: 0009-0000-1456-3056
    affiliation: 1
  - name: Carl-Erik GAUTHIER
    orcid: 0009-0009-4256-5425
    affiliation: 3
  - name: Emmanuel MALHERBE
    orcid: 0009-0006-0898-6873
    affiliation: 1
  - name: Erwan SCORNET
    orcid: 0000-0001-9946-4160
    affiliation: 2
affiliations:
 - name: Artefact Research Center, Paris, France
   index: 1
 - name: Laboratoire de Probabilités, Statistique et Modélisation Sorbonne Université and Université Paris Cité, CNRS, F-75005, Paris
   index: 2
 - name: Société Générale, Paris, France
   index: 3
date: 14 November 2025
bibliography: paper.bib
---

# Introduction

Interpretable machine learning has become an increasingly critical concern [@nussberger2022public;@sokol2024interpretable] as predictive models are deployed in high-stakes settings such as healthcare [@Khalilia:2011], marketing [@ex-churn] or finance [@ex-fraud;@sakho2025harnessing] which is moreover a regulated sector. While complex models, such as tree-based ensemble methods, often yield strong predictive performance, their opacity can pose challenges for accountability, trust and compliance. Among interpretable models, rule-based methods are especially attractive because they are explained in the form of “if-then” statements, which are often easier to audit and communicate than latent feature transformations.


# Statement of need

The original SIRUS algorithm [@benard2021sirus-classif;@benard2021interpretable-regression] offered a principled approach to generate simple and stable rule-based models from random forests. However, its implementations have been limited to R and Julia [@benard2021sirus-classif;@huijzer2023sirus-jl], creating accessibility barriers for the Python data science community. `WoodTapper` addresses this gap by offering a native Python implementation that integrates with the scikit-learn ecosystem. Furthermore, `WoodTapper` extends rules extraction $(i)$ from all the tree-based models in scikit learn (Random Forest, Gradient Boosting and Extremely Randomized Trees) and $(ii)$ to the multiclass classification setting.

In addition, `WoodTapper` introduces an example-based explainability methodology that can be applied to all scikit-learn tree-based models. This approach associates predicted samples with representative samples from the training data set, explaining tree-based models predictions through examples.

# Software design
`WoodTapper` package adheres to the scikit-learn [@pedregosa2011scikit] estimator interface. 
This design enables smooth integration with existing workflows involving pipelines, cross-validation, and model selection, and enables to efficiently benefit from future maintenance updates and improvements to scikit-learn.
The implementation leverages NumPy for numerical computation and joblib for parallel processing to optimize performance on large datasets (\ref{tab:comparison}).
The code architecture uses a Mixin inherited by all tree-based models to improve code reuse and factorization. For each tree-based ensemble type, a subclass inherits both the original scikit-learn class and the Mixin. The standard $\texttt{fit}$ and $\texttt{predict}$ methods remain unchanged, while additional methods of `WoodTapper` are available.  
We compared our Python implementation with the Julia, R and skgrf versions (see Table \ref{tab:comparison} and \ref{tab:comparison-grf}) and observed that WoodTapper provides broader options for tree-based model extraction, faster rule-extraction runtimes, and support for multiclass classification with unlimited tree depth.


# Research impact statement
Being a Python package, WoodTapper enables practitioners to perform rule extraction using SIRUS[^1], and easily integrate these rules into their projects.
Furthermore, WoodTapper provides an example-based auditing tool for black-box, tree-based models already deployed in production. It has been already applied successfully in the context of Artefact's consulting missions with clients of different sectors, including banking[^2].

`WoodTapper` has demonstrated significant research impact and has grown both its user base and contributor community since its initial release[^3]. The package has evolved through contributions from multiple developers, with community members able to adding new features, reporting and fixing bugs, and proposing enhancements. Furthermore, our fully reproducible benchmarks described in the following show concrete improvements in terms of model accuracy and computation time. 

[^1]: Cited roughly 200 times at end 2025.
[^2]: The details of these deployment remain confidential and are beyond the scope of this paper.
[^3]: More than 1,000 downloads were counted on pypi in the 2 first months.

# Rules Extraction Module

## Formulation
In this section, we present our $\texttt{RulesExtraction}$ module and we specifically consider its application to a random forest classifier, which corresponds to the SIRUS algorithm introduced by @benard2021sirus-classif.

We suppose that we have a training set $\mathcal{D}_{n}=\{(x_i,y_i)\}_{i=1}^{n}$ composed of $n$ pairs taking values in $\mathbb{R}^p$ and $\{0,1\}$ respectively (binary classification). We denote by $x_i^{(j)}$ the $j$-th component of the $i$-th sample in $\mathcal{D}_n$. We suppose we have a set of trees $\{\mathcal{T}_m, m=1, \dots, M \}$ from a tree ensemble procedure, each grown with randomness $\Theta_m$.

In a tree $\mathcal{T}_m$, we denote by $\mathcal{P}$ a path of successive splits from the root node. $\mathcal{P}$ defines thus a hyperrectangle in the input space, $\hat{H}(\mathcal{P}) \subset \mathbb{R}^p$. We associate $\mathcal{P}$ with a rule function $\hat{g}_{\mathcal{P}}$ returning the mean of $Y$ from the training sample inside and outside $\hat{H}(\mathcal{P})$.

For a set of trees $\{\mathcal{T}_m, m=1, \dots, M \}$ and a path $\mathcal{P}$, we define:
$$
    \hat{p}\left(\mathcal{P}\right) = \frac{1}{M} \sum_{m=1}^{M} \mathbb{1}_{\{\mathcal{P} \in \mathcal{T}(\Theta_m,\mathcal{D}_n)\}},
$$
which corresponds to the empirical probability that the path $\mathcal{P}$ belongs to the set of trees $\{\mathcal{T}_m, m=1, \dots, M \}$.
The set of final rules is $\{\hat{g}_{\mathcal{P}}, \mathcal{P} \in  \hat{\mathcal{P}}_{p_0}\}$ where $\hat{\mathcal{P}}_{p_0} = \left\{ \mathcal{P}, \, \hat{p}(\mathcal{P}) > p_0\right\}$ with $p_0 \in [0,1)$. The final rules are aggregated as follows for building the final estimator:
$$
    \hat{\eta}_{p_0}(x) = \frac{1}{|\hat{\mathcal{P}}_{p_0}|} \sum_{\mathcal{P} \in \hat{\mathcal{P}}_{p_0}} \hat{g}_{\mathcal{P}}(x).
$$

Beyond the binary classification detailed here, we also implemented the rule extractor for regression, where final rules are aggregated using weights learned via ridge regression.

## Running time

: **Comparison of SIRUS implementations across softwares.**\label{tab:comparison}

| **Feature**                 | **`WoodTapper` (Py)**                        | **SIRUS (R)**                          | **SIRUS (Jl)**           |
|-----------------------------|--------------------------------------------|----------------------------------------|---------------------------|
| Language          | Python 3.x                                 | R 4.x                                  | Julia 1.x                 |
| Forest        | `scikit-learn`                             | `ranger`                               | Own                       |
| Availability        | PyPI (`woodtapper`)                    | CRAN (`sirus`)                         | General registry          |
| Parallelism        | $\checkmark$  (via `joblib`)                           | Limited (via `parallel`)               | $\checkmark$  (native)               |
| ML pipelines                | $\checkmark$                                           | Partial                                | Partial                   |
| Tree models           | All                                        | random forest                          | random forest             |
| Rules interface             | Unified class methods                      | Function-based                         | Function-based            |
| Tree depth $≥ 3$            | $\checkmark$                                           | $\checkmark$                                       | $\times$                          |
| Classification              | Multiclass                                | Binary                                 | Multiclass                |

We compare the runtimes of SIRUS in Python (ours), R, and Julia using 5 threads on an AMD Ryzen Threadripper PRO 5955WX (16 cores, 4GHz) with 250GB RAM. We also experimented on large-scale industrial data sets, including from the banking sector, and observed the same trends as displayed here. SIRUS.jl exhibits higher runtime compared to Python and R implementations. The R version, relying on ranger, is faster for tree construction on large datasets than scikit-learn. Our Python implementation, however, is considerably more efficient for rule extraction, regardless of sample size or feature dimensionality (see Figures \ref{fig:run-time-samples} and \ref{fig:run-time-dim}).

![SIRUS running time for simulated data using 5 threads, with d=200 and M=1000.\label{fig:run-time-samples}](images/run-time-samples-log-5threads-final.pdf){ width=100% }

![SIRUS running time for simulated data using 5 threads, with $n$=300K and $M$=1000.\label{fig:run-time-dim}](images/run-time-dim-log-5threads-final.pdf){ width=100% }


## Extracted rules and predictive performances

The rules produced by the original SIRUS (R) and our $\texttt{RulesExtraction}$ in \ref{fig:sub-titanic-r} and \ref{fig:sub-titanic-py} on the Titanic data set are identical, and predictive performances in Table \ref{tab:perf_metrics} are very similar, confirming that our implementation faithfully reproduces the original algorithm.

![SIRUS (R) rules on Titanic data set. \label{fig:sub-titanic-r}](images/rules-titanic-r-final.pdf){ width=70% }

![WoodTapper SIRUS (Ours) rules on Titanic data set. \label{fig:sub-titanic-py}](images/rules-titanic-py-final.pdf){ width=70% }


: **Performance metrics for Titanic and House Sales datasets.**\label{tab:perf_metrics}

| **Dataset**   | **Metric** | **SIRUS (original R)** | **Ours**        |
|----------------|------------|------------------------|-----------------|
| **Titanic**    | Accuracy   | 0.79 ± 0.03            | 0.78 ± 0.02     |
|                | (ROC) AUC  | 0.84 ± 0.04            | 0.84 ± 0.04     |
| **House Sales**| MSE        | 0.35 ± 0.02            | 0.34 ± 0.01     |
|                | MAE        | 0.26 ± 0.01            | 0.26 ± 0.01     |


# Example-based explainability module

## Formulation

The $\texttt{ExampleExplanation}$ module of `WoodTapper` is independent of the $\texttt{RulesExtraction}$ module and provides an example-based explainability.
It enables tree-based models to identify the most similar training samples to $x$, using the similarity measure induced by generalized random forests [@breiman2001random;@grf].
For a new sample $x$ with unknown label and a decision tree $\mathcal{T}_m$, let $\mathcal{L}_m(x)$ denote the set of training samples that share the same leaf as $x$.
We define the similarity $w(x,x_i)$ between $x$ and $x_i$ as:
$$
w(x,x_i) = \frac{1}{M} \sum_{m=1}^{M} \frac{\mathbb{1}_{\{x_i \in \mathcal{L}_m(x)\}}}{|\mathcal{L}_m(x)|}.
$$

Finally, the $l$ training samples with the highest $w(x,x_i)$ values, along with their target values $y_i$, are shown as the examples that best explain the prediction of $x$ by the tree-based ensemble model.

In python, the $\textit{skgrf}$ [@skgrf] package is an interface for using the R implementation of generalized random forest, focusing on classifiers for specifics learning tasks (causal inference, quantile regression,...). For each task, the user can compute the kernel weights, equivalently to our leaf frequency match introduce above. Thus, we compare the kernel weights computation by $\textit{skgrf}$ and our module. We stress on the fact that our $\texttt{ExampleExplanation}$ is designed for usual tree-based models such as random forest of extra trees and not specifically in a context of causal inference or quantile regression. In particular, the tree building of our forest is different from the one in $\textit{skgrf}$.

## Running time

: **Comparison of GRF weight computations in several Python packages.**\label{tab:comparison-grf}

| **Feature**              | **`WoodTapper` (Py)**             | **skgrf (Py)**            |
|---------------------------|----------------------------------|----------------------------|
| Forest implementation     | `scikit-learn`                  | `ranger`                   |
| Language                  | Python                          | Python & R                 |
| Package availability      | PyPI (`woodtapper`)             | PyPI (`skgrf`)             |
| scikit-learn API compatible | $\checkmark$                            | $\checkmark$                         |
| Tree-based models         | All                             | Tree and random forest     |
| GRF                       | $\times$                               | $\checkmark$                        |


In figure \ref{fig:run-time-grf}, we compare the kernel weight computation runtime of $\texttt{ExampleExplanation}$ and $\textit{skgrf}$ [@skgrf] using the same hardware as previous experiments. $\texttt{ExampleExplanation}$ is consistently faster.

![Weights computation running time for simulated data using.\label{fig:run-time-grf}](images/run-time-grf-dim-log.pdf){ width=100% }

# AI usage disclosure

Generative AI tools were used in this software only to implement the Cython function in the $\texttt{ExampleExplanation}$ module and to draft certain docstring elements. For writing this manuscript and preparing supporting materials, generative AI was employed solely for formatting.


# Acknowledgements

We would like to express our gratitude to Clément BENARD, Alexandre CHAUSSARD and Vincent AURIAU for their valuable help and feedbacks.

# References
