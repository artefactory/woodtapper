---
title: 'WoodTapper: a Python package for tapping decision tree ensembles'
tags:
  - Python
  - machine learning
  - XAI
authors:
  - name: Abdoulaye SAKHO
    orcid: 0009-0002-0248-4881
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Jad AOUAD
    orcid: 0009-0000-1456-3056
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Emmanuel MALHERBE
    orcid: 0009-0006-0898-6873
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Erwan SCORNET
    affiliation: 2
affiliations:
 - name: Artefact Research Center, Paris, France
   index: 1
   ror: 00hx57361
 - name: Laboratoire de Probabilités, Statistique et Modélisation Sorbonne Université and Université Paris Cité, CNRS, F-75005, Paris
   index: 2
date: 31 November 2025
bibliography: paper.bib
---

# Summary

Interpretable machine learning has become an increasingly critical concern [@nussberger2022public;@sokol2024interpretable] as predictive models are deployed in high-stakes settings such as healthcare [@Khalilia:2011], finance[@ex-fraud;@sakho2025harnessing] or marketing [@ex-churn]. While complex models (e.g., deep neural networks, ensemble methods) often yield strong predictive performance, their opacity can pose challenges for accountability, trust and compliance. Among interpretable models, rule-based methods are especially attractive because they yield decision paths that humans can follow in the form of “if-then” statements, which are often easier to audit and communicate than latent feature transformations. Stable and Interpretable RUle Set (SIRUS)[@benard2021sirus-classif;@benard2021interpretable-regression] is one such method.

This article presents WoodTapper, a Python toolbox for tree ensemble models fully compatible with the scikit-learn API [@pedregosa2011scikit]. WoodTapper enables seamless integration of interpretable rule extraction into existing machine learning workflows. In addition to faithfully implementing the SIRUS methodology in Python, it introduces an example-based explanation module that links predictions to a small set of representative samples through a weighting scheme, enhancing human interpretability.

# Statement of need

The original SIRUS algorithm offered a principled approach to generate simple and stable rule-based models from random forests. However, its implementations have been limited to R and Julia [@benard2021sirus-classif;@huijzer2023sirus-jl], creating accessibility barriers for the Python data science community. WoodTapper addresses this gap by offering a native Python implementation that integrates with the scikit-learn ecosystem. Furthermore, WoodTapper extends rules extraction $(i)$ from all the tree-based models in scikit learn and $(ii)$ to the multiclass classification setting.

In addition, WoodTapper introduces an example-based explainability methodology that can be applied to all tree-based classifiers. This approach associates predicted samples with representative samples from the training data set, helping users explaining tree-based classifier outputs through concrete examples.



# Rules Extraction

In this section, we present our $\texttt{RulesExtractor}$ module, which is compatible with any ensemble of trees. In the following, we specifically consider its application to a random forest classifier, which corresponds to the SIRUS algorithm introduced by @benard2021sirus-classif.


## SIRUS: formulation
We suppose that we have a training set $\mathcal{D}_{n}=\{(x_i,y_i)\}_{i=1}^{n}$ composed of $n$ pairs of independent and identically distributed (i.i.d) as $(X, Y)$. The random variable $X$ and $Y$  takes values respectively in $\mathbb{R}^d$ and $\{0,1\}$ (binary classification). We denote by $x_i^{(j)}$ the $j$ components of the $i$-th sample in $\mathcal{D}_n$.

In a tree, we denote the path of successive splits from the root node by $\mathcal{P}$  [@benard2021sirus-classif]. A path $\mathcal{P}$ is thus defined as
$$
    \mathcal{P} = \{(j_k,r_k,s_k), k=1, \dots, d\},
$$
where $d$ is the path length, $j_k$ is the selected feature at depth $k$, $r_k$ the selected splitting position along $X^{(j_k)}$ and $s_k$ corresponds to the chosen child node (either $\leq$ corresponding to the left node or $>$ corresponding to the right node).
Thus, each path defines a hyperrectangle in the input space using $\mathcal{D}_n$, denoted $\hat{H}_n(\mathcal{P})$. Hence, each path can be associated with a rule $\hat{g}_{n,\mathcal{P}}(x)$, that returns the mean of $Y$ from its training sample for each of the two different cells. Thus,
$$
    \hat{g}_{n,\mathcal{P}}(x) =
    \begin{cases}
        \frac{1}{|\hat{H}_n(\mathcal{P})|} \sum_{i=1}^{n}y_i \mathbb{I}_{\{x_i \in \hat{H}_n(\mathcal{P})\}} \text{ if } x \in \hat{H}_n(\mathcal{P})\\
        \frac{1}{n-|\hat{H}_n(\mathcal{P})|} \sum_{i=1}^{n}y_i \mathbb{I}_{\{x_i \not\in \hat{H}_n(\mathcal{P})\}} \text{ otherwise }.
    \end{cases}
$$
The probability that a given path $\mathcal{P}$ belongs to a $\Theta$-random tree is
$$
    p_n\left(\mathcal{P}\right) = \mathbb{P}\left(\mathcal{P}\in T(\Theta,\mathcal{D}_n|\mathcal{D}_n)\right).
$$
For a path $\mathcal{P}$, $p_n\left(\mathcal{P}\right)$ is estimated via Monte-Carlo sampling with $\hat{p}_{M,n}$,
$$
    \hat{p}_{M,n}\left(\mathcal{P}\right) = \frac{1}{M} \sum_{l=1}^{M} \mathbb{1}_{\{\mathcal{P} \in T(\Theta_l,\mathcal{D}_n)\}}.
$$

The set of finals rules is $\{\hat{g}_{n,\mathcal{P}}, \mathcal{P} \in  \hat{\mathcal{P}}_{M,n,p_0}\}$ where $\hat{\mathcal{P}}_{M,n,p_0} = \left\{ \mathcal{P} \in \Pi, \, \hat{p}_{M,n}(\mathcal{P}) > p_0\right\}$ with $p_0 \in (0,1)$. The finals rules are aggregated as follows for building the final estimator:
$$
    \hat{\eta}_{M,np_0}(x) = \frac{1}{|\hat{\mathcal{P}}_{M,n,p_0}|} \sum_{\mathcal{P} \in \hat{\mathcal{P}}_{M,n,p_0}} \hat{g}_{n,\mathcal{P}}(x).
$$

So far, we have focused on binary classification for clarity. SIRUS was originally implemented in R for both binary classification and regression, with the regression version differing only in how the final rules are aggregated using weights learned via ridge regression. Our implementation extends SIRUS to multiclass classification (not available in the original R version) as well as regression.

## SIRUS: implementation
WoodTapper was developed to closely follow the algorithmic structure of the original SIRUS, translating its statistical logic into efficient Python code.
The package adheres to the scikit-learn [&pedregosa2011scikit] estimator interface, providing familiar methods such as $fit$, $predict$, and $get\_params$. This design enables smooth integration with existing workflows involving pipelines, cross-validation, and model selection (see Table \ref{tab:comparison}).

: **Comparison of SIRUS implementations across softwares.**\label{tab:comparison}

| **Feature**                 | **WoodTapper (Py)**                        | **SIRUS (R)**                          | **SIRUS (Jl)**           |
|-----------------------------|--------------------------------------------|----------------------------------------|---------------------------|
| Language ecosystem          | Python 3.x                                 | R 4.x                                  | Julia 1.x                 |
| Forest implementation       | `scikit-learn`                             | `ranger`                               | Own                       |
| Package availability        | PyPI (`forest-secrets`)                    | CRAN (`sirus`)                         | General registry          |
| Parallel computation        | $\checkmark$  (via `joblib`)                           | Limited (via `parallel`)               | $\checkmark$  (native)               |
| ML pipelines                | $\checkmark$                                           | Partial                                | Partial                   |
| Tree-based models           | All                                        | random forest                          | random forest             |
| Rules interface             | Unified class methods                      | Function-based                         | Function-based            |
| Tree depth ≥ 3              | $\checkmark$                                           | $\checkmark$                                       | $\times$                          |
| Classification              | Multiclass                                | Binary                                 | Multiclass                |


## SIRUS: rules and predictive performances

We compare the rules produced by the original SIRUS (R) and our Python implementation (WoodTapper) in \ref{fig:sub-titanic-r} and \ref{fig:sub-titanic-py}. On the Titanic dataset, both implementations yield identical rules, confirming that our Python version faithfully reproduces the original algorithm.
| ![\label{fig:sub-titanic-r}](images/rules-titanic-r.pdf){ width=70% } | ![\label{fig:sub-titanic-py}](images/rules-titanic-py.pdf){ width=70% } |
|:--:|:--:|
| (a) SIRUS (R) | (b) Python (Ours) |


We also observe that the predictive performance of our implementation is similar to that of the original algorithm (see Table \ref{tab:perf_metrics}).
: **Performance metrics for Titanic and House Sales datasets.**\label{tab:perf_metrics}

| **Dataset**   | **Metric** | **SIRUS (original R)** | **Ours**        |
|----------------|------------|------------------------|-----------------|
| **Titanic**    | Accuracy   | 0.79 ± 0.03            | 0.78 ± 0.02     |
|                | (ROC) AUC  | 0.84 ± 0.04            | 0.84 ± 0.04     |
| **House Sales**| MSE        | 0.35 ± 0.02            | 0.34 ± 0.01     |
|                | MAE        | 0.26 ± 0.01            | 0.26 ± 0.01     |


## SIRUS: running time
We compare the runtimes of SIRUS in Python (ours), R, and Julia using 5 threads on an AMD Ryzen Threadripper PRO 5955WX (16 cores, 4GHz) with 250GB RAM, tested on the same dataset generated via scikit-learn’s $\texttt{make\_classification}$.

SIRUS.jl exhibits higher runtime compared to Python and R implementations. The R version, relying on ranger, is faster for tree construction on large datasets than scikit-learn. Our Python implementation, however, is considerably more efficient for rule extraction, independent of sample size or feature dimensionality (see Figures \ref{fig:run-time-samples} and \ref{fig:run-time-dim}).

![Running time for simulated data using 5 threads, with d=200 and M=1000.\label{fig:run-time-samples}](images/run-time-samples-log-5threads-final.pdf){ width=70% }

![Running time for simulated data using 5 threads, with $n$=300K and $M$=1000.\label{fig:run-time-dim}](images/run-time-samples-log-5threads-final.pdf){ width=70% }

# Example-based explainability
The $\texttt{ExampleExplanation}$ module of WoodTapper is independent of rule extraction and provides a measure of example-based explainability. For a new sample $x$ with unknown label, let $\mathcal{L}_l(x)$ denote the set of training samples that share the same leaf as $x$ in tree $T_l$, $l = 1, \dots, M$.

The $\texttt{ExampleExplanation}$ module enables tree-based models to identify the $p \in \mathbb{N}$ training samples most similar to $x$, using the similarity measure induced by random forests [@breiman2001random;@grf]. Specifically, the similarity between $x$ and a training sample is defined as the proportion of trees in which the sample and $x$ fall into the same leaf. Letting $ w_{x}(X_i)$ the similarity between $x$ and $x_i$, we have
$$
w_{x}(x_i) = \frac{1}{M} \sum_{l=1}^{M} \frac{\mathbb{1}_{\{X_i \in \mathcal{L}_l(Z)\}}}{|\mathcal{L}_l(Z)|}.
$$


Let $\mathcal{W}_{x} = \left\{ w_{x}(x_i), x_i \in \mathcal{D}_n\right\}$ be the set of similarities between $x$ and each training sample. Finally the $p$ training samples with the  highest values in $\mathcal{W}_{x}$ are proposed as the examples that explain the most the prediction of $x$ by the tree-based ensemble model.

The $\textit{skgrf}$ [@skgrf] package is an interface for using the R implementation of generalized random forest in Python. $\textit{skgrf}$ has a specififc number of classifier for specfific learning task (causal inference, quantile regression,...). For each task, the user can compute the kernel weights, which are equivalent to our leaf frequency match introduce above. Thus, we aim at comparing the kernenl weights deribvation from $\textit{skgrf}$ to our $\texttt{ExampleExplanation}$ module. We stress on the fact that our $\texttt{ExampleExplanation}$ is designed for usual tree-based models such as random forest of extra trees and not specifically in a context of causal inference or quantile regression. Thus, the tree building (splitting criterion) of our forest are different from the ones from $\textit{skgrf}$.

## ExampleExplanation: implementation
As for SIRUS, our Python implementation of $\texttt{ExampleExplanation}$ adheres to the scikit-learn interface. To be more precise, we built this package on top of scikit-learn.


Our $\texttt{ExampleExplanation}$ module is implemented as a Python Mixin for handling example-based explanations. It is agnostic to the underlying tree ensemble, and can be used with random forests or extra trees (\ref{tab:comparison-grf}). For each ensemble type, a subclass inherits both the original scikit-learn class and the Mixin. The standard $\texttt{fit}$ and $\texttt{predict}$ methods remain unchanged, while an additional $\texttt{explain}$ method provides example-based explanations for new samples. This allows users to train and predict using standard scikit-learn workflows, while enabling access to $\texttt{ExampleExplanation}$ for interpretability analyses.

: **Comparison of GRF weight computations in several Python packages.**\label{tab:comparison-grf}

| **Feature**              | **WoodTapper (Py)**             | **skgrf (Py)**            |
|---------------------------|----------------------------------|----------------------------|
| Forest implementation     | `scikit-learn`                  | `ranger`                   |
| Language ecosystem        | Python                          | Python & R                 |
| Package availability      | PyPI (`forest-secrets`)         | PyPI (`skgrf`)             |
| scikit-learn API compatible | $\checkmark$                            | $\checkmark$                         |
| Tree-based models         | All                             | Tree and random forest     |
| GRF                       | $\times$                               | $\checkmark$                        |


## ExampleExplanation: running time
We compare the runtime of $\texttt{ExampleExplanation}$ with the kernel weight computation in $\textit{skgrf}$ [@skgrf] using the same hardware as in the SIRUS experiments. $\texttt{ExampleExplanation}$ is consistently faster (see figure \ref{fig:run-time-grf}).
![Running time for simulated data using weights extraction.\label{fig:run-time-grf}](images/run-time-grf-dim-log.pdf){ width=50% }

# Conclusions
WoodTapper is a Python package that extracts interpretability and explainability insights from tree ensembles. SIRUS, a more parsimonious alternative to random forests, naturally supports interpretability, and a Python implementation can boost its adoption. Example-based explainability is provided by the $\texttt{ExampleExplanation}$ module, offering precise insights at the individual sample level.

# Acknowledgements

We would like to express our gratitude to Alexandre CHAUSSARD and Vincent AURIAU for their valuable help and feedbacks.

# References
