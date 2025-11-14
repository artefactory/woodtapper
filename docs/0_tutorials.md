# Tutorials

 A few tutorials to get started with the Woodtapper package are available here:

  - [Rules extractor](1_tutorials.md)
  - [Example-based expalinability](2_tutorials_example_exp.md)

Let's describe some mathematical background that will be used in the different modules:

We define the training set $\mathcal{D}_{n}=\{(x_i,y_i)\}_{i=1}^{n}$ composed of $n$ pairs of independent and identically distributed (i.i.d) as $(X, Y)$. The random variables $X$ and $Y$ take values respectively in $\mathbb{R}^d$ and $\{0,1\}$ (binary classification). We denote by $x_i^{(j)}$ the $j$ components of the $i$-th sample in $\mathcal{D}_n$.

## Rules extraction:


In a tree, we denote the path of successive splits from the root node by $\mathcal{P}$. A path $\mathcal{P}$ is thus defined as:
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

$$p_n\left(\mathcal{P}\right) = \mathbb{P}\left(\mathcal{P}\in T(\Theta,\mathcal{D}_n|\mathcal{D}_n)\right).$$

For a path $\mathcal{P}$, $p_n\left(\mathcal{P}\right)$ is estimated via Monte-Carlo sampling with $\hat{p}_{M,n}$,

$$\hat{p}_{M,n}\left(\mathcal{P}\right) = \frac{1}{M} \sum_{l=1}^{M} \mathbb{1}_{\{\mathcal{P} \in T(\Theta_l,\mathcal{D}_n)\}}.$$

All in all, the extraction algorithm follows the following steps:


1. Train a random forest with $M$ trees where splits can only be performed on the empirical $q$-quantiles (computed on the whole data set) of each variable.
2. Extract all paths $\mathcal{P}$ from the random forest. Let $\Pi$ be the set of these  paths.
3. Let $p_0 \in (0,1)$ be an hyperparameter of the extraction procedure. Only the paths that have a frequency superior to $p_0$ are kept. Denote the set of such paths by $\hat{\mathcal{P}}_{M,n,p_0} = \left\{ \mathcal{P} \in \Pi, \, \hat{p}_{M,n}(\mathcal{P}) > p_0\right\}$. Then all paths that are linearly dependent on paths with higher $\hat{p}_{M,n}$ are removed from $\hat{\mathcal{P}}_{M,n,p_0}$.


The set of finals rules is $\{\hat{g}_{n,\mathcal{P}}, \mathcal{P} \in  \hat{\mathcal{P}}_{M,n,p_0}\}$ is aggregated as follows for building the final estimator:

$$
    \hat{\eta}_{M,np_0}(x) = \frac{1}{|\hat{\mathcal{P}}_{M,n,p_0}|} \sum_{\mathcal{P} \in \hat{\mathcal{P}}_{M,n,p_0}} \hat{g}_{n,\mathcal{P}}(x).
$$

So far, we have focused on binary classification for clarity. The extraction procedure was originally implemented in R for both binary classification and regression, with the regression version differing only in how the final rules are aggregated using weights learned via ridge regression. Our implementation extends the procedure to multiclass classification (not available in the original R version) as well as regression.

## Example-based explainability:

The $\texttt{ExampleExplanation}$ module of WoodTapper is independent of rule extraction and provides a measure of example-based explainability. For a new sample $x$ with unknown label, let $\mathcal{L}_l(x)$ denote the set of training samples that share the same leaf as $x$ in tree $T_l$, $l = 1, \dots, M$.

The $\texttt{ExampleExplanation}$ module enables tree-based models to identify the $p \in \mathbb{N}$ training samples most similar to $x$, using the similarity measure induced by random forests. Specifically, the similarity between $x$ and a training sample is defined as the proportion of trees in which the sample and $x$ fall into the same leaf. Letting $w_{x}(X_i)$ the similarity between $x$ and $x_i$, we have
$$
w_{x}(x_i) = \frac{1}{M} \sum_{l=1}^{M} \frac{\mathbb{1}_{\{X_i \in \mathcal{L}_l(Z)\}}}{|\mathcal{L}_l(Z)|}.
$$


Let $\mathcal{W}_{x} = \left\{ w_{x}(x_i), x_i \in \mathcal{D}_n\right\}$ be the set of similarities between $x$ and each training sample. Finally the $p$ training samples with the  highest values in $\mathcal{W}_{x}$ are proposed as the examples that explain the most the prediction of $x$ by the tree-based ensemble model.
