# Tutorials

 A few tutorials to get started with the Woodtapper package are available here:

  - [Rules extractor](1_tutorials.md)
  - [Example-based expalinability](2_tutorials_example_exp.md)

Let's describe some mathematical background that will be used in the different modules:

We suppose that we have a training set $\mathcal{D}_{n}=\{(x_i,y_i)\}_{i=1}^{n}$ composed of $n$ pairs that takes values respectively in $\mathbb{R}^p$ and $\{0,1\}$ (binary classification). We denote by $x_i^{(j)}$ the $j$ component of a $i$-th sample in $\mathcal{D}_n$.

## Rules extraction:


In this section, we present our $\texttt{RulesExtraction}$ module, which is compatible with any ensemble of trees. In the following, we specifically consider its application to a random forest classifier, which corresponds to the SIRUS algorithm.


In a tree $\mathcal{T}$, we denote the path of successive splits from the root node by $\mathcal{P}$, defined as
$$\mathcal{P} = \{(j_k,r_k,s_k), k=1, \dots, K\},$$
where $K$ is the path length, $j_k \in \{1, \dots,p\}$ is the selected feature at depth $k$, $r_k \in \mathbb{R}$ the selected splitting position along $x^{(j_k)}$ and $s_k$ the corresponding sign (either $\leq$ corresponding to the left node or $>$ corresponding to the right node).
Thus, each path defines a hyperrectangle in the input space, denoted $\hat{H}(\mathcal{P}) \subset \mathbb{R}^p$. Hence, each path can be associated with a rule function $\hat{g}_{\mathcal{P}}$, that returns the mean of $Y$ from the training sample inside and outside of $\hat{H}(\mathcal{P})$:

$$ \hat{g}_{\mathcal{P}}(x) =\begin{cases} \frac{\sum_{i=1}^{n}y_i \mathbb{I}_{\{x_i \in \hat{H}(\mathcal{P})\}}}{\sum_{i=1}^{n} \mathbb{I}_{\{x_i \in \hat{H}(\mathcal{P})\}}}  \text{ if } x \in \hat{H}(\mathcal{P})\\ \frac{\sum_{i=1}^{n}y_i \mathbb{I}_{\{x_i \not\in \hat{H}(\mathcal{P})\}}}{\sum_{i=1}^{n} \mathbb{I}_{\{x_i \not\in \hat{H}(\mathcal{P})\}}}  \text{ otherwise }.\end{cases}$$

We suppose we have a set of trees $\{\mathcal{T}_m, m=1, \dots, M \}$ from a tree ensemble procedure, each grown with randomness $\Theta_m$. We denote by $\Pi$ the set of all possibles paths from $\{\mathcal{T}_m, m=1, \dots, M \}$. For a path $\mathcal{P} \in \Pi$, we estimate the rule probability $p\left(\mathcal{P}\right)$ via Monte-Carlo sampling with $\hat{p}\left(\mathcal{P}\right)$:

$$\hat{p}\left(\mathcal{P}\right) = \frac{1}{M} \sum_{m=1}^{M} \mathbb{1}_{\{\mathcal{P} \in \mathcal{T}(\Theta_m,\mathcal{D}_n)\}},$$

which corresponds to the empirical probability that the path $\mathcal{P} \in \Pi$ belongs to the set of trees $\{\mathcal{T}_m, m=1, \dots, M \}$.

The set of final rules is $\{\hat{g}_{\mathcal{P}}, \mathcal{P} \in  \hat{\mathcal{P}}_{p_0}\}$ where $\hat{\mathcal{P}}_{p_0} = \left\{ \mathcal{P} \in \Pi, \, \hat{p}(\mathcal{P}) > p_0\right\}$ with $p_0 \in [0,1)$. The finals rules are aggregated as follows for building the final estimator:

$$\hat{\eta}_{p_0}(x) = \frac{1}{|\hat{\mathcal{P}}_{p_0}|} \sum_{\mathcal{P} \in \hat{\mathcal{P}}_{p_0}} \hat{g}_{\mathcal{P}}(x).$$

So far, we have focused on binary classification for clarity.
We also implemented the rule extractor for regression, where final rules are aggregated using weights learned via ridge regression. Our implementation extends SIRUS, i.e. rules extracted from random forest, to multiclass classification (not available in the original R version). Finally, our implementation also leverages scikit-learn's implementations for tree-based models fitting.

## Example-based explainability:

The $\texttt{ExampleExplanation}$ module of WoodTapper is independent of the rule extraction module and provides an example-based explainability.
It enables tree-based models to identify the $l \in \mathbb{N}$ most similar training samples to $x$, using the similarity measure induced by generalized random forests.
For a new sample $x$ with unknown label and $\mathcal{T}_m$ a decision tree, let $\mathcal{L}_m(x)$ denote the set of training samples that share the same leaf as $x$ in tree $\mathcal{T}_m$ for $m = 1, \dots, M$.
Letting $w(x,x_i)$ be the similarity between $x$ and $x_i$, we have

$$w(x,x_i) = \frac{1}{M} \sum_{m=1}^{M} \frac{\mathbb{1}_{\{x_i \in \mathcal{L}_m(x)\}}}{|\mathcal{L}_m(x)|}.$$

Finally, the $l$ training samples with the highest $w(x,x_i)$ values, along with their target values, are proposed as the examples that best explain the prediction of $x$ by the tree-based ensemble model.
