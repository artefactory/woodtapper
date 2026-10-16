# Tutorials

 A few tutorials to get started with the Woodtapper package are available here:

  - [Rules extractor](1_tutorials.md)
  - [Fairness-aware classification](3_tutorials_fairness.md)
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

## Fairness:

We build upon the functional gradient boosting paradigm exemplified by frameworks like XGBoost  and LightGBM . The model learns to estimate $\hat y$ from sum of weak learners $F_t=\sum_{k=1}^t f_k(x)$.
%
In the case of classification, this continuous output is converted to a probability using a sigmoid function $\sigma(F_t(x))=\frac{1}{1+e^{-F_t(x)}}$, and represents the estimated $P(\hat y=1)$.
To iteratively learn $F_t$, at each boosting iteration $t$, a new decision tree $f_t$ is added to the model ensemble in order to improve the predictions on the dataset $\mathcal{D}$, as quantified by a samplewise loss $\ell\big(y_i, F_t(x_i)\big)$ (e.g., binary cross-entropy or squared error).
The tree $f_t$ aims at minimizing a second-order Taylor expansion of the objective function around the previous iteration's predictions $F_{t-1}(x_i)$:
\begin{align}
&\mathcal{L}^{(t)}  = \sum_{i=1}^n \ell\!\left(y_i, F_{t-1}(x_i) + f_t(x_i)\right) + \Omega(f_t)
\end{align}
\begin{align}
&\approx \sum_{i=1}^n \left[ \ell\!\left(y_i, F_{t-1}(x_i)\right) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
& = \tilde{\mathcal{L}}^{(t)},
\end{align}
where $\ell(\cdot, \cdot)$ is twice differentiable, with first-order gradients $g_i$ and second-order Hessians $h_i$ defined samplewise as:
\begin{align} g_i = \frac{\partial \ell\left(y_i, F_{t}(x_i)\right)}{\partial {F_{t}(x_i)}} \left(F_{t-1}(x_i)\right),
\end{align}

\begin{align} h_i = \frac{\partial^2 \ell\left(y_i, {F}_{t}(x_i)\right)}{\partial {\left({F}_{t}(x_i)\right)^2}} \left({F}_{t-1}(x_i)\right). \end{align}

and the term $\Omega(f_t) = \rho T + \frac{1}{2}\eta \sum_{j=1}^T w_j^2$ regularizes the structural complexity of the tree $f_t$.

The practical algorithm to minimize this $\tilde{\mathcal{L}}^{(t)}$ and the corresponding leaf split criterion is detailed in \cite{chen2016xgboost}. Besides, please note that in the classification setting, $\ell$ is implicitly composed with $\sigma$ to obtain probabilities in $[0,1]$ from $F_t(x_i)$. In this case, the derivative includes a factor $\sigma'(F_t(x))$ coming from the chain rule.


To enforce algorithmic fairness during this iterative optimization, we must formalize the statistical dependence between the model's predictions and the sensitive attribute.


To operationalize DP and EOdds over continuous prediction scales, let us focus on the Demographic Parity. To ease notations, we will write $\hat{z} \in \mathbb{R}$ the continuous prediction of the model, in the case of classification and regression:
\begin{align}
\hat z^{(t)} =
\begin{cases}
   F_t(x) & \text{if } y \in \mathbb{R} \\
  \sigma\left(F_t(x)\right)   & \text{if } y \in \{0,1\}
\end{cases}
\end{align}
%
We group $\hat z_i^{(t)}$ values by sensitive attribute $s_i$. We define the empirical set of prediction scores at iteration $t$ for group $s$ as
$$\mathcal{Z}^s = \{\hat{z}_i^{(t)} \in \mathbb{R} \mid s_i = s\}=\{{\hat{z}_j}^{s^{(t)}} \in \mathbb{R} \mid j = 1, \dots, n_s\}$$
where $n^s = |\{i : s_i = s\}|$ denote the size of sensitive group $s \in \{0, 1\}$, and each group is re-indexed. In the following, we omit to write $^{(t)}$ for clarity. The empirical probability measure $\mu^{\mathcal{Z}^s}$ for the demographic cohort $\mathcal{Z}^s$ is expressed as:
\begin{equation}
\mu^{\mathcal{Z}^s} = \frac{1}{n_s} \sum_{j=1}^{n^s} \delta_{\hat{z}^s_j},
\end{equation}
where $\delta_z$ denotes the Dirac delta measure centered at $z$.

We propose an in-processing regularization framework that directly incorporates a statistical distance metric $\mathcal{M}$ between these group-conditional empirical measures $\mu^{\mathcal{Z}^0}$ and $\mu^{\mathcal{Z}^1}$ into the boosting objective at iteration $t$:
\begin{equation}
\mathcal{L}^{(t)}_{\text{fair}} = \mathcal{L}^{(t)} + \lambda \mathcal{M}\left(\mu^{\mathcal{Z}^0}, \mu^{\mathcal{Z}^1}\right),
\end{equation}
where $\lambda \in \mathbb{R}^+$ is a user-defined hyperparameter controlling the accuracy-fairness trade-off frontier. The functional structure of $\mathcal{M}$ can be adapted to accommodate diverse fairness definitions (e.g., conditioning on ground truth $y$ to satisfy EOdds). In this work, we consider the discrete Wasserstein distance, to follow the nature of our distributions $\mu^{\mathcal{Z}^s}$.

## Example-based explainability:

The $\texttt{ExampleExplanation}$ module of WoodTapper is independent of the rule extraction module and provides an example-based explainability.
It enables tree-based models to identify the $l \in \mathbb{N}$ most similar training samples to $x$, using the similarity measure induced by generalized random forests.
For a new sample $x$ with unknown label and $\mathcal{T}_m$ a decision tree, let $\mathcal{L}_m(x)$ denote the set of training samples that share the same leaf as $x$ in tree $\mathcal{T}_m$ for $m = 1, \dots, M$.
Letting $w(x,x_i)$ be the similarity between $x$ and $x_i$, we have

$$w(x,x_i) = \frac{1}{M} \sum_{m=1}^{M} \frac{\mathbb{1}_{\{x_i \in \mathcal{L}_m(x)\}}}{|\mathcal{L}_m(x)|}.$$

Finally, the $l$ training samples with the highest $w(x,x_i)$ values, along with their target values, are proposed as the examples that best explain the prediction of $x$ by the tree-based ensemble model.
