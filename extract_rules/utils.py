import numpy as np
from scipy.stats import binom
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import get_scorer


class Node:
    """
    Tree node class

    Parameters
    ----------
    feature : int
        Current node feature indice splitting
    treshold : flaot
        Current node treshold splittting
    side : str
        Side of the rule. 'L' for Left (i.e. less or equal) and 'R' for right (i.e. gretter)
    node_id : int
       Current  Node id
    children : list of Node
        Child Nodes if not a leaf node
    Returns
    ----------
    Node: Node
        The current Node instance
    """

    def __init__(self, feature=None, treshold=-1, side=None, node_id=-1, *children):
        self.node_id = node_id
        self.feature = feature
        self.treshold = treshold
        self.side = side
        if children:
            self.children = children
        else:
            self.children = []

def get_top_rules(all_possible_rules_list_str,p0):
    unique_str_rules, indices_rules, frequence_rules = np.unique(
    all_possible_rules_list_str, return_counts=True, return_index=True
    )  # get the unique rules and count
    frequence_rules = frequence_rules / frequence_rules.sum()  # convert to frequency
    unique_str_rules_and_freq = zip(unique_str_rules, frequence_rules) # combine rules and frequency
    all_rules_sorted = sorted(unique_str_rules_and_freq, key=lambda x: x[1], reverse=True) # sort by frequency
    all_possible_rules_and_freq_list = [(eval(unique_str_rule),freq) for unique_str_rule, freq in all_rules_sorted if freq > p0] # filter by p0
    if len(all_possible_rules_and_freq_list) == 0:
        if len(all_possible_rules_and_freq_list) == 0:
            raise ValueError(
                "No rule found with the given p0 value. Try to decrease it."
            )
    all_possible_rules_list, all_possible_freq_list =zip(*all_possible_rules_and_freq_list) # unzip
    return all_possible_rules_list, all_possible_freq_list


def compute_staibility_criterion(model):
    M = model.n_estimators
    list_p0 = np.arange(0.1, 1, 0.08)
    list_epsilon = []
    print("Computing stability criterion...")
    for p0_curr in list_p0:
        epsilon_numerator = np.sum(
            [
                binom.cdf(k=p0_curr * M, n=M, p=pm)
                * (1 - binom.cdf(k=p0_curr * M, n=M, p=pm))
                for pm in model.all_possible_rules_frequency_list
            ]
        )
        epsilon_denominator = np.sum(
            [
                (1 - binom.cdf(k=p0_curr * M, n=M, p=pm))
                for pm in model.all_possible_rules_frequency_list
            ]
        )
        epsilon = (
            epsilon_numerator / epsilon_denominator
            if epsilon_denominator > 0
            else 0
        )
        list_epsilon.append(epsilon)
    print("***** \n Stability criterion value:", np.mean(list_epsilon), "\n*****")



def ridge_cv_positive(X, y, alphas=np.linspace(0,1,25), scoring="neg_mean_squared_error", cv=5, random_state=None):
    """
    Cross-validate Ridge regression with positive=True manually,
    and return the best model fitted on all data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,)
        Target values.

    alphas : list or array-like
        List of alpha (regularization strength) values to test.

    scoring : str or callable
        A scoring function name (from sklearn) or a callable with signature
        `scoring(estimator, X_val, y_val)` returning a scalar score.

    cv : int or cross-validation generator, default=5
        Number of CV folds or a CV splitter.

    random_state : int, optional
        Random state for reproducibility (if cv is an integer).

    Returns
    -------
    best_alpha : float
        The alpha value that yields the best mean CV score.

    best_model : sklearn.linear_model.Ridge
        Ridge model (positive=True) trained on all data with best_alpha.

    results : dict
        Dictionary mapping alpha -> mean CV score.
    """

    # Handle scoring
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring

    # Handle CV splitter
    if isinstance(cv, int):
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    results = {}

    # Cross-validation loop
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, positive=True, random_state=random_state)
        scores = []

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            m = clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            score = scorer._score_func(y_val,y_pred)
            scores.append(score)

        results[alpha] = np.mean(scores)

    # Select best alpha
    best_alpha = min(results, key=results.get)
    print(results)
    print("Best alpha:", best_alpha)

    return best_alpha, results

