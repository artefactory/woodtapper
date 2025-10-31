import numpy as np


def iterative_random_choice(probas):
    """
    Function for applying a np.random.choice several times with succesive values of probas.

    Parameters
    ----------
    probas : np.ndarray of shape (n_samples, n_classes)
        Probabilities for each class for each sample.
    Returns
    -------
    np.ndarray of shape (n_samples,)
        Chosen class for each sample.
    """
    thresholds = np.random.uniform(size=len(probas))
    cumulative_weights = np.cumsum(probas, axis=1)
    return np.argmax((cumulative_weights.T > thresholds), axis=0)


def compute_leaf_sizes(train_samples_leaves):
    """
    Given train_samples_leaves shape (n_train, n_trees), return leaf_sizes
    of same shape where leaf_sizes[j, t] = number of train samples in
    the same leaf id as train_samples_leaves[j, t] for tree t.
    """
    n_train, n_trees = train_samples_leaves.shape
    leaf_sizes = np.empty((n_train, n_trees), dtype=np.int32)

    for t in range(n_trees):
        vals, counts = np.unique(train_samples_leaves[:, t], return_counts=True)
        # build mapping from leaf_id -> count
        # vectorized mapping via searchsorted
        order = np.argsort(vals)
        sorted_vals = vals[order]
        sorted_counts = counts[order]
        # map each train sample's leaf id -> count
        idx = np.searchsorted(sorted_vals, train_samples_leaves[:, t])
        leaf_sizes[:, t] = sorted_counts[idx]

    return leaf_sizes
