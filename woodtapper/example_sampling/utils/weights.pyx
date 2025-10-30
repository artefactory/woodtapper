# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
cimport cython

import numpy as np
cimport numpy as np

ctypedef np.int32_t INT_t
ctypedef np.float64_t FLOAT_t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_kernel_weights(np.ndarray[INT_t, ndim=2] leafs_by_sample not None,
                           np.ndarray[INT_t, ndim=2] train_samples_leaves not None,
                           np.ndarray[INT_t, ndim=2] leaf_sizes not None):
    """
    Fully working GRF kernel weight computation (single-threaded, Cython optimized).
    """
    cdef:
        INT_t[:, :] leaves = leafs_by_sample
        INT_t[:, :] trains = train_samples_leaves
        INT_t[:, :] sizes = leaf_sizes
        Py_ssize_t n_samples = leaves.shape[0]
        Py_ssize_t n_trees = leaves.shape[1]
        Py_ssize_t n_train = trains.shape[0]
        np.ndarray[FLOAT_t, ndim=2] result = np.empty((n_samples, n_train), dtype=np.float64)
        FLOAT_t[:, :] res = result
        Py_ssize_t i, j, t
        double s, inv_leaf_size
        INT_t lv, tv

    for i in range(n_samples):
        for j in range(n_train):
            s = 0.0
            for t in range(n_trees):
                lv = leaves[i, t]
                tv = trains[j, t]
                if lv == tv:
                    inv_leaf_size = 1.0 / sizes[j, t]
                    s += inv_leaf_size
            res[i, j] = s / n_trees

    return result
