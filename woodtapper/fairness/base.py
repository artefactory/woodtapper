"""
Wasserstein-2 fairness gradient and hessian computations for LightGBM.
"""

import numpy as np


from woodtapper.fairness.utils import _invert_cdf, _split_groups, _make_cdf


def w2_fair_grad_binary(y_preds,idx0, idx1, n_steps_cdf=1024):
    """
    Compute the Wasserstein-2 fair gradient for binary sensitive attributes.

    Parameters
    ----------
    y_preds : np.ndarray
        Predicted values.
    idx0 : np.ndarray
        Indices of group 0 in y_preds.
    idx1 : np.ndarray
        Indices of group 1 in y_preds.
    n_steps_cdf : int
        Number of steps for the cumulative distribution function (CDF) approximation.

    Returns
    -------
    grad : np.ndarray
        Gradient adjusted for fairness.
    """
    n = len(y_preds)
    n0 = idx0.size
    n1 = idx1.size
    if idx0.size == 0 or idx1.size == 0:
        return np.zeros(n, dtype=float)
    
    y0 = y_preds[idx0]
    y1 = y_preds[idx1]
    
    if np.min(y_preds) == np.max(y_preds):
        return np.zeros(n, dtype=float)
    array_steps_eta, eta_step = np.linspace(
        y_preds.min(), y_preds.max(), n_steps_cdf + 1, retstep=True
    )
    lims = (float(array_steps_eta[0]), float(array_steps_eta[-1]))


    cdf_H0 = _make_cdf(y0, n_steps_cdf, lims)
    cdf_H1 = _make_cdf(y1, n_steps_cdf, lims)
    target_u_proportion0 = np.interp(y0, array_steps_eta, cdf_H0)
    target_u_proportion1 = np.interp(y1, array_steps_eta, cdf_H1)

    cor1_y0 = _invert_cdf(array_steps_eta, cdf_H1, target_u_proportion0)
    cor0_y1 = _invert_cdf(array_steps_eta, cdf_H0, target_u_proportion1)

    grad = np.zeros(n, dtype=float)

    grad[idx0] = (2 * (y0 - cor1_y0)) / n0
    grad[idx1] = (2 * (y1 - cor0_y1)) / n1
    return grad


def w2_fair_grad_hess(y_preds, partitions_group, n_steps_cdf=1024):
    """
    Compute the Wasserstein-2 fair gradient and hessian for binary or multi-class sensitive attributes.

    Parameters
    ----------
    y_preds : np.ndarray
        Predicted values.
    sensitive_attribute : np.ndarray
        Sensitive attribute values (binary or multi-class).
    n_steps_cdf : int, optional
        Number of steps for the CDF approximation, by default 1024.

    Returns
    -------
    grad : np.ndarray
        Gradient adjusted for fairness.
    hess : np.ndarray
        Hessian adjusted for fairness.
    """
    n = len(y_preds)
    grad = np.zeros(n, dtype=float)
    hess = np.zeros(n, dtype=float)
    partitions = partitions_group
    if len(partitions) == 0:
        return grad, hess

    for idx0, idx1 in partitions:
        if not(idx0.size and idx1.size):
            raise ValueError("One of the groups has size 0.")

        #Gradients
        grad_ovr = w2_fair_grad_binary(
                    y_preds, idx0, idx1, n_steps_cdf=n_steps_cdf
                )
        grad[idx0] = grad_ovr[idx0] # Group 0 only, the rest no update
        hess[idx0] = 2 / idx0.size
        

    return grad, hess
