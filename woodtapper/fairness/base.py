import numpy as np
from scipy.stats import cumfreq


from woodtapper.fairness.utils import _invert_cdf, _split_groups



####################################################################################
###################################### Gradient ###################################
####################################################################################
def w2_fair_grad_binary(y_preds, sensitive_attribute, n_steps_cdf=1024):
    """
    Compute the Wasserstein-2 fair gradient for binary sensitive attributes.

    Parameters
    ----------
    y_preds : np.ndarray
        Predicted values.
    sensitive_attribute : np.ndarray
        Binary sensitive attribute (0 or 1).
    n_steps_cdf : int
        Number of steps for the cumulative distribution function (CDF) approximation.

    Returns
    -------
    grad : np.ndarray
        Gradient adjusted for fairness.
    """
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)
    n = len(y_preds)
    n0 = np.sum(sensitive_attribute == 0)
    n1 = np.sum(sensitive_attribute == 1)

    idx0, idx1, y0, y1 = _split_groups(y_preds, sensitive_attribute)
    if y0.size == 0 or y1.size == 0:
        return np.zeros(n, dtype=float)
    array_steps_eta, eta_step = np.linspace(y_preds.min(), y_preds.max(), n_steps_cdf + 1, retstep=True)
    lims = (float(array_steps_eta[0]), float(array_steps_eta[-1]))

    def make_cdf(y: np.ndarray) -> np.ndarray:
        return cumfreq(y, numbins=n_steps_cdf + 1, defaultreallimits=lims).cumcount / y.size

    cdf_H0 = make_cdf(y0)
    cdf_H1 = make_cdf(y1)
    target_u_proportion0 = np.interp(y0, array_steps_eta, cdf_H0)
    target_u_proportion1 = np.interp(y1, array_steps_eta, cdf_H1)

    cor1_y0 = _invert_cdf(array_steps_eta, cdf_H1, target_u_proportion0)
    cor0_y1 = _invert_cdf(array_steps_eta, cdf_H0, target_u_proportion1)
    
    grad = np.zeros(n, dtype=float)
 
    grad[idx0] = (2 * (y0 - cor1_y0)) / n0
    grad[idx1] = (2 * (y1 - cor0_y1)) / n1
    return grad

def w2_fair_grad(y_preds, sensitive_attribute, n_steps_cdf=1024):
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)
    n = len(y_preds)
    #n0 = np.sum(sensitive_attribute == 0)
    #n1 = np.sum(sensitive_attribute == 1)
    classes = np.unique(sensitive_attribute)
    grad = np.zeros(n, dtype=float)
    if classes.size < 2:
        return grad

    if classes.size == 2:
        return w2_fair_grad_binary(y_preds, sensitive_attribute, n_steps_cdf=n_steps_cdf)

    for group in classes:
        s_ovr = np.where(sensitive_attribute == group, 0, 1) # group (as 0) vs rest (as 1)
        grad_ovr = w2_fair_grad_binary(y_preds, s_ovr, n_steps_cdf=n_steps_cdf)
        idx_group = np.where(sensitive_attribute == group)[0]
        grad[idx_group] = grad_ovr[idx_group]

    return grad

####################################################################################
####################################### Hessian ####################################
####################################################################################

def w2_fair_hess_binary(y_preds, sensitive_attribute):
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)
    n = len(y_preds)
    n0 = np.sum(sensitive_attribute == 0)
    n1 = np.sum(sensitive_attribute == 1)

    idx0, idx1, y0, y1 = _split_groups(y_preds, sensitive_attribute)
    if y0.size == 0 or y1.size == 0:
        return np.zeros(n, dtype=float)
    hess = np.zeros(n, dtype=float)

    h0 = 2 / n0
    h1 = 2 / n1
    hess[idx0] = h0
    hess[idx1] = h1

    return hess

def w2_fair_hess(y_preds, sensitive_attribute, n_steps_cdf=1024):
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)
    n = len(y_preds)
    #n0 = np.sum(sensitive_attribute == 0)
    #n1 = np.sum(sensitive_attribute == 1)
    classes = np.unique(sensitive_attribute)
    hess = np.zeros(n, dtype=float)
    if classes.size < 2:
        return hess

    if classes.size == 2:
        return w2_fair_hess_binary(y_preds, sensitive_attribute)

    for group in classes:
        s_ovr = np.where(sensitive_attribute == group, 0, 1) # group (as 0) vs rest (as 1)
        hess_ovr = w2_fair_hess_binary(y_preds, s_ovr)
        idx_group = np.where(sensitive_attribute == group)[0]
        hess[idx_group] = hess_ovr[idx_group]

    return hess

def w2_fair_grad_hess(y_preds, sensitive_attribute, n_steps_cdf=1024):
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)
    n = len(y_preds)
    #n0 = np.sum(sensitive_attribute == 0)
    #n1 = np.sum(sensitive_attribute == 1)
    classes = np.unique(sensitive_attribute)
    grad = np.zeros(n, dtype=float)
    hess = np.zeros(n, dtype=float)
    if classes.size < 2:
        return grad, hess

    if classes.size == 2:
        grad = w2_fair_grad_binary(y_preds, sensitive_attribute, n_steps_cdf=n_steps_cdf)
        hess = w2_fair_hess_binary(y_preds, sensitive_attribute)
        return grad, hess

    for group in classes:
        s_ovr = np.where(sensitive_attribute == group, 0, 1) # group (as 0) vs rest (as 1)
        grad_ovr = w2_fair_grad_binary(y_preds, s_ovr, n_steps_cdf=n_steps_cdf)
        hess_ovr = w2_fair_hess_binary(y_preds, s_ovr)
        idx_group = np.where(sensitive_attribute == group)[0]
        grad[idx_group] = grad_ovr[idx_group]
        hess[idx_group] = hess_ovr[idx_group]

    return grad, hess