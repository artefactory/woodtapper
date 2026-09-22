import numpy as np
from scipy.stats import cumfreq


from woodtapper.fairness.utils import _invert_cdf, _split_groups


def w2_fair_grad_binary(y_preds, sensitive_attribute, n_steps_cdf):
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
    s = np.asarray(sensitive_attribute, dtype=int)
    n = len(y_preds)
    n0 = np.sum(s == 0)
    n1 = np.sum(s == 1)

    idx0, idx1, y0, y1 = _split_groups(y_preds, s)
    if y0.size == 0 or y1.size == 0:
        return np.zeros(n, dtype=float)
    array_steps_eta, eta_step = np.linspace(y_preds.min(), y_preds.max(), n_steps_cdf + 1, retstep=True)
    lims = (float(array_steps_eta[0]), float(array_steps_eta[-1]))

    def make_cdf(y: np.ndarray) -> np.ndarray:
        return cumfreq(y, numbins=n_steps_cdf + 1, defaultreallimits=lims).cumcount / y.size

    cdf_H0 = make_cdf(y0)
    cdf_H1 = make_cdf(y1)
    u0 = np.interp(y0, array_steps_eta, cdf_H0)
    u1 = np.interp(y1, array_steps_eta, cdf_H1)

    cor1_y0 = _invert_cdf(array_steps_eta, cdf_H1, u0)
    cor0_y1 = _invert_cdf(array_steps_eta, cdf_H0, u1)
    
    grad = np.zeros(n, dtype=float)
 
    grad[idx0] = (2 * (y0 - cor1_y0)) / n0
    grad[idx1] = (2 * (y1 - cor0_y1)) / n1
    return grad
