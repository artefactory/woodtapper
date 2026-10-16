import numpy as np


def _invert_cdf(array_steps_eta, cdf_H, target_u_proportion):
    """
    Invert the cumulative distribution function (CDF) for a given target proportion.

    Parameters
    ----------
    array_steps_eta : np.ndarray
        Array of steps for the CDF.
    cdf_H : np.ndarray
        CDF values corresponding to array_steps_eta.
    target_u_proportion : np.ndarray
        Target proportion values to invert.

    Returns
    -------
    np.ndarray
        Inverted CDF values corresponding to the target proportions.
    """
    H = np.maximum.accumulate(cdf_H)
    u = np.clip(target_u_proportion, H[0], H[-1])
    j = np.clip(np.searchsorted(H, u, side="left"), 1, H.size - 1)

    h_lo, h_hi = H[j - 1], H[j]
    e_lo, e_hi = array_steps_eta[j - 1], array_steps_eta[j]

    dh = h_hi - h_lo
    t = np.where(dh > 0.0, (u - h_lo) / np.where(dh > 0.0, dh, 1.0), 1.0)
    return e_lo + t * (e_hi - e_lo)


def _split_groups(y_preds, sensitive_attribute):
    """
    Split predictions into two groups based on a binary sensitive attribute.

    Parameters
    ----------
    y_preds : np.ndarray
        Predicted values.
    sensitive_attribute : np.ndarray
        Binary sensitive attribute (0 or 1).

    Returns
    -------
    idx0 : np.ndarray
        Indices of the first group (sensitive attribute == 0).
    idx1 : np.ndarray
        Indices of the second group (sensitive attribute == 1).
    y0 : np.ndarray
        Predicted values for the first group.
    y1 : np.ndarray
        Predicted values for the second group.
    """
    y_preds = np.asarray(y_preds, dtype=float)
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)

    idx0 = np.where(sensitive_attribute == 0)[0]
    idx1 = np.where(sensitive_attribute == 1)[0]
    y0 = y_preds[idx0]
    y1 = y_preds[idx1]
    return idx0, idx1, y0, y1


def _sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out
