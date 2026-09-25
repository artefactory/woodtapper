import numpy as np


def _invert_cdf(array_steps_eta, cdf_H, target_u_proportion):
    H = np.maximum.accumulate(cdf_H)
    u = np.clip(target_u_proportion, H[0], H[-1])
    j = np.clip(np.searchsorted(H, u, side="left"), 1, H.size - 1)

    h_lo, h_hi = H[j - 1], H[j]
    e_lo, e_hi = array_steps_eta[j - 1], array_steps_eta[j]

    dh = h_hi - h_lo
    t = np.where(dh > 0.0, (u - h_lo) / np.where(dh > 0.0, dh, 1.0), 1.0)
    return e_lo + t * (e_hi - e_lo)


def _split_groups(y_preds, sensitive_attribute):
    y_preds = np.asarray(y_preds, dtype=float)
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)

    idx0 = np.where(sensitive_attribute == 0)[0]
    idx1 = np.where(sensitive_attribute == 1)[0]
    y0 = y_preds[idx0]
    y1 = y_preds[idx1]
    return idx0, idx1, y0, y1
