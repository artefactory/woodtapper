import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _safe_div(num: float, den: float) -> float:
    """Safe division helper (returns NaN if denominator is zero)."""
    return float(num / den) if den != 0 else np.nan


def _binary_rates(y_true, y_preds):
    """
    Compute classic binary classification rates.

    Important: y_true and y_pred must be 0/1 encoded (positive class = 1).

    Returns:
        - TPR (recall / sensitivity)
        - FPR
        - PPV (precision)
        - PPR (predicted positive rate) -> used for DP/DI
    """
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    # cm is guaranteed to be 2x2 with labels=[0,1]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    ppv = _safe_div(tp, tp + fp)
    ppr = float(np.mean(y_preds)) if y_preds.size else np.nan

    return {"TPR": tpr, "FPR": fpr, "PPV": ppv, "PPR": ppr}


def _worst_pairwise_gap(values):
    vals = [v for v in values.values() if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return np.nan
    return float(np.max(vals) - np.min(vals))


def _worst_pairwise_di(values):
    vals = [v for v in values.values() if v is not None and not np.isnan(v)]
    vals = [v for v in vals if v > 0]  # if zero exists -> DI collapses to 0

    if len(vals) < 2:
        return np.nan

    return float(np.min(vals) / np.max(vals))


def _one_vs_rest_gap(values, weights):
    """
    Computes the worst one-vs-rest gap: max_s |r_s - r_{-s}|.
    
    Parameters
    ----------
    values  : {group s: rate r_s}, i.e. PPR, TPR, FPR or PPV of each group.
    weights : {group s: the number of samples the rate is computed over}, i.e. PPR -> n_s, 
               TPR -> #(Y=1), FPR -> #(Y=0), PPV -> #(Yhat=1).
    """
    keys = [
        g
        for g, v in values.items()
        if v is not None
        and not np.isnan(v)
        and weights.get(g) is not None
        and weights[g] > 0
    ]
    if len(keys) < 2:
        return np.nan
 
    r = np.array([values[g] for g in keys], dtype=float)
    w = np.array([weights[g] for g in keys], dtype=float)
 
    num_total = np.sum(w * r)
    w_total = np.sum(w)
 
    rest = (num_total - w * r) / (w_total - w)
 
    return float(np.max(np.abs(r - rest)))


def derive_fairness_metrics(y_true, y_preds, sensitive_attribute, min_group_n=10):
    if sensitive_attribute.ndim > 2 or (
        sensitive_attribute.ndim == 2 and sensitive_attribute.shape[1] > 1
    ):
        raise ValueError("sensitive_attribute must be 1-dimensional")
    if isinstance(sensitive_attribute, np.ndarray):
        sensitive_attribute = sensitive_attribute.ravel()
    else:
        sensitive_attribute = np.asarray(sensitive_attribute).ravel()
    groups = pd.Series(sensitive_attribute).astype(str)
    rows = []

    ppr_by_g = {}
    tpr_by_g = {}
    fpr_by_g = {}
    ppv_by_g = {}
    
    n_by_g = {}
    n_pos_by_g = {}
    n_neg_by_g = {}
    n_pred_pos_by_g = {}

    for g in groups.unique():
        idx = (groups == g).to_numpy()
        n_g = int(idx.sum())

        if n_g < min_group_n:
            rows.append(
                {
                    "group": g,
                    "n": n_g,
                    "PPR": np.nan,
                    "TPR": np.nan,
                    "FPR": np.nan,
                    "PPV": np.nan,
                }
            )
            continue
            
        yt_g = y_true[idx].astype(int)
        yp_g = y_preds[idx].astype(int)

        rates = _binary_rates(yt_g, yp_g)
        ppr_by_g[g] = rates["PPR"]
        tpr_by_g[g] = rates["TPR"]
        fpr_by_g[g] = rates["FPR"]
        ppv_by_g[g] = rates["PPV"]
        
        n_by_g[g] = n_g
        n_pos_by_g[g] = int((yt_g == 1).sum())
        n_neg_by_g[g] = int((yt_g == 0).sum())
        n_pred_pos_by_g[g] = int((yp_g == 1).sum())

        rows.append({"group": g, "n": n_g, **rates})

    rates_df = pd.DataFrame(rows)
    
    # max - min gaps
    dp_gap = _worst_pairwise_gap(ppr_by_g)
    di = _worst_pairwise_di(ppr_by_g)
    eopp_gap = _worst_pairwise_gap(tpr_by_g)
    fpr_gap = _worst_pairwise_gap(fpr_by_g)
    ppv_gap = _worst_pairwise_gap(ppv_by_g)
    
    # one-vs-rest gaps
    dp_ovr_gap = _one_vs_rest_gap(ppr_by_g, n_by_g)
    eopp_ovr_gap = _one_vs_rest_gap(tpr_by_g, n_pos_by_g)
    fpr_ovr_gap = _one_vs_rest_gap(fpr_by_g, n_neg_by_g)
    ppv_ovr_gap = _one_vs_rest_gap(ppv_by_g, n_pred_pos_by_g)

    return (dp_gap, di, eopp_gap, fpr_gap, ppv_gap, dp_ovr_gap, 
            eopp_ovr_gap, fpr_ovr_gap, ppv_ovr_gap, rates_df)
