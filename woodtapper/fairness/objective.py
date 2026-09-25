"""
Fairness objective functions for LightGBM.
"""

import numpy as np

from .base import w2_fair_grad_hess
from .utils import _sigmoid


def derive_fair_grad_hess_terms(
    y_true,
    y_preds,
    sensitive_attribute,
    fairness_mode="Demographic_Parity",
    n_steps_cdf=1024,
    is_classification=True,
):
    """
    Derive the gradient and hessian terms for the fairness objective.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_preds : np.ndarray
        Predicted scores.
    sensitive_attribute : np.ndarray
        Sensitive attribute values.
    fairness_mode : str, optional
        Fairness mode, by default "Demographic_Parity".
    n_steps_cdf : int, optional
        Number of steps for the CDF approximation, by default 1024.
    is_classification : bool, optional
        Whether the task is classification, by default True.

    Returns
    -------
    grad : np.ndarray
        Gradient of the fairness objective.
    hess : np.ndarray
        Hessian of the fairness objective.
    """
    if fairness_mode == "Demographic_Parity":
        grad, hess = w2_fair_grad_hess(
            y_preds, sensitive_attribute, n_steps_cdf=n_steps_cdf
        )

    elif fairness_mode == "Equality_of_Odds":
        mask_pos = y_true == 1.0
        mask_neg = y_true == 0.0
        grad_pos, hess_pos = w2_fair_grad_hess(
            y_preds[mask_pos], sensitive_attribute[mask_pos], n_steps_cdf=n_steps_cdf
        )
        grad_neg, hess_neg = w2_fair_grad_hess(
            y_preds[mask_neg], sensitive_attribute[mask_neg], n_steps_cdf=n_steps_cdf
        )

        grad = np.zeros_like(y_preds)
        hess = np.zeros_like(y_preds)
        pi_pos = float(np.mean(mask_pos))
        pi_neg = float(np.mean(mask_neg))
        grad[mask_pos] = pi_pos * grad_pos
        hess[mask_pos] = pi_pos * hess_pos
        grad[mask_neg] = pi_neg * grad_neg
        hess[mask_neg] = pi_neg * hess_neg

    else:
        raise ValueError(f"Unsupported fairness mode: {fairness_mode}")

    if is_classification:
        dscore_dz = y_preds * (1.0 - y_preds)
        d2score_dz2 = dscore_dz * (1.0 - 2.0 * y_preds)
        grad = grad * dscore_dz
        hess = hess * (dscore_dz**2) + grad * d2score_dz2

    return grad, hess


def build_fair_loss(
    sensitive_attribute,
    lambda_fairness_value,
    fairness_mode="Demographic_Parity",
    n_steps_cdf=1024,
    is_classification=True,
):
    def fair_loss(preds, train_data):  # LGBM signature ?
        y_true = train_data.get_label()
        y_preds = preds
        n = y_true.size

        grad_fairness_perf, hess_fairness_perf = derive_fair_grad_hess_terms(
            y_true,
            y_preds,
            sensitive_attribute,
            fairness_mode=fairness_mode,
            n_steps_cdf=n_steps_cdf,
            is_classification=is_classification,
        )

        if is_classification:
            y_preds_probas = _sigmoid(preds)
            grad_predictive_perf = y_preds_probas - y_true
            hess_predictive_perf = y_preds_probas * (1.0 - y_preds_probas)
        else:  # regression case
            y_preds_probas = preds
            grad_predictive_perf = y_preds_probas - y_true
            hess_predictive_perf = np.ones_like(y_true)

        grad = (
            (1 / n) * grad_predictive_perf
        ) + lambda_fairness_value * grad_fairness_perf
        hess = (
            (1 / n) * hess_predictive_perf
        ) + lambda_fairness_value * hess_fairness_perf

        sample_weights = train_data.get_weight()  # Returns None or a 1D numpy array
        if sample_weights is not None and len(sample_weights) > 0:
            grad *= sample_weights
            hess *= sample_weights

        return grad, hess

    return fair_loss
