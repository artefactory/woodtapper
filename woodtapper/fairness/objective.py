"""
Fairness objective functions for LightGBM.
"""

import numpy as np

from .base import w2_fair_grad_hess
from .utils import _sigmoid


def derive_fair_grad_hess_terms(
    partitions_group,
    partitions_target,
    y_preds,
    fairness_mode="Demographic_Parity",
    n_steps_cdf=1024,
    is_classification=True,
):
    """
    Derive the gradient and hessian terms for the fairness objective.

    Parameters
    ----------
    partitions_group : list of tuples
        List of partitions for the groups.
    partitions_target : list of tuples
        List of partitions for the target.
    y_preds : np.ndarray
        Predicted scores.
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

    if is_classification:
        y_preds_probas = _sigmoid(y_preds)
    else:
        y_preds_probas = y_preds

    if fairness_mode == "Demographic_Parity":
        grad, hess = w2_fair_grad_hess(
            y_preds_probas, partitions_group, n_steps_cdf=n_steps_cdf
        )

    elif fairness_mode == "Equality_of_Odds":
        grad = np.zeros_like(y_preds_probas)
        hess = np.zeros_like(y_preds_probas)
        mask_pos = partitions_target[0]
        mask_neg = partitions_target[1]
        pi_pos = float(np.mean(mask_pos))
        pi_neg = float(np.mean(mask_neg))

        grad_pos, hess_pos = w2_fair_grad_hess(
            y_preds_probas[mask_pos], partitions_group, n_steps_cdf=n_steps_cdf
        )
        grad_neg, hess_neg = w2_fair_grad_hess(
            y_preds_probas[mask_neg], partitions_group, n_steps_cdf=n_steps_cdf
        )

        grad[mask_pos] = pi_pos * grad_pos
        hess[mask_pos] = pi_pos * hess_pos
        grad[mask_neg] = pi_neg * grad_neg
        hess[mask_neg] = pi_neg * hess_neg

    else:
        raise ValueError(f"Unsupported fairness mode: {fairness_mode}")

    if is_classification:
        dscore_dz = y_preds_probas * (1.0 - y_preds_probas)
        d2score_dz2 = dscore_dz * (1.0 - 2.0 * y_preds_probas)
        grad_final = grad * dscore_dz
        hess_final = hess * (dscore_dz**2) + grad * d2score_dz2
    else:
        grad_final = grad
        hess_final = hess

    return grad_final, hess_final


def build_fair_loss(
    sensitive_attribute,
    lambda_fairness_value,
    fairness_mode="Demographic_Parity",
    n_steps_cdf=1024,
    is_classification=True,
):
    sensitive_attribute = np.asarray(sensitive_attribute, dtype=int)
    classes = np.unique(sensitive_attribute)
    partitions_sensitive_attribute = []
    if classes.size < 2:
        raise ValueError("Sensitive attribute must have at least two classes.")
    if classes.size == 2:
        partitions_sensitive_attribute.append(
            (
                np.flatnonzero(sensitive_attribute == 0),
                np.flatnonzero(sensitive_attribute == 1),
            )
        )
    elif classes.size > 2:
        for group in classes:
            mask = sensitive_attribute == group
            partitions_sensitive_attribute.append(
                (np.flatnonzero(mask), np.flatnonzero(~mask))
            )

    partitions_target = None

    def fair_loss(y_true, y_pred, weight=None):
        nonlocal partitions_target
        n = y_true.size
        if partitions_target is None:
            partitions_target = [
                (np.flatnonzero(y_true == 0), np.flatnonzero(y_true == 1))
            ]

        grad_fairness_perf, hess_fairness_perf = derive_fair_grad_hess_terms(
            partitions_group=partitions_sensitive_attribute,
            partitions_target=partitions_target,
            y_preds=y_pred,
            fairness_mode=fairness_mode,
            n_steps_cdf=n_steps_cdf,
            is_classification=is_classification,
        )

        if is_classification:
            y_preds_probas = _sigmoid(y_pred)
            grad_predictive_perf = y_preds_probas - y_true
            hess_predictive_perf = y_preds_probas * (1.0 - y_preds_probas)
        else:  # regression case
            y_preds_probas = y_pred
            grad_predictive_perf = y_preds_probas - y_true
            hess_predictive_perf = np.ones_like(y_true)

        grad = (
            (1 / n) * grad_predictive_perf
        ) + lambda_fairness_value * grad_fairness_perf
        hess = (
            (1 / n) * hess_predictive_perf
        ) + lambda_fairness_value * hess_fairness_perf

        if weight is not None and len(weight) > 0:
            grad *= weight
            hess *= weight

        return grad, hess

    return fair_loss
