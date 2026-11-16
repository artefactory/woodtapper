"""
Fairness-aware losses compatible with scikit-learn's HistGradientBoosting models.

scikit-learn's HistGradientBoosting estimators only accept losses deriving from
``sklearn._loss.loss.BaseLoss``. Their per-sample Cython kernels cannot express the
Wasserstein-2 fairness term, which depends on the whole prediction vector, so we
subclass the sklearn losses and override ``gradient_hessian`` at the Python level.
"""

import numpy as np
from sklearn._loss.loss import HalfBinomialLoss, HalfSquaredError

from .objective import build_fair_loss


class _FairLossMixin:
    """
    Replace the gradients and hessians of a sklearn loss by those of ``build_fair_loss``.

    Gradients and hessians are rescaled by ``n_samples``, so the optimized objective is
    the same as for the LightGBM and XGBoost fair models. The loss value itself (used for
    early stopping scores) is the one of the parent sklearn loss.
    """

    _is_classification = None

    def _init_fair_loss(
        self,
        sensitive_attribute,
        lambda_fairness_value,
        n_steps_cdf,
        fairness_mode,
        partitions_target,
    ):
        self.n_samples = len(sensitive_attribute)
        # HGB skips the hessian computation for constant-hessian losses.
        self.constant_hessian = False
        self._fair_loss = build_fair_loss(
            sensitive_attribute=sensitive_attribute,
            partitions_target=partitions_target,
            lambda_fairness_value=lambda_fairness_value,
            fairness_mode=fairness_mode,
            n_steps_cdf=n_steps_cdf,
            is_classification=self._is_classification,
        )

    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        hessian_out=None,
        n_threads=1,
    ):
        raw_prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
        if raw_prediction.shape[0] != self.n_samples:
            raise ValueError(
                f"The fair loss was built for {self.n_samples} samples but received "
                f"{raw_prediction.shape[0]} predictions."
            )
        grad, hess = self._fair_loss(
            np.asarray(y_true, dtype=float), raw_prediction, sample_weight
        )
        # build_fair_loss scales the objective by 1/n. Rescale by n so hessians stay
        # comparable to sklearn's (HGB's hard-coded min_hessian_to_split=1e-3 would
        # otherwise prevent splits on large datasets). Newton leaf values are unchanged.
        grad *= self.n_samples
        hess *= self.n_samples

        if gradient_out is None:
            gradient_out = np.empty_like(raw_prediction)
        if hessian_out is None:
            hessian_out = np.empty_like(gradient_out)
        gradient_out.reshape(-1)[:] = grad
        hessian_out.reshape(-1)[:] = hess
        return gradient_out, hessian_out

    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        n_threads=1,
    ):
        gradient_out, _ = self.gradient_hessian(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        return gradient_out


class HalfBinomialFairLoss(_FairLossMixin, HalfBinomialLoss):
    """
    Half binomial loss penalized by the Wasserstein-2 fairness term.

    Parameters
    ----------
    sensitive_attribute : array-like of shape (n_samples,)
        Sensitive attribute aligned with the training samples.
    lambda_fairness_value : float
        The weight of the fairness constraint.
    n_steps_cdf : int, default=1024
        Number of steps used to approximate the CDFs in the fairness term.
    fairness_mode : str, default="Demographic_Parity"
        The fairness criterion to enforce.
    sample_weight : array-like of shape (n_samples,), default=None
        Training sample weights.
    """

    _is_classification = True

    def __init__(
        self,
        sensitive_attribute,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        sample_weight=None,
        partitions_target=None,
    ):
        super().__init__(sample_weight=sample_weight)
        self._init_fair_loss(
            sensitive_attribute,
            lambda_fairness_value,
            n_steps_cdf,
            fairness_mode,
            partitions_target,
        )


class HalfSquaredFairLoss(_FairLossMixin, HalfSquaredError):
    """
    Half squared error penalized by the Wasserstein-2 fairness term.

    Parameters
    ----------
    sensitive_attribute : array-like of shape (n_samples,)
        Sensitive attribute aligned with the training samples.
    lambda_fairness_value : float
        The weight of the fairness constraint.
    n_steps_cdf : int, default=1024
        Number of steps used to approximate the CDFs in the fairness term.
    fairness_mode : str, default="Demographic_Parity"
        The fairness criterion to enforce.
    sample_weight : array-like of shape (n_samples,), default=None
        Training sample weights.
    """

    _is_classification = False

    def __init__(
        self,
        sensitive_attribute,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        sample_weight=None,
        partitions_target=None,
    ):
        super().__init__(sample_weight=sample_weight)
        self._init_fair_loss(
            sensitive_attribute,
            lambda_fairness_value,
            n_steps_cdf,
            fairness_mode,
            partitions_target,
        )
