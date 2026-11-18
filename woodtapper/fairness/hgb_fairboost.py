"""Scikit-learn HistGradientBoosting-based fairness-aware estimators."""

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .scikit_loss import HalfBinomialFairLoss, HalfSquaredFairLoss
from .utils import _check_hgb_fair_fit_inputs


class OTHGBFairBoostClassifier(HistGradientBoostingClassifier):
    """
    OTHGBFairBoostClassifier that incorporates fairness constraints into scikit-learn's
    HistGradientBoostingClassifier.
    The binomial log-loss is replaced by the fair loss ``HalfBinomialFairLoss``, built at
    fit time from the sensitive attribute. Only binary classification is supported.

    Parameters
    ----------
    lambda_fairness_value : float
        The weight of the fairness constraint in the custom loss function.
    n_steps_cdf : int, default=1024
        The number of steps to use for approximating the cumulative distribution function (CDF) in the fairness loss.
    fairness_mode : str, default="Demographic_Parity"
        The fairness criterion to enforce.
    early_stopping : 'auto' or bool, default=False
        Early stopping is only supported with ``validation_fraction=None`` since the
        fairness term must be computed on the full training set.
    All other parameters are inherited from sklearn.ensemble.HistGradientBoostingClassifier.
    """

    def __init__(
        self,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features="from_dtype",
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping=False,
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        class_weight=None,
        partitions_target=None,
    ):
        super().__init__(
            loss="log_loss",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.lambda_fairness_value = lambda_fairness_value
        self.n_steps_cdf = n_steps_cdf
        self.fairness_mode = fairness_mode
        self.partitions_target = partitions_target

    def fit(self, X, y, sensitive_attribute, sample_weight=None):
        """
        Fit the OTHGBFairBoostClassifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The binary target values.
        sensitive_attribute : array-like of shape (n_samples,)
            The sensitive attribute used for enforcing fairness.
        sample_weight : array-like of shape (n_samples,), optional (default=None)
            Weights of the training samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.partitions_target = [(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]
        self._sensitive_attribute = _check_hgb_fair_fit_inputs(
            self, y, sensitive_attribute
        )
        try:
            super().fit(X, y, sample_weight=sample_weight)
        finally:
            del self._sensitive_attribute
        return self

    def _get_loss(self, sample_weight):
        if self.n_trees_per_iteration_ != 1:
            raise ValueError(
                "OTHGBFairBoostClassifier only supports binary classification."
            )
        return HalfBinomialFairLoss(
            sensitive_attribute=self._sensitive_attribute,
            lambda_fairness_value=self.lambda_fairness_value,
            n_steps_cdf=self.n_steps_cdf,
            fairness_mode=self.fairness_mode,
            sample_weight=sample_weight,
            partitions_target=self.partitions_target,
        )


class OTHGBFairBoostRegressor(HistGradientBoostingRegressor):
    """
    OTHGBFairBoostRegressor that incorporates fairness constraints into scikit-learn's
    HistGradientBoostingRegressor.
    The squared error is replaced by the fair loss ``HalfSquaredFairLoss``, built at
    fit time from the sensitive attribute.

    Parameters
    ----------
    lambda_fairness_value : float
        The weight of the fairness constraint in the custom loss function.
    n_steps_cdf : int, default=1024
        The number of steps to use for approximating the cumulative distribution function (CDF) in the fairness loss.
    fairness_mode : str, default="Demographic_Parity"
        The fairness criterion to enforce.
    early_stopping : 'auto' or bool, default=False
        Early stopping is only supported with ``validation_fraction=None`` since the
        fairness term must be computed on the full training set.
    All other parameters are inherited from sklearn.ensemble.HistGradientBoostingRegressor.
    """

    def __init__(
        self,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features="from_dtype",
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping=False,
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        partitions_target=None,
    ):
        super().__init__(
            loss="squared_error",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.lambda_fairness_value = lambda_fairness_value
        self.n_steps_cdf = n_steps_cdf
        self.fairness_mode = fairness_mode
        self.partitions_target = partitions_target

    def fit(self, X, y, sensitive_attribute, sample_weight=None):
        """
        Fit the OTHGBFairBoostRegressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sensitive_attribute : array-like of shape (n_samples,)
            The sensitive attribute used for enforcing fairness.
        sample_weight : array-like of shape (n_samples,), optional (default=None)
            Weights of the training samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.partitions_target = [(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]
        self._sensitive_attribute = _check_hgb_fair_fit_inputs(
            self, y, sensitive_attribute
        )
        try:
            super().fit(X, y, sample_weight=sample_weight)
        finally:
            del self._sensitive_attribute
        return self

    def _get_loss(self, sample_weight):
        return HalfSquaredFairLoss(
            sensitive_attribute=self._sensitive_attribute,
            lambda_fairness_value=self.lambda_fairness_value,
            n_steps_cdf=self.n_steps_cdf,
            fairness_mode=self.fairness_mode,
            sample_weight=sample_weight,
            partitions_target=self.partitions_target,
        )
