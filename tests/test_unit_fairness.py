import pytest
import numpy as np

from woodtapper.fairness import (
    OTFairBoostClassifier,
    OTFairBoostRegressor,
    OTXFairBoostClassifier,
    OTXFairBoostRegressor,
)
from woodtapper.fairness.base import w2_fair_grad_binary


def test_fair_gradient_is_zero_for_constant_predictions():
    predictions = np.full(4, 0.5)
    idx0 = np.array([0, 1])
    idx1 = np.array([1, 2])

    gradient = w2_fair_grad_binary(predictions, idx0, idx1, n_steps_cdf=32)

    np.testing.assert_array_equal(gradient, np.zeros(4))


def test_xgb_parameters_exclude_fairness_and_fit_only_options():
    model = OTXFairBoostClassifier(
        lambda_fairness_value=0.1,
        n_steps_cdf=32,
        fairness_mode="Demographic_Parity",
    )

    xgb_params = model.get_xgb_params()
    assert not {
        "lambda_fairness_value",
        "n_steps_cdf",
        "fairness_mode",
    }.intersection(xgb_params)


def test_xgb_regressor_parameters_exclude_fairness_options():
    model = OTXFairBoostRegressor(
        lambda_fairness_value=0.1,
        n_steps_cdf=32,
        fairness_mode="Demographic_Parity",
    )

    xgb_params = model.get_xgb_params()
    assert not {
        "lambda_fairness_value",
        "n_steps_cdf",
        "fairness_mode",
    }.intersection(xgb_params)
    assert xgb_params["reg_lambda"] == 0.0


def test_xgb_regressor_rejects_nonzero_min_child_weight(simple_regression_data):
    X, y = simple_regression_data
    model = OTXFairBoostRegressor(
        lambda_fairness_value=0.1,
        min_child_weight=1.0,
        n_estimators=2,
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="min_child_weight to be 0.0"):
        model.fit(X, y, sensitive_attribute=X[:, 0] > 0)


def test_classifier_rejects_explicit_objective(simple_dataset):
    X, y = simple_dataset
    sensitive_attribute = X[:, 0] > 0
    model = OTFairBoostClassifier(lambda_fairness_value=0.1, objective="binary")

    with pytest.raises(ValueError, match="explicit objective"):
        model.fit(X, y, sensitive_attribute=sensitive_attribute)


def test_regressor_rejects_explicit_objective(simple_regression_data):
    X, y = simple_regression_data
    sensitive_attribute = X[:, 0] > 0
    model = OTFairBoostRegressor(lambda_fairness_value=0.1, objective="regression")

    with pytest.raises(ValueError, match="explicit objective"):
        model.fit(X, y, sensitive_attribute=sensitive_attribute)
