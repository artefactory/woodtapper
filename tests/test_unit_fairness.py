import pytest
import numpy as np

from woodtapper.fairness import OTFairBoostClassifier, OTFairBoostRegressor
from woodtapper.fairness.base import w2_fair_grad_binary


def test_fair_gradient_is_zero_for_constant_predictions():
    predictions = np.full(4, 0.5)
    sensitive_attribute = np.array([0, 0, 1, 1])

    gradient = w2_fair_grad_binary(predictions, sensitive_attribute, n_steps_cdf=32)

    np.testing.assert_array_equal(gradient, np.zeros(4))


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
