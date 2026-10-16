import pytest

from woodtapper.fairness import OTFairBoostClassifier, OTFairBoostRegressor


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
