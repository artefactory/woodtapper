import numpy as np
import pytest
import warnings
import xgboost as xgb

from woodtapper.fairness import (
    OTFairBoostClassifier,
    OTFairBoostRegressor,
    OTXFairBoostClassifier,
    OTXFairBoostRegressor,
)


def test_fair_classifier_fit_and_predict(simple_dataset):
    X, y = simple_dataset
    sensitive_attribute = X[:, 0] > 0
    model = OTFairBoostClassifier(
        lambda_fairness_value=0.1,
        n_estimators=5,
        num_leaves=4,
        min_child_samples=2,
        n_steps_cdf=32,
        random_state=0,
        n_jobs=1,
    )

    model.fit(X, y, sensitive_attribute=sensitive_attribute)
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert probabilities.shape == (len(y), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert predictions.shape == y.shape
    assert set(np.unique(predictions)).issubset(set(np.unique(y)))


def test_fair_regressor_fit_and_predict(simple_regression_data):
    X, y = simple_regression_data
    sensitive_attribute = X[:, 0] > 0
    model = OTFairBoostRegressor(
        lambda_fairness_value=0.1,
        n_estimators=5,
        num_leaves=4,
        min_child_samples=2,
        n_steps_cdf=32,
        random_state=0,
        n_jobs=1,
    )

    model.fit(X, y, sensitive_attribute=sensitive_attribute)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))


def test_xgb_fair_classifier_fit_predict_and_filter_parameters(simple_dataset):
    X, y = simple_dataset
    sensitive_attribute = X[:, 0] > 0
    model = OTXFairBoostClassifier(
        lambda_fairness_value=0.1,
        n_steps_cdf=32,
        fairness_mode="Demographic_Parity",
        n_estimators=5,
        max_depth=2,
        n_jobs=1,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(X, y, sensitive_attribute=sensitive_attribute)

    assert not any("Parameters:" in str(item.message) for item in caught_warnings)
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert probabilities.shape == (len(y), 2)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert predictions.shape == y.shape
    assert set(np.unique(predictions)).issubset(set(np.unique(y)))


def test_xgb_fair_classifier_rejects_nonzero_min_child_weight(simple_dataset):
    X, y = simple_dataset
    model = OTXFairBoostClassifier(
        lambda_fairness_value=0.1,
        min_child_weight=1.0,
        n_estimators=2,
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="min_child_weight to be 0.0"):
        model.fit(X, y, sensitive_attribute=X[:, 0] > 0)


def test_xgb_fair_regressor_fit_predict_and_filter_parameters(
    simple_regression_data,
):
    X, y = simple_regression_data
    model = OTXFairBoostRegressor(
        lambda_fairness_value=0.1,
        n_steps_cdf=32,
        fairness_mode="Demographic_Parity",
        n_estimators=10,
        max_depth=2,
        reg_lambda=0.0,
        min_child_weight=0.0,
        n_jobs=1,
        verbosity=0,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fitted_model = model.fit(X, y, sensitive_attribute=X[:, 0] > 0)

    assert fitted_model is model
    assert not any("Parameters:" in str(item.message) for item in caught_warnings)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))
    assert np.std(predictions) > 0.0


def test_xgb_fair_regressor_matches_xgb_when_fairness_weight_is_zero(
    simple_regression_data,
):
    X, y = simple_regression_data
    params = {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.1,
        "reg_lambda": 0.0,
        "min_child_weight": 0.0,
        "base_score": 0.5,
        "n_jobs": 1,
        "random_state": 0,
        "verbosity": 0,
    }
    reference = xgb.XGBRegressor(**params).fit(X, y)
    model = OTXFairBoostRegressor(
        lambda_fairness_value=0.0,
        n_steps_cdf=32,
        **params,
    ).fit(X, y, sensitive_attribute=X[:, 0] > 0)

    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-5)
