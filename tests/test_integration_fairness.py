import numpy as np

from woodtapper.fairness import OTFairBoostClassifier, OTFairBoostRegressor


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
