import numpy as np
from woodtapper.extract_rules import (
    SirusClassifier,
    SirusRegressor,
    GbExtractorClassifier,
    GbExtractorRegressor,
)


def test_sirus_fit(simple_dataset, simple_regression_data):
    X, y = simple_dataset
    X_reg, y_reg = simple_regression_data

    model = SirusClassifier(n_estimators=100, p0=0.0, max_n_rules=5)
    model.fit(X, y)
    assert len(model.estimators_) > 0
    assert model.max_n_rules > 0

    model_regressor = SirusRegressor(n_estimators=100, p0=0.0, max_n_rules=5)
    model_regressor.fit(X_reg, y_reg)
    assert len(model_regressor.estimators_) > 0
    assert model_regressor.max_n_rules > 0

    model_gb_clf = GbExtractorClassifier(n_estimators=100, p0=0.0, max_n_rules=5)
    model_gb_clf.fit(X, y)
    assert len(model_gb_clf.estimators_) > 0
    assert model_gb_clf.max_n_rules > 0

    model_gb_regressor = GbExtractorRegressor(n_estimators=100, p0=0.0, max_n_rules=5)
    model_gb_regressor.fit(X_reg, y_reg)
    assert len(model_gb_regressor.estimators_) > 0
    assert model_gb_regressor.max_n_rules > 0


def test_sirus_fit_sets_attributes(simple_dataset):
    X, y = simple_dataset
    model = SirusClassifier(n_estimators=50, max_n_rules=3)
    model.fit(X, y)
    assert hasattr(model, "rules_"), "SIRUS should store extracted rules"
    assert isinstance(model.rules_, list)


def test_predict_output_shape(trained_sirus_on_simple, simple_dataset):
    X, _ = simple_dataset
    preds = trained_sirus_on_simple.predict(X)
    assert preds.shape[0] == X.shape[0], (
        "Number of predictions should match number of samples"
    )


def test_predict_value_range(trained_sirus_on_simple, simple_dataset):
    X, _ = simple_dataset
    preds = trained_sirus_on_simple.predict(X)
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be probabilities"
