import io
import sys

import numpy as np
from woodtapper.extract_rules import (
    SirusClassifier,
    SirusRegressor,
    ExtraTreesRulesClassifier,
    ExtraTreesRulesRegressor,
    GBRulesClassifier,
    GBRulesRegressor,
)
from woodtapper.extract_rules.visualization import show_rules


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

    model = ExtraTreesRulesClassifier(n_estimators=100, p0=0.0, max_n_rules=5)
    model.fit(X, y)
    assert len(model.estimators_) > 0
    assert model.max_n_rules > 0

    model_regressor = ExtraTreesRulesRegressor(n_estimators=100, p0=0.0, max_n_rules=5)
    model_regressor.fit(X_reg, y_reg)
    assert len(model_regressor.estimators_) > 0
    assert model_regressor.max_n_rules > 0

    model_gb_clf = GBRulesClassifier(n_estimators=100, p0=0.0, max_n_rules=5)
    model_gb_clf.fit(X, y)
    assert len(model_gb_clf.estimators_) > 0
    assert model_gb_clf.max_n_rules > 0

    model_gb_regressor = GBRulesRegressor(n_estimators=100, p0=0.0, max_n_rules=5)
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


def test_rules_vizualization(simple_dataset, simple_regression_data):
    X, y = simple_dataset
    X_reg, y_reg = simple_regression_data

    model = SirusClassifier(n_estimators=100, p0=0.0, max_n_rules=5, random_state=0)
    model.fit(X, y)  # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output  # Call the function
    show_rules(
        model, max_rules=3, is_regression=False, target_class_index=1
    )  # Reset stdout
    sys.stdout = sys.__stdout__  # Check the result
    output = captured_output.getvalue().strip()
    print(output)
    expected_output = """Estimated average rate for target class 1 (from 'else' clauses) p_s = 66%.
(Note: True average rate should be P(Class=1) from training data).

   Condition              THEN P(C1)      ELSE P(C1)
---------------------------------------------------------
if   Feature[0] <= 0.42   then 20%                else 90%
if   Feature[1] > -0.37   then 63%                else 13%
if   Feature[1] <= 0.97   then 38%                else 90%"""

    def normalize(s):
        return "\n".join([line.rstrip() for line in s.splitlines()])

    assert normalize(output) == normalize(expected_output), (
        f"Unexpected output: {output}"
    )

    model_regressor = SirusRegressor(
        n_estimators=100, p0=0.0, max_n_rules=5, random_state=0
    )
    model_regressor.fit(X_reg, y_reg)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    show_rules(model_regressor, max_rules=3, is_regression=True)
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue().strip()
    expected_output = """---------------------------------------------------------
Intercept : -6.896237002187981
if   Feature[1] <= 0.53   then -42.01             else 127.47 | coeff=0.03
if   Feature[1] <= 0.34   then -55.79             else 105.77 | coeff=0.35
if   Feature[1] > -0.51   then 57.46              else -104.64 | coeff=0.63"""
    assert normalize(output) == normalize(expected_output), (
        f"Unexpected output: {output}"
    )
