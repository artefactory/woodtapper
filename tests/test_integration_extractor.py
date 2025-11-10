import pytest
import numpy as np
from woodtapper.extract_rules import SirusClassifier


def test_training_and_prediction_consistency(simple_dataset):
    X, y = simple_dataset
    model = SirusClassifier(n_estimators=100, max_n_rules=5, p0=0.0)
    model.fit(X, y)
    preds1 = model.predict(X)
    preds2 = model.predict(X)
    np.testing.assert_allclose(preds1, preds2, atol=1e-6)


def test_rules_extraction_stability(trained_sirus_on_simple):
    rules = trained_sirus_on_simple.rules_
    assert isinstance(rules, list)
    assert len(rules) <= trained_sirus_on_simple.max_n_rules


@pytest.mark.parametrize("n_trees,max_rules", [(50, 3), (100, 5), (300, 10)])
def test_sirus_scaling(simple_dataset, n_trees, max_rules):
    X, y = simple_dataset
    model = SirusClassifier(n_estimators=n_trees, max_n_rules=max_rules)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
