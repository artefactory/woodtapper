from woodtapper.extract_rules.utils import get_top_rules, ridge_cv_positive
import numpy as np
import pytest


def test_normalize_weights_sum_to_one():
    rules_str = [
        "[(np.int64(3), np.float64(0.7699999958276749), 'L')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R'), (np.int64(2), np.float64(4.8999998569488525), 'L')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R'), (np.int64(2), np.float64(4.8999998569488525), 'R')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R')]",
    ]
    p0 = 0.1
    rules_, rules_freq_ = get_top_rules(rules_str, p0)
    assert rules_[0] == [(np.int64(3), np.float64(0.7699999958276749), "R")]
    assert all(w >= 0 for w in rules_freq_)


def test_normalize_weights_empty_list():
    with pytest.raises(ValueError):
        get_top_rules([], p0=0.1)
    with pytest.raises(ValueError):
        get_top_rules([[]], p0=0.1)


def test_ridge_cv(simple_regression_data):
    X_reg, y_reg = simple_regression_data
    best_alpha, results = ridge_cv_positive(
        X_reg,
        y_reg,
        random_state=0,
    )
    assert isinstance(best_alpha, float)
