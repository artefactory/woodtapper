from extract_rules.utils import get_top_rules
import numpy as np
import pytest


def test_normalize_weights_sum_to_one():
    all_possible_rules_list_str = [
        "[(np.int64(3), np.float64(0.7699999958276749), 'L')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R'), (np.int64(2), np.float64(4.8999998569488525), 'L')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R'), (np.int64(2), np.float64(4.8999998569488525), 'R')]",
        "[(np.int64(3), np.float64(0.7699999958276749), 'R')]",
    ]
    p0 = 0.1
    all_possible_rules_list, all_possible_freq_list = get_top_rules(
        all_possible_rules_list_str, p0
    )
    assert all_possible_rules_list[0] == [
        (np.int64(3), np.float64(0.7699999958276749), "R")
    ]
    assert all(w >= 0 for w in all_possible_freq_list)


def test_normalize_weights_empty_list():
    with pytest.raises(ValueError):
        get_top_rules([], p0=0.1)
    with pytest.raises(ValueError):
        get_top_rules([[]], p0=0.1)
