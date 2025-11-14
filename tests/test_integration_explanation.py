import numpy as np
from woodtapper.example_sampling import (
    RandomForestClassifierExplained,
    ExtraTreesClassifierExplained,
    GradientBoostingClassifierExplained,
    RandomForestRegressorExplained,
    ExtraTreesRegressorExplained,
    GradientBoostingRegressorExplained,
)


def test_random_forest_classifier_explained(simple_dataset):
    X, y = simple_dataset
    model = RandomForestClassifierExplained(n_estimators=100)
    model.fit(X, y)
    Xy_exp1 = model.explanation(X)
    Xy_exp2 = model.explanation(X)
    assert model.n_estimators == 100
    np.testing.assert_allclose(
        Xy_exp1[0][0], Xy_exp2[0][0], atol=1e-6
    )  # Check explained samples
    np.testing.assert_allclose(
        Xy_exp1[0][1], Xy_exp2[0][1], atol=1e-6
    )  # Check explained targets


def test_explanation_shape(simple_dataset, simple_regression_data):
    X, y = simple_dataset
    model = RandomForestClassifierExplained(n_estimators=50)
    model.fit(X, y)
    Xy_exp = model.explanation(X)
    assert len(Xy_exp) == X.shape[0]  ## Check number of test samples
    assert (
        Xy_exp[0][0].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
    assert (
        Xy_exp[0][0].shape[1] == X.shape[1]
    )  ## Check number of features matches input
    assert (
        Xy_exp[0][1].shape[0] == 5
    )  ## Check number of explained sample is 5 by default

    model = ExtraTreesClassifierExplained(n_estimators=50)
    model.fit(X, y)
    Xy_exp = model.explanation(X)
    assert len(Xy_exp) == X.shape[0]  ## Check number of test samples
    assert (
        Xy_exp[0][0].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
    assert (
        Xy_exp[0][0].shape[1] == X.shape[1]
    )  ## Check number of features matches input
    assert (
        Xy_exp[0][1].shape[0] == 5
    )  ## Check number of explained sample is 5 by default

    model = GradientBoostingClassifierExplained(n_estimators=50)
    model.fit(X, y)
    Xy_exp = model.explanation(X)
    assert len(Xy_exp) == X.shape[0]  ## Check number of test samples
    assert (
        Xy_exp[0][0].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
    assert (
        Xy_exp[0][0].shape[1] == X.shape[1]
    )  ## Check number of features matches input
    assert (
        Xy_exp[0][1].shape[0] == 5
    )  ## Check number of explained sample is 5 by default

    X_reg, y_reg = simple_regression_data
    model = RandomForestRegressorExplained(n_estimators=50)
    model.fit(X_reg, y_reg)
    Xy_exp = model.explanation(X_reg)
    assert len(Xy_exp) == X_reg.shape[0]  ## Check number of test samples
    assert (
        Xy_exp[0][0].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
    assert (
        Xy_exp[0][0].shape[1] == X_reg.shape[1]
    )  ## Check number of features matches input
    assert (
        Xy_exp[0][1].shape[0] == 5
    )  ## Check number of explained sample is 5 by default

    model = ExtraTreesRegressorExplained(n_estimators=50)
    model.fit(X_reg, y_reg)
    Xy_exp = model.explanation(X_reg)
    assert len(Xy_exp) == X_reg.shape[0]  ## Check number of test samples
    assert (
        Xy_exp[0][0].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
    assert (
        Xy_exp[0][0].shape[1] == X_reg.shape[1]
    )  ## Check number of features matches input
    assert (
        Xy_exp[0][1].shape[0] == 5
    )  ## Check number of explained sample is 5 by default

    model = GradientBoostingRegressorExplained(n_estimators=50)
    model.fit(X_reg, y_reg)
    Xy_exp = model.explanation(X_reg)
    assert len(Xy_exp) == X_reg.shape[0]  ## Check number of test samples
    assert (
        Xy_exp[0][0].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
    assert (
        Xy_exp[0][0].shape[1] == X_reg.shape[1]
    )  ## Check number of features matches input
    assert (
        Xy_exp[0][1].shape[0] == 5
    )  ## Check number of explained sample is 5 by default
