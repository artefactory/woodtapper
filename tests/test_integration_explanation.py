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
    X_exp1, y_exp1 = model.explanation(X)
    X_exp2, y_exp2 = model.explanation(X)
    assert model.n_estimators == 100
    np.testing.assert_allclose(X_exp1, X_exp2, atol=1e-6)
    np.testing.assert_allclose(y_exp1, y_exp2, atol=1e-6)


def test_explanation_shape(simple_dataset, simple_regression_data):
    X, y = simple_dataset
    model = RandomForestClassifierExplained(n_estimators=50)
    model.fit(X, y)
    X_exp, y_exp = model.explanation(X)
    assert X_exp.shape[0] == X.shape[0]  ## Check number of test samples
    assert X_exp.shape[1] == 5  ## Check number of features in explanation
    assert X_exp.shape[2] == X.shape[1]  ## Check number of features matches input
    assert y_exp.shape[1] == 5
    assert y_exp.shape[0] == X.shape[0]

    model = ExtraTreesClassifierExplained(n_estimators=50)
    model.fit(X, y)
    X_exp, y_exp = model.explanation(X)
    assert X_exp.shape[0] == X.shape[0]  ## Check number of test samples
    assert X_exp.shape[1] == 5  ## Check number of features in explanation
    assert X_exp.shape[2] == X.shape[1]  ## Check number of features matches input
    assert y_exp.shape[1] == 5
    assert y_exp.shape[0] == X.shape[0]

    model = GradientBoostingClassifierExplained(n_estimators=50)
    model.fit(X, y)
    X_exp, y_exp = model.explanation(X)
    assert X_exp.shape[0] == X.shape[0]  ## Check number of test samples
    assert X_exp.shape[1] == 5  ## Check number of features in explanation
    assert X_exp.shape[2] == X.shape[1]  ## Check number of features matches input
    assert y_exp.shape[1] == 5
    assert y_exp.shape[0] == X.shape[0]

    X_reg, y_reg = simple_regression_data
    model = RandomForestRegressorExplained(n_estimators=50)
    model.fit(X_reg, y_reg)
    X_exp, y_exp = model.explanation(X_reg)
    print(X_exp.shape, y_exp.shape)
    assert X_exp.shape[0] == X_reg.shape[0]  ## Check number of test samples
    assert X_exp.shape[1] == 5  ## Check number of features in explanation
    assert X_exp.shape[2] == X_reg.shape[1]  ## Check number of features matches input
    assert y_exp.shape[1] == 5
    assert y_exp.shape[0] == X_reg.shape[0]

    model = ExtraTreesRegressorExplained(n_estimators=50)
    model.fit(X_reg, y_reg)
    X_exp, y_exp = model.explanation(X_reg)
    print(X_exp.shape, y_exp.shape)
    assert X_exp.shape[0] == X_reg.shape[0]  ## Check number of test samples
    assert X_exp.shape[1] == 5  ## Check number of features in explanation
    assert X_exp.shape[2] == X_reg.shape[1]  ## Check number of features matches input
    assert y_exp.shape[1] == 5
    assert y_exp.shape[0] == X_reg.shape[0]

    model = GradientBoostingRegressorExplained(n_estimators=50)
    model.fit(X_reg, y_reg)
    X_exp, y_exp = model.explanation(X_reg)
    print(X_exp.shape, y_exp.shape)
    assert X_exp.shape[0] == X_reg.shape[0]  ## Check number of test samples
    assert X_exp.shape[1] == 5  ## Check number of features in explanation
    assert X_exp.shape[2] == X_reg.shape[1]  ## Check number of features matches input
    assert y_exp.shape[1] == 5
    assert y_exp.shape[0] == X_reg.shape[0]
