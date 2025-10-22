import sys
import os

sys.path.append(os.getcwd())

import pytest
from extract_rules.extractors import (
    SirusClassifier,
    SirusRegressor,
)
import numpy as np
from sklearn.datasets import load_iris, make_regression


@pytest.fixture(scope="session")
def random_seed():
    np.random.seed(0)


@pytest.fixture
def simple_dataset(random_seed):
    """Generate a simple synthetic dataset for binary classification."""
    X = np.random.randn(100, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def trained_sirus_on_simple(simple_dataset):
    """Train a small SIRUS model on the simple dataset."""
    X, y = simple_dataset
    model = SirusClassifier(n_estimators=100, p0=0.0, num_rule=5, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture
def iris_dataset():
    """Generate a simple synthetic dataset for binary classification."""
    data = load_iris()
    X = data.data
    y = data.target
    return X, y


@pytest.fixture
def trained_sirusDtree_on_iris(iris_dataset):
    """Train a small SIRUS model on the simple dataset."""
    X, y = iris_dataset
    model = SirusClassifier(
        n_estimators=1, p0=0.1, random_state=0, bootstrap=False, max_features=1.0
    )
    model.fit(X, y)
    return model


@pytest.fixture
def trained_sirus_on_iris(iris_dataset):
    """Train a small SIRUS model on the simple dataset."""
    X, y = iris_dataset
    model = SirusClassifier(n_trees=1000, p0=0.1, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture
def simple_regression_data(random_seed):
    """Generate a simple synthetic dataset for binary classification."""
    X, y = make_regression(n_samples=100, n_features=5, noise=1, random_state=42)
    return X, y


@pytest.fixture
def trained_regression(simple_regression_data):
    """Train a small SIRUS model on the simple dataset."""
    X, y = simple_regression_data
    model = SirusRegressor(n_estimators=100, p0=0.0, num_rule=5, random_state=0)
    model.fit(X, y)
    return model
