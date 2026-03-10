import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.abspath(Path(__file__).parents[2]))
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from woodtapper.extract_rules import SirusClassifier


def benchmark_model(model: BaseEstimator, X_train, y_train, X_test):
    """
    Benchmark both fit() and predict() steps of a scikit-learn model:
    measures elapsed time, absolute peak memory, and incremental memory usage.

    Parameters
    ----------
    model : sklearn estimator
        The model instance (e.g., LogisticRegression()).
    X_train : array-like or sparse matrix
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like or sparse matrix
        Test features for prediction.

    Returns
    -------
    fitted_model : sklearn estimator
        The trained model.
    metrics : dict
        Dictionary containing timing and memory usage for both fit and predict:
        {
            "fit_time": float,
        }
    """

    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    metrics = {
        "fit_time": fit_time,
    }

    return metrics


X_sim, y_sim = make_classification(
    n_samples=625000,
    n_features=200,
    n_informative=196,
    n_redundant=2,
    n_repeated=2,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    random_state=42,
)
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X_sim, y_sim, test_size=0.2, random_state=42
)
y_train_original = y_train_original.astype(int)
y_test_original = y_test_original.astype(int)

output_dir_data_set = "paper/reproduce-exp/sim-data-time/"  #
os.makedirs(output_dir_data_set, exist_ok=True)
np.savetxt(f"{output_dir_data_set}/X_train.csv", X_train_original, delimiter=",")
np.savetxt(f"{output_dir_data_set}/y_train.csv", y_train_original, delimiter=",")
np.savetxt(f"{output_dir_data_set}/X_test.csv", X_test_original, delimiter=",")
np.savetxt(f"{output_dir_data_set}/y_test.csv", y_test_original, delimiter=",")
# We save the data in .csv so that R and Julia version can access to the exact same data points.

output_dir_Rules = "paper/timing/times-csv/"
os.makedirs(output_dir_Rules, exist_ok=True)

n_samples = [100000, 200000, 300000, 400000, 500000]
n_dim = 200
list_time_samples = []
for n_samples_train in n_samples:
    X_train = X_train_original[:n_samples_train, :n_dim]
    y_train = y_train_original[:n_samples_train]

    X_test = X_test_original[:, :n_dim]
    y_test = y_test_original
    curr_list_time = []
    for run in range(5):
        print("********* RUN SAMPLES ", run + 1, " *********")
        RFSirus = SirusClassifier(
            n_estimators=1000,
            max_depth=2,
            max_features=14,
            quantile=10,
            p0=0.0,
            max_n_rules=25,
            to_not_binarize_colindexes=None,
            starting_index_one_hot=None,
            random_state=0,
            splitter="quantile",
            n_jobs=5,
        )

        print(
            "############################## n_samples_train = ",
            n_samples_train,
            "and n_dim=",
            n_dim,
            " ##############################",
        )
        metrics = benchmark_model(RFSirus, X_train, y_train, X_test)
        curr_list_time.append(metrics["fit_time"])
        print("=== Benchmark Results ===")
        print(f"Fit time: {metrics['fit_time']:.4f} s")
        print("=========================")
    print("\n \n")
    list_time_samples.append(curr_list_time)
np.save(
    os.path.join(output_dir_Rules, "list_time_samples_python.npy"),
    np.array(list_time_samples),
)


n_samples_train = 300000
n_dims = [15, 25, 50, 75, 100, 125, 150, 175, 200]
list_time_dims = []
for n_dim in n_dims:
    X_train = X_train_original[:n_samples_train, :n_dim]
    y_train = y_train_original[:n_samples_train]

    X_test = X_test_original[:, :n_dim]
    y_test = y_test_original
    curr_list_time = []
    for run in range(5):
        print("********* RUN DIMENSION ", run + 1, " *********")
        RFSirus = SirusClassifier(
            n_estimators=1000,
            max_depth=2,
            max_features=14,
            quantile=10,
            p0=0.01,
            to_not_binarize_colindexes=None,
            starting_index_one_hot=None,
            random_state=0,
            splitter="quantile",
            n_jobs=5,
        )

        print(
            "############################## n_samples_train = ",
            n_samples_train,
            "and n_dim=",
            n_dim,
            " ##############################",
        )
        metrics = benchmark_model(RFSirus, X_train, y_train, X_test)
        curr_list_time.append(metrics["fit_time"])
        print("=== Benchmark Results ===")
        print(f"Fit time: {metrics['fit_time']:.4f} s")
        print("=========================")
    print("\n \n")
    list_time_dims.append(curr_list_time)
np.save(
    os.path.join(output_dir_Rules, "list_time_dims_python.npy"),
    np.array(list_time_dims),
)
