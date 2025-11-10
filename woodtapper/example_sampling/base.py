"""
ExampleExplanation mixin for tree-based models.
"""

import copy
import numpy as np

from .utils.utils import compute_leaf_sizes
from .utils.weights import compute_kernel_weights


class ExplanationMixin:
    """
    Mixin for ExplanationExample indicator for tree-based models.
    The explanation method show the 5 most similar samples based on
    the frequency of training samples ending in the same leaf as the new sample.
    """

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values (class labels in classification, real numbers in regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        super().fit(X=X, y=y, sample_weight=sample_weight)
        self.train_y = y
        self.train_X = X
        self.train_samples_leaves = (
            super().apply(X).astype(np.int32)
        )  # train_samples_leaves: size n_train x n_trees

    def load_forest(cls, model, X, y):
        """
        Loads a pre-fitted forest from scikit-learn into a Explanation class.
        Parameters
        ----------
        model: scikit-learn model of forest, previously fitted on X, y
            Needs to be of the corresponding skclass class (e.g RandomForestClassifier, GradientBoostingRegressor)
        X : array-like of shape (n_samples, n_features)
            Training data used for the fitting of model.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values used for the fitting of model.
        Returns
        -------
        A instance of the current class, with a deep copy of pre-fitted model, and saved X, y for examples sampling.
        """
        is_model_right_sklearn_class = False
        for parent_class in cls.__bases__:
            is_model_right_sklearn_class += isinstance(model, parent_class)

        assert is_model_right_sklearn_class, (
            "Needs to load a model of same class. {} not found in: {}".format(
                type(model), cls.__bases__
            )
        )

        explanation_model = cls()
        vars(explanation_model).update(copy.deepcopy(vars(model)))
        explanation_model.train_y = y
        explanation_model.train_samples_leaves = model.apply(X).astype(
            np.int32
        )  # train_samples_leaves: size n_train x n_trees

        return explanation_model

    def get_weights(self, X):
        """
        Derive frequency of training samples ending in the same leaf as the new sample X.
        (see GRF algorithm for details)
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples for which to compute the weights.
        Returns
        -------
        np.ndarray of shape (n_samples, n_train)
            Weights for each sample in X based on the training samples leaves.
            Each element is the frequency of the training sample's leaf in the new sample.
        """
        leafs_by_sample = (
            super().apply(X).astype(np.int16)
        )  # taille n_samples x n_trees
        leaves_match = np.array(
            [leafs_by_sample[i] == self.train_samples_leaves for i in range(len(X))]
        )
        n_by_tree = leaves_match.sum(axis=1)[:, np.newaxis, :]

        # shape of output: n_samples x n_train
        return (leaves_match / n_by_tree).mean(axis=2)

    def get_weights_cython(self, X):
        """
        Derive frequency of training samples ending in the same leaf as the new sample X.
        (see GRF algorithm for details)
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples for which to compute the weights.
        Returns
        -------
        np.ndarray of shape (n_samples, n_train)
            Weights for each sample in X based on the training samples leaves.
            Each element is the frequency of the training sample's leaf in the new sample.
        """
        leafs_by_sample = (
            super().apply(X).astype(np.int32)
        )  # taille n_samples x n_trees
        leaf_sizes = compute_leaf_sizes(self.train_samples_leaves)
        return compute_kernel_weights(
            leafs_by_sample, self.train_samples_leaves, leaf_sizes
        )

    def explanation(self, X, batch_size=None):
        """
        Explanation procedure.
        Show the 5 most similar samples based on the frequency of training samples ending in the same leaf as the new sample
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples for which to predict the target values.
        batch_size : int, optional
            Size of the batch to process at once. If None, the entire dataset is processed at once.
        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for each sample in X.
            If the model is a classifier, the output will be class labels.
            If the model is a regressor, the output will be real numbers.
        """
        if batch_size is None:
            weights = self.get_weights_cython(X)
        else:
            list_weights = []
            for batch in np.array_split(X, len(X) // batch_size):
                list_weights.extend(self.get_weights_cython(batch))
            weights = np.array(list_weights)  # n_samples x n_train
        most_similar_idx = np.argsort(-weights, axis=1)[:, :5]
        # Get the 5 most similar samples
        return self.train_X[most_similar_idx], self.train_y[most_similar_idx]
