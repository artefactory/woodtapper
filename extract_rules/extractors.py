import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.ensemble._gb import set_huber_delta, _update_terminal_regions
from sklearn._loss.loss import HuberLoss
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import StrOptions
import time

from .base import RulesExtractorMixin
from .utils import compute_staibility_criterion


class SirusClassifier(RulesExtractorMixin, RandomForestClassifier):
    """
    SIRUS class applied with a RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, "entropy" for the information gain and
        "log_loss" for the reduction in log loss.
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
    class_weight : {"balanced", "balanced_subsample"}, default=None
        Weights associated with classes in the form {class_label: weight}.
        If None, all classes are supposed to have weight one.
    splitter : {"best", "random", "quantile"}, default="quantile"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random
        split. "quantile" is similar to "best" but the split point is chosen to
        be a a value in the training set and not the beetween to values as for best and random.
    p0 : float, default=0.01
        The threshold for rule selection.
    num_rule : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).
    Attributes
    ----------
    all_possible_rules_list : list
        List of all possible rules extracted from the forest.


    """

    _parameter_constraints: dict = {**RandomForestClassifier._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        splitter="quantile",
        p0=0.01,
        num_rule=25,
        quantile=10,
        to_not_binarize_colindexes=None,
        starting_index_one_hot=None,
    ):
        super(ForestClassifier, self).__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
                "splitter",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha
        self.splitter = splitter
        self.p0 = p0
        self.num_rule = num_rule
        self.quantile = quantile
        self.to_not_binarize_colindexes = to_not_binarize_colindexes
        self.starting_index_one_hot = starting_index_one_hot  # index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the SIRUS model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : object
            Fitted estimator.

        """
        start = time.time()
        self._fit_quantile_classifier(X, y, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_:  ## extraction  of all trees rules
            tree = dtree.tree_
            all_possible_rules_list.extend(self._extract_single_tree_rules(tree))
        self._fit_rules(X, y, all_possible_rules_list, sample_weight)
        end = time.time()
        print(f"All fit took {end - start:.4f} seconds")
        compute_staibility_criterion(self)


class QuantileDecisionTreeRegressor(RulesExtractorMixin, DecisionTreeRegressor):
    """
    DecisionTreeRegressor of scikit -learn with the "quantile" spliiter option.
    Used for GradientBoostingClassifier in GbExtractorClassifier
    """

    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]


class GbExtractorClassifier(RulesExtractorMixin, GradientBoostingClassifier):
    """
    Class for rules extraction from  a GradientBoostingClassifier
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by
        `learning_rate`. There is a trade-off between learning_rate and
    loss : {'log_loss', 'deviance'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to
        logistic regression for classification with probabilistic outputs.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, "entropy" for the information gain and
        "log_loss" for the reduction in log loss.
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
    splitter : {"best", "random", "quantile"}, default="quantile"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random
        split. "quantile" is similar to "best" but the split point is chosen to
        be a a value in the training set and not the beetween to values as for best and random.
    p0 : float, default=0.01
        The threshold for rule selection.
    num_rule : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).
    Attributes
    ----------
    all_possible_rules_list : list
        List of all possible rules extracted from the forest.
    """

    _parameter_constraints: dict = {**GradientBoostingClassifier._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        splitter="quantile",
        p0=0.01,
        num_rule=25,
        quantile=10,
        to_not_binarize_colindexes=None,
        starting_index_one_hot=None,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
        self.splitter = splitter
        self.p0 = p0
        self.num_rule = num_rule
        self.quantile = quantile
        self.to_not_binarize_colindexes = to_not_binarize_colindexes
        self.starting_index_one_hot = starting_index_one_hot  # index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules)

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of ``n_trees_per_iteration_`` trees."""
        original_y = y

        if isinstance(self._loss, HuberLoss):
            set_huber_delta(
                loss=self._loss,
                y_true=y,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight,
            )
        # TODO: Without oob, i.e. with self.subsample = 1.0, we could call
        # self._loss.loss_gradient and use it to set train_score_.
        # But note that train_score_[i] is the score AFTER fitting the i-th tree.
        # Note: We need the negative gradient!
        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
            sample_weight=None,  # We pass sample_weights to the tree directly.
        )
        # 2-d views of shape (n_samples, n_trees_per_iteration_) or (n_samples, 1)
        # on neg_gradient to simplify the loop over n_trees_per_iteration_.
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                y = np.array(original_y == k, dtype=np.float64)

            # induce regression tree on the negative gradient
            tree = QuantileDecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,  ## ici
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha,
            )

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csc if X_csc is not None else X
            tree.fit(
                X, neg_g_view[:, k], sample_weight=sample_weight, check_input=False
            )

            # update tree leaves
            X_for_tree_update = X_csr if X_csr is not None else X
            _update_terminal_regions(
                self._loss,
                tree.tree_,
                X_for_tree_update,
                y,
                neg_g_view[:, k],
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the SIRUS model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : object
            Fitted estimator.

        """
        self._fit_quantile_classifier(X, y, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_[
            :, 0
        ]:  ## extraction  of all trees rules ## [:,0] WORKS only for binary clf (see n_tree_per_iter = 1)
            tree = dtree.tree_
            curr_tree_rules = self._extract_single_tree_rules(tree)
            if (
                len(curr_tree_rules) > 0 and len(curr_tree_rules[0]) > 0
            ):  # to avoid empty rules
                # Boosting may produce trees with no splits, for example when the number of estimators is high
                all_possible_rules_list.extend(curr_tree_rules)
        self._fit_rules(X, y, all_possible_rules_list, sample_weight)
        compute_staibility_criterion(self)


######### Regressor ############


class SirusRegressor(RulesExtractorMixin, RandomForestRegressor):
    """
    SIRUS class applied with a RandomForestRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, "entropy" for the information gain and
        "log_loss" for the reduction in log loss.
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
    splitter : {"best", "random", "quantile"}, default="quantile"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random
        split. "quantile" is similar to "best" but the split point is chosen to
        be a a value in the training set and not the beetween to values as for best and random.
    p0 : float, default=0.01
        The threshold for rule selection.
    num_rule : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).
    ridge: ridge regression model fitted on the rules

    Attributes
    ----------
    all_possible_rules_list : list
        List of all possible rules extracted from the forest.

    """

    _parameter_constraints: dict = {**RandomForestRegressor._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        splitter="quantile",
        p0=0.01,
        num_rule=25,
        quantile=10,
        to_not_binarize_colindexes=None,
        starting_index_one_hot=None,
    ):
        super(ForestRegressor, self).__init__(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
                "splitter",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.splitter = splitter
        self.p0 = p0
        self.num_rule = num_rule
        self.quantile = quantile
        self.to_not_binarize_colindexes = to_not_binarize_colindexes
        self.starting_index_one_hot = starting_index_one_hot  # index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the SIRUS model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : object
            Fitted estimator.

        """
        if isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame)):
            self.feature_names_in_ = X.columns.to_numpy()
            X = X.values
            if isinstance(y, (pd.core.series.Series, pd.core.frame.DataFrame)):
                y = y.values
            elif len(y.shape) > 1:
                y = y.ravel()
        elif not isinstance(X, np.ndarray):
            raise Exception(
                "Wrong type for X. except numpy array, pandas dataframe or series"
            )
        self._fit_quantile_classifier(X, y, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_:  ## extraction  of all trees rules
            tree = dtree.tree_
            all_possible_rules_list.extend(self._extract_single_tree_rules(tree))
        self._fit_rules_regressor(X, y, all_possible_rules_list, sample_weight)
        compute_staibility_criterion(self)

    def predict(self, X, to_add_probas_outside_rules=True):
        if isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame)):
            self.feature_names_in_ = X.columns.to_numpy()
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise Exception("Wrong type for X")
        return self._predict_regressor(X, to_add_probas_outside_rules)


class GbExtractorRegressor(RulesExtractorMixin, GradientBoostingRegressor):
    """
    Class for rules extraction from a GradientBoostingRegressor
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by
        `learning_rate`. There is a trade-off between learning_rate and
    loss : {'log_loss', 'deviance'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to
        logistic regression for classification with probabilistic outputs.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, "entropy" for the information gain and
        "log_loss" for the reduction in log loss.
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
    splitter : {"best", "random", "quantile"}, default="quantile"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random
        split. "quantile" is similar to "best" but the split point is chosen to
        be a a value in the training set and not the beetween to values as for best and random.
    p0 : float, default=0.01
        The threshold for rule selection.
    num_rule : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).
    Attributes
    ----------
    all_possible_rules_list : list
        List of all possible rules extracted from the forest.
    ridge: ridge regression model fitted on the rules

    """

    _parameter_constraints: dict = {**GradientBoostingRegressor._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        splitter="quantile",
        p0=0.01,
        num_rule=25,
        quantile=10,
        to_not_binarize_colindexes=None,
        starting_index_one_hot=None,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
        self.splitter = splitter
        self.p0 = p0
        self.num_rule = num_rule
        self.quantile = quantile
        self.to_not_binarize_colindexes = to_not_binarize_colindexes
        self.starting_index_one_hot = starting_index_one_hot  # index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules)

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of ``n_trees_per_iteration_`` trees."""
        original_y = y

        if isinstance(self._loss, HuberLoss):
            set_huber_delta(
                loss=self._loss,
                y_true=y,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight,
            )
        # TODO: Without oob, i.e. with self.subsample = 1.0, we could call
        # self._loss.loss_gradient and use it to set train_score_.
        # But note that train_score_[i] is the score AFTER fitting the i-th tree.
        # Note: We need the negative gradient!
        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
            sample_weight=None,  # We pass sample_weights to the tree directly.
        )
        # 2-d views of shape (n_samples, n_trees_per_iteration_) or (n_samples, 1)
        # on neg_gradient to simplify the loop over n_trees_per_iteration_.
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                y = np.array(original_y == k, dtype=np.float64)

            # induce regression tree on the negative gradient
            tree = QuantileDecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,  ## ici
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha,
            )

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csc if X_csc is not None else X
            tree.fit(
                X, neg_g_view[:, k], sample_weight=sample_weight, check_input=False
            )

            # update tree leaves
            X_for_tree_update = X_csr if X_csr is not None else X
            _update_terminal_regions(
                self._loss,
                tree.tree_,
                X_for_tree_update,
                y,
                neg_g_view[:, k],
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the RulesExtractor model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : object
            Fitted estimator.

        """
        if isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame)):
            self.feature_names_in_ = X.columns.to_numpy()
            X = X.values
            y = y.values
        elif not isinstance(X, np.ndarray):
            raise Exception("Wrong type for X")
        self._fit_quantile_classifier(X, y, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_[:, 0]:  ## extraction  of all trees rules
            tree = dtree.tree_
            curr_tree_rules = self._extract_single_tree_rules(tree)
            if (
                len(curr_tree_rules) > 0 and len(curr_tree_rules[0]) > 0
            ):  # to avoid empty rules
                # Boosting may produce trees with no splits, for example when the number of estimators is high
                all_possible_rules_list.extend(curr_tree_rules)
        self._fit_rules_regressor(X, y, all_possible_rules_list, sample_weight)
        compute_staibility_criterion(self)

    def predict(self, X, to_add_probas_outside_rules=True):
        """
        Predict using the RulesExtractorMixin regressor.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        to_add_probas_outside_rules : bool, default=True
            Whether to add the predictions from outside the rules.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.

        """
        y_pred = self._predict_regressor(X, to_add_probas_outside_rules)
        return y_pred
