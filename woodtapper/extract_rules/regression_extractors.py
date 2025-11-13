import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.ensemble._forest import ForestRegressor
from sklearn.ensemble._gb import set_huber_delta, _update_terminal_regions
from sklearn._loss.loss import HuberLoss
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils._param_validation import StrOptions

from .base import RulesExtractorRegressorMixin
from .classification_extractors import QuantileDecisionTreeRegressor


class SirusRegressor(RulesExtractorRegressorMixin, RandomForestRegressor):
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
    max_n_rules : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).

    Attributes
    ----------
    rules_ : list
        List of all possible rules extracted from the forest.
    ridge: ridge regression model fitted on the rules

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
        max_n_rules=25,
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
        self.max_n_rules = max_n_rules
        self.quantile = quantile
        self.to_not_binarize_colindexes = to_not_binarize_colindexes
        self.starting_index_one_hot = starting_index_one_hot  # index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules)


class ExtraTreesRulesRegressor(RulesExtractorRegressorMixin, ExtraTreesRegressor):
    """
    Rules extractor applied with a ExtraTreeRegressor.

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
    max_n_rules : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).

    Attributes
    ----------
    rules_ : list
        List of all possible rules extracted from the forest.
    ridge: ridge regression model fitted on the rules

    """

    _parameter_constraints: dict = {**ExtraTreesRegressor._parameter_constraints}
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
        bootstrap=False,
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
        max_n_rules=25,
        quantile=10,
        to_not_binarize_colindexes=None,
        starting_index_one_hot=None,
    ):
        super(ForestRegressor, self).__init__(
            estimator=ExtraTreeRegressor(),
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
        self.max_n_rules = max_n_rules
        self.quantile = quantile
        self.to_not_binarize_colindexes = to_not_binarize_colindexes
        self.starting_index_one_hot = starting_index_one_hot  # index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules)


class GBRulesRegressor(RulesExtractorRegressorMixin, GradientBoostingRegressor):
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
    max_n_rules : int, default=25
        The maximum number of rules to extract.
    quantile : int, default=10
        The number of quantiles to use for the "quantile" splitter.
    to_not_binarize_colindexes : list of int, default=None
        List of column indexes to not binarize when extracting the rules.
    starting_index_one_hot : int, default=None
        Index of the first one-hot encoded variable in the dataset (to handle correctly the binarization of the rules).
    Attributes
    ----------
    rules_ : list
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
        max_n_rules=25,
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
        self.max_n_rules = max_n_rules
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
