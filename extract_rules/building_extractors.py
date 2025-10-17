import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.ensemble._gb import set_huber_delta, _update_terminal_regions
from sklearn._loss.loss import HuberLoss
from sklearn.utils._param_validation import StrOptions
from sklearn.linear_model import RidgeCV

from .base import SirusMixin
from .extractors import QuantileDecisionTreeRegressor
from .utils import compute_staibility_criterion


class SirusGBClassifierDouble(SirusMixin, GradientBoostingClassifier):
    """
    Class for rules extraction from  a GradientBoostingClassifier
    Parameters
    ----------
    splitter : {"best", "random", "quantile"}, default="quantile"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random
        split. "quantile" is similar to "best" but the split point is chosen to
        be a a value in the training set and not the beetween to values as for best and random.
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
        self._fit_quantile_classifier(X, y, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_[:,0]:  ## extraction  of all trees rules ## [:,0] WORKS only for binary clf (see n_tree_per_iter = 1)
            tree = dtree.tree_
            all_possible_rules_list.extend(self._extract_single_tree_rules(tree))
        self._fit_rules(X, y, all_possible_rules_list, sample_weight)
        compute_staibility_criterion(self)


class SirusGBRegressorDouble(SirusMixin, GradientBoostingRegressor):
    """
    Class for rules extraction from  a GradientBoostingRegressor
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
        self._fit_quantile_classifier(X, y, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_:  ## extraction  of all trees rules
            tree = dtree.tree_
            all_possible_rules_list.extend(self._extract_single_tree_rules(tree))
        self._fit_rules_regressor(X, y, all_possible_rules_list, sample_weight)
        gamma_array = np.zeros((X.shape[0], 2 * self.n_rules))
        for indice in range(self.n_rules):
            current_rules = self.all_possible_rules_list[indice]
            final_mask = self._generate_mask_rule(
                X=X, rules=current_rules
            )  # On X and not on X_bin ???,
            gamma_array[final_mask, indice] = 1
            gamma_array[~final_mask, indice + self.n_rules] = 1  ## NOT the current rule
        self.ridge = RidgeCV(
            alphas=np.arange(0.01, 1, 0.1),
            cv=5,
            scoring="neg_mean_squared_error",
            fit_intercept=True,
        )

        self.ridge.fit(gamma_array, y, sample_weight=sample_weight)
        for indice in range(
            self.n_rules
        ):  ## We weight the probabilities by the coefficients of the ridge
            self.list_probas_by_rules[indice] = (self.ridge.coef_[indice]).tolist()
            self.list_probas_outside_by_rules[indice] = (
                self.ridge.coef_[indice + self.n_rules]
            ).tolist()
        compute_staibility_criterion(self)

    def predict(self, X, to_add_probas_outside_rules=True):
        gamma_array = np.zeros((X.shape[0], 2 * self.n_rules))
        for indice in range(self.n_rules):
            current_rules = self.all_possible_rules_list[indice]
            final_mask = self._generate_mask_rule(
                X=X, rules=current_rules
            )  # On X and not on X_bin
            gamma_array[final_mask, indice] = 1
            gamma_array[~final_mask, indice + self.n_rules] = 1  ## NOT the current rule
        y_pred = self.ridge.predict(
            gamma_array
        )  # Do not sum() or mean() becaus it can be multiclass
        return y_pred
