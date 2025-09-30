import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.ensemble._forest import ForestClassifier,ForestRegressor
from sklearn.ensemble._gb import set_huber_delta, _update_terminal_regions
from sklearn._loss.loss import HuberLoss
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import StrOptions

from pysirus.models.basic import SirusMixin

class SirusDTreeClassifier(SirusMixin, DecisionTreeClassifier):
    """
    SIRUS class applied with a DecisionTreeClassifier
    Parameters
    ----------

    """

    _parameter_constraints: dict = {**DecisionTreeClassifier._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = self.extract_single_tree_rules(self.tree_)
        self.fit_forest_rules(
            X, y, all_possible_rules_list, p0, sample_weight
        )  ## Checker que cx'est bien sur X et non le X_bin
        return self


class SirusRFClassifier(SirusMixin, RandomForestClassifier):  # DecisionTreeClassifier
    """
    SIRUS class applied with a RandomForestClassifier

    """

    _parameter_constraints: dict = {**RandomForestClassifier._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
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

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True):
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_:  ## extraction  of all trees rules
            tree = dtree.tree_
            all_possible_rules_list.extend(self.extract_single_tree_rules(tree))
        self.fit_forest_rules(X, y, all_possible_rules_list, p0,sample_weight)


######### Regressor ############


class SirusDTreeRegressor(SirusMixin, DecisionTreeRegressor):
    """
    SIRUS class applied with a DecisionTreeClassifier
    Parameters
    ----------

    """

    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def fit_forest_rules_regressor( 
        self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True
    ): 
        """Build a decision tree classifier from the training set (X, y)."""
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = self.extract_single_tree_rules(self.tree_)
        self.fit_forest_rules(
            X, y, all_possible_rules_list, p0
        )  ## Checker que cx'est bien sur X et non le X_bin
        return self

    def predict(self, X, to_add_probas_outside_rules=True):
        return self.predict_regressor(X, to_add_probas_outside_rules)
    
class DecisionTreeRegressor2(SirusMixin, DecisionTreeRegressor):
    """
    DecisionTreeRegressor of scikit -learn with the "quantile" spliiter option.
    Used for GradientBoostingClassifier in SirusGBClassifier
    """

    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]



class SirusGBClassifier(SirusMixin, GradientBoostingClassifier):
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
            tree = DecisionTreeRegressor2(
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

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True):
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = []
        for i in range(self.n_estimators_):  ## extraction  of all trees rules
            #print('self.estimators_.shape', self.estimators_.shape)
            dtree = self.estimators_[i,0]  
            tree = dtree.tree_
            all_possible_rules_list.extend(self.extract_single_tree_rules(tree))
        #self.fit_forest_rules_regressor(X, y, all_possible_rules_list, p0,batch_size_post_treatment)
        self.fit_forest_rules(X, y, all_possible_rules_list, p0,sample_weight)


class SirusRFRegressor(SirusMixin, RandomForestRegressor):
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

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True):
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = []
        for i in range(self.n_outputs_):  ## extraction  of all trees rules
            dtree = self.estimators_[i]  
            tree = dtree.tree_
            all_possible_rules_list.extend(self.extract_single_tree_rules(tree))
        self.fit_forest_rules_regressor(X, y, all_possible_rules_list, p0,sample_weight)