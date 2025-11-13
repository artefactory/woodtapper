import numpy as np
from sklearn.tree import _splitter
import sklearn.tree._classes
from sklearn.linear_model import Ridge
from sklearn.utils.validation import validate_data

from .Splitter.QuantileSplitter import QuantileBestSplitter
from .utils import (
    compute_stability_criterion,
    get_top_rules,
    ridge_cv_positive,
    generate_masks_rules,
    _extract_single_tree_rules,
    _rules_filtering_stochastic,
)

sklearn.tree._classes.DENSE_SPLITTERS = {
    "best": _splitter.BestSplitter,
    "random": _splitter.BestSplitter,
    "quantile": QuantileBestSplitter,
}


class RulesExtractorMixin:
    """
    Mixin for rules extraction. Base of all extractors models.

    Attributes
    ----------
    p0 : float, optional (default=0.01)
        Frequency threshold for rule selection.
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    n_jobs : int, optional (default=1)
        Number of parallel jobs for tree construction.
    n_features_in_ : int
        Number of features in the input data.
    n_classes_ : int
        Number of classes in the target variable (for classification tasks).
    classes_ : array-like
        Unique classes in the target variable (for classification tasks).
    n_rules : int
        Number of rules extracted from the ensemble.
    rules_ : list
        List of all possible rules extracted from the ensemble.
    rules_freq_ : list
        List of frequencies associated with each rule.
    list_probas_by_rules : list
        List of probabilities associated with each rule (for classification tasks).
    list_probas_outside_by_rules : list
        List of probabilities for samples not satisfying each rule (for classification tasks).
    type_target : dtype
        Data type of the target variable.
    ridge : Ridge or RidgeCV instance
        Ridge regression model for final prediction (for regression tasks).
    _list_unique_categorical_values : list
        List of unique values for each categorical feature.
    _list_categorical_indexes : list
        List of indexes of categorical features.
    _array_quantile : array-like
        Array of quantiles for continuous features.
    Returns
    ----------
    RulesExtractorMixin: RulesExtractorMixin
        The current RulesExtractorMixin instance.
    Note
    ----
    This mixin provides core functionalities for SIRUS models, including rule extraction from decision trees,
    rule filtering, and prediction methods for both classification and regression tasks.
    It is designed to be inherited by specific SIRUS model classes.
    1. Tree exploration and rule extraction using a custom Node class.
    2. Generation of masks for data samples based on extracted rules.
    3. Filtering of redundant rules based on linear dependence.
    4. Fit and predict methods for classification and regression tasks.
    5. Integration with Ridge regression for regression tasks.
    6. Handling of both continuous and categorical features.
    7. Efficient memory and time management for large datasets.
    8. Compatibility with scikit-learn's decision tree structures.
    9. Customizable parameters for rule selection and model fitting.
    10. Designed for interpretability and simplicity in model predictions.
    """

    #######################################################
    ################ Fit main classifer   #################
    #######################################################
    def _fit_quantile_classifier(self, X, y, sample_weight=None):
        """
        fit method for RulesExtractorMixin.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)

            The target values (class labels in classification, real numbers in regression).
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights. If None, then samples are equally weighted.
        to_not_binarize_colindexes : list of int, optional (default=None)
            List of column indices in X that should not be binarized (i.e., treated as categorical).
        starting_index_one_hot : int, optional (default=None)
            If provided, all columns from this index onward are treated as one-hot encoded categorical features.
        Returns
        ----------
        self : object
            Returns the instance itself.
        1. Binarize continuous features in X using quantiles, while leaving specified categorical features unchanged.
        2. Fit the main classifier using the modified dataset.
        3. Store quantile information and categorical feature details for future use.
        4. Return the fitted instance.
        5. If no columns are specified for exclusion from binarization, treat all features as continuous.
        6. If columns are specified for exclusion, treat those as categorical and binarize only the continuous features.
        7. Handle one-hot encoded features if a starting index is provided, treating all features from that index onward as categorical.
        8. Use quantiles to binarize continuous features, ensuring that the binarization respects the distribution of the data.
        9. Store the quantiles used for binarization, unique values of categorical features, and their indices for future reference.
        10. Fit the main classifier with the modified dataset, ensuring that it can handle both continuous and categorical features appropriately.
        11. Ensure that sample weights are appropriately handled during the fitting process.
        12. Raise an error if no rules are found with the given p0 value, suggesting to decrease it.
        """
        if self.p0 > 1.0 or self.p0 < 0.0:
            raise ValueError("Invalid value for p0: p0 must be in the range (0, 1].")
        if self.max_n_rules <= 0:
            raise ValueError("max_n_rules must be a positive integer.")
        if self.quantile <= 1:
            raise ValueError("quantile must be an integer greater than 1.")

        X_bin = X.copy()
        if (self.to_not_binarize_colindexes is None) and (
            self.starting_index_one_hot is None
        ):  # All variables are continuous
            list_quantile = [
                np.quantile(X_bin, q=i, axis=0)
                for i in np.linspace(0, 1, self.quantile + 1)
            ]
            array_quantile = np.array(list_quantile)
            for dim in range(X.shape[1]):
                out = np.searchsorted(
                    array_quantile[:, dim], X_bin[:, dim], side="left"
                )
                X_bin[:, dim] = array_quantile[out, dim]
            _list_unique_categorical_values = (
                None  # set these to None if all variables are continuous
            )
            _list_categorical_indexes = (
                None  # set these to None if all variables are continuous
            )
        else:
            categorical = np.zeros((X.shape[1],), dtype=bool)
            if self.starting_index_one_hot is None:
                _list_categorical_indexes = self.to_not_binarize_colindexes
            elif self.to_not_binarize_colindexes is None:
                _list_categorical_indexes = [
                    i for i in range(self.starting_index_one_hot, X_bin.shape[1])
                ]
            else:
                _list_categorical_indexes = self.to_not_binarize_colindexes + [
                    i for i in range(self.starting_index_one_hot, X_bin.shape[1])
                ]
            ## the last indexes of X must contains the one hot encoded variables !
            categorical[_list_categorical_indexes] = True
            list_quantile = [
                np.quantile(X_bin[:, ~categorical], q=i, axis=0)
                for i in np.linspace(0, 1, self.quantile + 1)
            ]
            _list_unique_categorical_values = [
                np.unique(X_bin[:, i]) for i in _list_categorical_indexes
            ]
            array_quantile = np.array(list_quantile)

            array_dim_indices_samples = np.arange(0, X.shape[1])
            array_continuous_dim_indices_samples = array_dim_indices_samples[
                ~categorical
            ]
            for ind_dim_quantile, cont_dim_samples in enumerate(
                array_continuous_dim_indices_samples
            ):
                out = np.searchsorted(
                    array_quantile[:, ind_dim_quantile],
                    X_bin[:, cont_dim_samples],
                    side="left",
                )
                X_bin[:, cont_dim_samples] = array_quantile[out, ind_dim_quantile]

        super().fit(
            X_bin,
            y,
            sample_weight=sample_weight,
        )
        self._array_quantile = array_quantile
        self._list_unique_categorical_values = _list_unique_categorical_values  # list of each categorical features containing unique values for each of them
        self._list_categorical_indexes = _list_categorical_indexes  # indices of each categorical features, including the one hot encoded

    def fit(self, X, y, sample_weight=None):
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
        X, y = validate_data(self, X, y)
        self._fit_quantile_classifier(X, y, sample_weight)
        rules_ = []
        estimators = (
            self.estimators_.flatten()  # flattened if needed
            if isinstance(self.estimators_[0], np.ndarray)
            else self.estimators_
        )
        for dtree in estimators:  ## extraction  of all trees rules
            tree = dtree.tree_
            curr_tree_rules = _extract_single_tree_rules(tree)
            if len(curr_tree_rules) > 0 and len(curr_tree_rules[0]) > 0:
                # to avoid empty rules
                rules_.extend(curr_tree_rules)
        self._fit_rules(X, y, rules_, sample_weight)
        # Will call the _fit_rules for classifier or regressor (implemented in child class)
        compute_stability_criterion(self)


class RulesExtractorClassifierMixin(RulesExtractorMixin):
    def _fit_rules(self, X, y, rules_, sample_weight=None):
        """
        Fit method for RulesExtractorMixin in classification case.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels).
        rules_ : list
            List of all possible rules extracted from the ensemble of trees.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights for each instance.
        Returns
        ----------
        None
        1. Count unique rules and their frequencies.
        2. Apply post-treatment to filter redundant rules.
        3. Calculate probabilities for each rule based on the training data.
        4. Store the extracted rules and their associated probabilities.
        5. The method ensures that only relevant and non-redundant rules are retained for the final model.
        6. It handles both the presence and absence of sample weights during probability calculations.
        """
        rules_str = [str(elem) for elem in rules_]  # Trick for np.unique
        rules_, rules_freq_ = get_top_rules(rules_str=rules_str, p0=self.p0)
        #### APPLY POST TREATMEANT : remove redundant rules
        res = _rules_filtering_stochastic(
            rules=rules_,
            probas=rules_freq_,
            max_n_rules=self.max_n_rules,
            n_features_in_=self.n_features_in_,
            quantiles=self._array_quantile,
            random_state=self.random_state,
            list_unique_categorical_values=self._list_unique_categorical_values,
            list_categorical_indexes=self._list_categorical_indexes,
        )  ## Maximum number of rule to keep=25
        self.rules_ = res["rules"]
        self.rules_freq_ = res["probas"]  # usefull ?

        list_probas_by_rules = []
        list_probas_outside_by_rules = []
        if sample_weight is None:
            sample_weight = np.full((len(y),), 1)  ## vector of ones
        rules_mask = generate_masks_rules(X, self.rules_)
        for i in range(len(self.rules_)):
            # for loop for getting all the values in train (X) passing the rules
            final_mask = rules_mask[:, i]
            y_train_rule, y_train_outside_rule = y[final_mask], y[~final_mask]
            sample_weight_rule, sample_weight_outside_rule = (
                sample_weight[final_mask],
                sample_weight[~final_mask],
            )

            list_probas = []
            list_probas_outside_rules = []
            for cl in range(self.n_classes_):  # iteration on each class of the target
                curr_probas = (
                    sample_weight_rule[y_train_rule == cl].sum()
                    / sample_weight_rule.sum()
                    if len(y_train_rule) != 0
                    else 0
                )
                curr_probas_outside_rules = (
                    sample_weight_outside_rule[y_train_outside_rule == cl].sum()
                    / sample_weight_outside_rule.sum()
                    if len(y_train_outside_rule) != 0
                    else 0
                )
                list_probas.append(curr_probas)  # len n_classes_
                list_probas_outside_rules.append(
                    curr_probas_outside_rules
                )  # len n_classes_

            list_probas_by_rules.append(
                list_probas
            )  # list of len n_rules of list of len n_classes_
            list_probas_outside_by_rules.append(list_probas_outside_rules)

        self.list_probas_by_rules = list_probas_by_rules
        self.list_probas_outside_by_rules = list_probas_outside_by_rules
        self.type_target = y.dtype

    def predict_proba(self, X):
        """
        predict_proba method for RulesExtractorMixin. in classification case
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        ----------
        y_pred_probas : array-like, shape (n_samples, n_classes)
            The predicted class probabilities for each sample.
        """
        X = validate_data(self, X)
        y_pred_probas = np.zeros((len(X), self.n_classes_))
        rules_mask = generate_masks_rules(X, self.rules_)
        for indice in range(len(self.rules_)):
            final_mask = rules_mask[:, indice]
            y_pred_probas[final_mask] += self.list_probas_by_rules[indice]
            # add the asociated rule probability
            y_pred_probas[~final_mask] += self.list_probas_outside_by_rules[
                indice
            ]  # If the rule is not verified we add the probas of the training samples not verifying the rule.

        y_pred_probas = (1 / len(self.rules_)) * (y_pred_probas)
        return y_pred_probas

    def predict(self, X):
        """
        Predict method for RulesExtractorMixin in classification case.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        ----------
        y_pred : array-like, shape (n_samples,)
            The predicted classes for each sample.
        """
        X = validate_data(self, X)
        y_pred_probas = self.predict_proba(X=X)
        y_pred_numeric = np.argmax(y_pred_probas, axis=1)
        if self.type_target is not int:
            y_pred = y_pred_numeric.copy().astype(self.type_target)
            for i, cls in enumerate(self.classes_):
                y_pred[y_pred_numeric == i] = cls
            return y_pred.ravel().reshape(
                -1,
            )
        else:
            return y_pred_numeric.ravel().reshape(
                -1,
            )


class RulesExtractorRegressorMixin(RulesExtractorMixin):
    def _fit_rules(self, X, y, rules_, sample_weight=None):
        """
        Fit method for RulesExtractorMixin in regression case.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values (real numbers).
        rules_ : list
            List of all possible rules extracted from the ensemble of trees.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights for each instance.
        Returns
        ----------
        None
        1. Validate input data and initialize parameters.
        2. Count unique rules and their frequencies.
        3. Apply post-treatment to filter redundant rules.
        4. Calculate mean target values for samples satisfying and not satisfying each rule.
        5. Store the extracted rules and their associated mean target values.
        6. Fit a Ridge regression model using the rule-based features.
        7. The method ensures that only relevant and non-redundant rules are retained for the final model.
        8. It handles both the presence and absence of sample weights during model fitting.
        """
        rules_str = [str(elem) for elem in rules_]  # Trick for np.unique
        rules_, rules_freq_ = get_top_rules(rules_str=rules_str, p0=self.p0)
        if len(rules_) == 0:
            raise ValueError(
                "No rule found with the given p0 value. Try to decrease it."
            )

        #### APPLY POST TREATMEANT : remove redundant rules
        res = _rules_filtering_stochastic(
            rules=rules_,
            probas=rules_freq_,
            max_n_rules=self.max_n_rules,
            n_features_in_=self.n_features_in_,
            quantiles=self._array_quantile,
            random_state=self.random_state,
            list_unique_categorical_values=self._list_unique_categorical_values,
            list_categorical_indexes=self._list_categorical_indexes,
        )  ## Maximum number of rule to keep=25
        self.rules_ = res["rules"]
        self.rules_freq_ = res["probas"]
        # list_mask_by_rules = []
        list_output_by_rules = []
        list_output_outside_by_rules = []
        gamma_array = np.zeros((X.shape[0], len(self.rules_)))
        rules_mask = generate_masks_rules(X, self.rules_)
        for rule_number, current_rules in enumerate(self.rules_):
            # for loop for getting all the values in train (X) passing the rules
            final_mask = rules_mask[:, rule_number]
            output_value = y[final_mask].mean() if final_mask.any() else 0
            output_outside_value = y[~final_mask].mean() if (~final_mask).any() else 0

            list_output_by_rules.append(output_value)
            list_output_outside_by_rules.append(output_outside_value)

            gamma_array[final_mask, rule_number] = output_value
            gamma_array[~final_mask, rule_number] = output_outside_value

        self.list_probas_by_rules = list_output_by_rules
        self.list_probas_outside_by_rules = list_output_outside_by_rules
        self.type_target = y.dtype

        ## final predictor fitting : Ridge regression with positive coefficients
        best_alpha, results = ridge_cv_positive(
            gamma_array, y, random_state=self.random_state
        )
        self.ridge = Ridge(
            alpha=best_alpha,
            fit_intercept=True,
            positive=True,
            random_state=self.random_state,
        )
        self.ridge.fit(gamma_array, y)
        self.list_probas_by_rules_without_coefficients = (
            self.list_probas_by_rules.copy()
        )
        self.list_probas_outside_by_rules_without_coefficients = (
            self.list_probas_outside_by_rules.copy()
        )
        self.list_coefficients_by_rules = self.ridge.coef_
        self.coeff_intercept = self.ridge.intercept_
        for indice in range(len(self.rules_)):
            # Scale the probabilities by the learned coefficients
            coeff = (
                self.ridge.coef_[indice]
                if self.ridge.coef_.ndim == 1
                else self.ridge.coef_[:, indice]
            )
            self.list_probas_by_rules[indice] = (
                coeff * self.list_probas_by_rules[indice]
            ).tolist()
            self.list_probas_outside_by_rules[indice] = (
                coeff * self.list_probas_outside_by_rules[indice]
            ).tolist()

    def predict(self, X):
        """
        Predict using the RulesExtractorMixin regressor.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.

        """
        X = validate_data(self, X)
        rules_mask = generate_masks_rules(X, self.rules_)
        gamma_array = np.zeros((len(X), len(self.rules_)))
        for indice in range(len(self.rules_)):
            final_mask = rules_mask[:, indice]
            gamma_array[final_mask, indice] = self.list_probas_by_rules[indice]
            gamma_array[~final_mask, indice] = self.list_probas_outside_by_rules[indice]
        # y_pred = self.ridge.predict(gamma_array)
        y_pred = gamma_array.sum(axis=1) + self.ridge.intercept_

        return y_pred
