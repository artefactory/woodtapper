from functools import reduce
from operator import and_

import numpy as np
from sklearn.tree import _tree
from sklearn.tree import _splitter
import sklearn.tree._classes
from sklearn.linear_model import Ridge
import time

from .Splitter.QuantileSplitter import QuantileBestSplitter
from .utils import Node, get_top_rules, ridge_cv_positive

sklearn.tree._classes.DENSE_SPLITTERS = {
    "best": _splitter.BestSplitter,
    "random": _splitter.BestSplitter,
    "quantile": QuantileBestSplitter,
}


class SirusMixin:
    """
    Mixin of SIRUS. Base of all SIRUS models.

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
    all_possible_rules_list : list
        List of all possible rules extracted from the ensemble.
    all_possible_rules_frequency_list : list
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
    SirusMixin: SirusMixin
        The current SirusMixin instance.
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
    ##### Auxiliary function for path construction  #######
    #######################################################
    def _explore_tree(self, node_id, side, tree):
        """
        Whole tree structure recursive explorator (with Node class).
        Node class are associated to their childs if internal node.
        Parameters
        ----------
        node_id : int
            Starting node id for the tree structure exploration.
        side : str
            Current node cutting side. 'L' for left and 'R' for right. 'root' for the root node.
        Returns
        ----------
        Node: Node
            The starting Node of the first call of this function (given node_id by user).

        """
        if (
            tree.children_left[node_id] != _tree.TREE_LEAF
        ):  # possible to add a max_depth constraint exploration value
            id_left_child = tree.children_left[node_id]
            id_right_child = tree.children_right[node_id]
            children = [
                self._explore_tree(id_left_child, "L", tree),  # L for \leq
                self._explore_tree(id_right_child, "R", tree),
            ]
        else:
            return Node(
                feature=tree.feature[node_id],
                treshold=tree.threshold[node_id],
                side=side,
                node_id=node_id,
            )

        return Node(
            tree.feature[node_id], tree.threshold[node_id], side, node_id, *children
        )

    def _construct_longest_paths(self, root):
        """
        Generate tree_strucre, i.e a list of rules that all starts FROM root node TO a leaf.
        The lengh of this list is equal to the number of leaf.
        Parameters
        ----------
        root : Node instance
            The tree root.
        Returns
        ----------
        tree_structure : list
            list of longest paths, i.e a list of rules that all starts FROM root node TO a leaf

        """
        tree_structure = [[]]
        stack = [(root, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            curr_rule, indice_in_tree_struct = stack.pop()
            is_split_node = curr_rule.feature != -2  # -2 means leaf node in sklearn

            if is_split_node:
                rule_left = (curr_rule.feature, curr_rule.treshold, "L")
                rule_right = (curr_rule.feature, curr_rule.treshold, "R")
                common_path_rules = tree_structure[indice_in_tree_struct].copy()
                common_path_rules.append(rule_right)
                tree_structure.append(common_path_rules)  ## RIGHT : Added at the end
                tree_structure[indice_in_tree_struct].append(
                    rule_left
                )  ## LEFT  : Added depending on indice_in_tree_struct

                stack.append((curr_rule.children[0], indice_in_tree_struct))
                stack.append((curr_rule.children[1], len(tree_structure) - 1))
            else:
                continue
        return tree_structure

    def _split_sub_rules(self, path):
        """
        From a multiple rule, generate the associated sub multiple/single rules.
        Auxiliar function for _generate_all_possible_rules.

        Parameters
        ----------
        path : list
            A multiple rule (list of single rules).

        Returns
        ----------
        list_sub_path : list
            List of sub-rules extracted from the given multiple rule.
        1. Iterate through the given path to generate sub-rules.
        2. Return the list of generated sub-rules.
        3. The function ensures that only valid sub-rules (with at least two conditions) are included when required.
        4. This method is essential for expanding the rule set derived from decision trees.
        5. It helps in capturing more granular patterns within the data by considering all possible combinations of conditions.
        8. The generated sub-rules can be used for further analysis or model fitting.
        """
        list_sub_path = []
        max_size_curr_path = len(path)
        for j in range(max_size_curr_path):
            curr_path = path[: (max_size_curr_path - j)]
            if len(curr_path) >= 2:
                list_sub_path.append(curr_path)
        return list_sub_path

    def _generate_all_possible_rules(self, tree_structure):
        """
        Generate all possibles rules (single and multiple) from a tree_strucre (i.e a list of node to leafs paths)
        Auxiliar function for _extract_single_tree_rules.

        Parameters
        ----------
        tree_structure : list
            list of longest paths, i.e a list of rules that all starts FROM root node TO a leaf

        Returns
        ----------
        all_paths_list : list
            List of all possible rules (single and multiple) extracted from the tree.
        """
        all_paths_list = []
        for i in range(len(tree_structure)):
            curr_path = tree_structure[i]
            list_sub_path = self._split_sub_rules(curr_path)
            all_paths_list.extend(list_sub_path)
            all_paths_list.append([curr_path[0]])
        return all_paths_list

    def _extract_single_tree_rules(self, tree):
        """
        Extract all possible rules (single and multiple) from a single tree.
        Parameters
        ----------
        tree : sklearn DecisionTree instance
            The tree from which to extract rules.
        Returns
        ----------
        all_possible_rules_list : list
            List of all possible rules (single and multiple) extracted from the tree.
        1. Explore the tree structure and create Node instances.
        2. Generate the tree structure with Node instances.
        3. Explore the tree structure to extract the longest rules (rules from root to a leaf).
        4. Generate all possible rules (single and multiple) from the tree structure.
        5. Return the list of all possible rules.
        """
        root = self._explore_tree(0, "Root", tree)  ## Root node
        tree_structure = self._construct_longest_paths(
            root
        )  ## generate the tree structure with Node instances
        if (
            len(tree_structure[0]) == 0 and root.feature == -2
        ):  # -2 means leaf node in sklearn
            # case where root node is also a leaf
            all_possible_rules_list = [[]]  # Tree with only one leaf
        else:
            all_possible_rules_list = self._generate_all_possible_rules(
                tree_structure
            )  # Explre the tree structure to extract the longest rules (rules from root to a leaf)
        return all_possible_rules_list

    def _generate_single_rule_mask(self, X, dimension, treshold, sign):
        """
        Uses constraints of a unitary rule (len 1) to generate the associated mask for data set X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        dimension : int
            The feature indice of the rule.
        treshold : float
            The treshold of the rule.
        sign : str
            The sign of the rule ('L' for less or equal and 'R' for greater)
        Returns
        ----------
        mask : array-like, shape (n_samples,)
            Boolean mask indicating which samples satisfy the single rule.
        1. Generate a boolean mask based on the rule's dimension, threshold, and sign.
        2. The mask indicates which samples in X satisfy the condition defined by the rule.
        3. Return the generated mask.
        4. The function supports two types of conditions: 'L' for less than or equal to the threshold, and 'R' for greater than the threshold.
        5. This method is essential for filtering data samples based on specific rule conditions.
        """
        if sign == "L":
            return X[:, dimension] <= treshold
        else:
            return X[:, dimension] > treshold

    def _from_rules_to_constraint(self, rule):
        """
        Extract informations from a single rule.
        Auxiliar function for  _generate_single_rule_mask.

        Parameters
        ----------
        rule : tuple
            A single rule (dimension, treshold, sign)

        Returns
        ----------
        dimension : int
            The feature indice of the rule.
        treshold : float
            The treshold of the rule.
        sign : str
            The sign of the rule ('L' for less or equal and 'R' for greater)
        """
        dimension = rule[0]
        treshold = rule[1]
        sign = rule[2]
        return dimension, treshold, sign

    def _generate_mask_rule(self, X, rules):
        """
        Generate the mask associated to a rule of len >=1.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        rules : list of tuples
            A rule (list of single rules).
        Returns
        ----------
        final_mask : array-like, shape (n_samples,)
            Boolean mask indicating which samples satisfy the rule.
        """
        list_mask = []
        for j in range(len(rules)):
            dimension, treshold, sign = self._from_rules_to_constraint(rule=rules[j])
            mask = self._generate_single_rule_mask(
                X=X, dimension=dimension, treshold=treshold, sign=sign
            )
            list_mask.append(mask)
        final_mask = reduce(and_, list_mask)
        return final_mask

    def _generate_masks_rules(self, X):
        """
        Generate the masks associated to all the rules.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        ----------
        rules_mask : array-like, shape (n_samples, n_rules)
            Boolean mask matrix indicating which samples satisfy each rule.
        """
        rules_mask = np.zeros((X.shape[0], self.n_rules), dtype=bool)
        for rule_number, current_rules in enumerate(self.all_possible_rules_list):
            # for loop for getting all the values in train (X) passing the rules
            final_mask = self._generate_mask_rule(X=X, rules=current_rules)
            rules_mask[:, rule_number] = final_mask
        return rules_mask

    def _paths_filtering_matrix_stochastic(self, paths, proba, num_rule):
        """
        Post-treatment for rules when tree depth is at most 2 (deterministic algorithm).
        Parameters
        ----------
        paths : list
            List of rules (each rule is a list of splits; each split [var, thr, dir])
        proba : list
            Probabilities associated with each path/rule
        num_rule : int
            Max number of rules to keep
        Returns
        ----------
        dict: {'paths': filtered_paths, 'proba': filtered_proba}
        1. Generate an independent dataset for checking rule redundancy.
        2. Iterate through the paths and apply redundancy checks.
        3. Return the filtered paths and their associated probabilities.
        4. The redundancy check is based on the rank of a matrix formed by the masks of the rules.
        5. If the rank of the matrix increases when adding a new rule, it is considered non-redundant and kept.
        6. This method ensures that the selected rules are diverse and not linearly dependent.
        7. The process continues until the desired number of rules is reached or all paths are evaluated.
        8. The function returns a dictionary containing the filtered paths and their probabilities.
        """
        paths_ftr = []
        proba_ftr = []
        # split_gen = []
        ind_max = len(paths)
        ind = 0
        num_rule_temp = 0

        n_samples_indep = 10000
        data_indep = np.zeros((n_samples_indep, self.n_features_in_), dtype=float)
        ind_dim_continuous_array_quantile = (
            0  ## indice dans array_quantile des variables continues
        )
        ind_dim_categorcial_list_unique_elements = 0  ## indice dans _list_unique_categorical_values des variables catégorielles
        # Generate an independent data set for checking rule redundancy
        for ind_dim_abs in range(self.n_features_in_):
            np.random.seed(ind_dim_abs)
            if (self._list_categorical_indexes is not None) and (
                ind_dim_abs in self._list_categorical_indexes
            ):  # Categorical variable
                data_indep[:, ind_dim_abs] = np.random.choice(
                    np.unique(
                        self._list_unique_categorical_values[
                            ind_dim_categorcial_list_unique_elements
                        ]
                    ),
                    size=n_samples_indep,
                    replace=True,
                )
                ind_dim_categorcial_list_unique_elements += 1
            else:  # Continuous variable
                elem_low = (
                    self._array_quantile[:, ind_dim_continuous_array_quantile].min() - 1
                )
                elem_high = (
                    self._array_quantile[:, ind_dim_continuous_array_quantile].max() + 1
                )
                data_indep[:, ind_dim_abs] = np.random.uniform(
                    low=elem_low, high=elem_high, size=n_samples_indep
                )
                ind_dim_continuous_array_quantile += 1
        np.random.seed(self.random_state)

        while num_rule_temp < num_rule and ind < ind_max:
            curr_path = paths[ind]
            if curr_path in paths_ftr:  ## Avoid duplicates
                ind += 1
                num_rule_temp = len(paths_ftr)
                continue
            elif len(paths_ftr) != 0:  ## If there are already filtered paths
                related_paths_ftr = paths_ftr  # We comlpare the new rule to all the previous ones already selected.
                if len(related_paths_ftr) == 0:  ## If there are no related paths
                    paths_ftr.append(curr_path)
                    proba_ftr.append(proba[ind])
                else:
                    rules_ensemble = related_paths_ftr + [curr_path]
                    list_matrix = [[] for i in range(len(rules_ensemble))]
                    for i, x in enumerate(rules_ensemble):
                        mask_x = self._generate_mask_rule(X=data_indep, rules=x)
                        list_matrix[i] = mask_x

                    if len(list_matrix) > 0:
                        # Check if the current rule is redundant with the previous ones trough matrix rank
                        matrix = np.array(list_matrix).T
                        ones_vector = np.ones((len(matrix), 1))  # Vector of ones
                        matrix = np.hstack((matrix, ones_vector))
                        matrix_rank = np.linalg.matrix_rank(matrix)
                        n_rules_compared = len(rules_ensemble)
                        if matrix_rank == (n_rules_compared) + 1:
                            # The current rule is not redundant with the previous ones
                            paths_ftr.append(curr_path)
                            proba_ftr.append(proba[ind])
                ind += 1
                num_rule_temp = len(paths_ftr)

            else:  ## If there are no filtered paths yet
                paths_ftr.append(curr_path)
                proba_ftr.append(proba[ind])
                ind += 1
                num_rule_temp = len(paths_ftr)

        return {"paths": paths_ftr, "proba": proba_ftr}

    def _paths_filtering_stochastic(self, paths, proba, num_rule):
        """
        Post-treatment for rules.

        Args:
            paths (list): List of rules (each rule is a list of splits; each split [var, thr, dir])
            proba (list): Probabilities associated with each path/rule
            num_rule (int): Max number of rules to keep

        Returns:
            dict: {'paths': filtered_paths, 'proba': filtered_proba}
        """
        return self._paths_filtering_matrix_stochastic(
            paths=paths, proba=proba, num_rule=num_rule
        )

    #######################################################
    ############ Classification fit and predict  ##########
    #######################################################
    def _fit_rules(self, X, y, all_possible_rules_list, sample_weight=None):
        """
        Fit method for SirusMixin in classification case.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels).
        all_possible_rules_list : list
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
        start = time.time()
        all_possible_rules_list_str = [
            str(elem) for elem in all_possible_rules_list
        ]  # Trick for np.unique
        all_possible_rules_list, all_possible_freq_list = get_top_rules(
            all_possible_rules_list_str=all_possible_rules_list_str, p0=self.p0
        )
        #### APPLY POST TREATMEANT : remove redundant rules
        start_lin_dep = time.time()
        res = self._paths_filtering_stochastic(
            paths=all_possible_rules_list,
            proba=all_possible_freq_list,
            num_rule=self.num_rule,
        )  ## Maximum number of rule to keep=25
        end_lin_dep = time.time()
        print(
            f"Linear dep post-treatment took {end_lin_dep - start_lin_dep:.4f} seconds"
        )
        self.all_possible_rules_list = res["paths"]
        self.all_possible_rules_frequency_list = res["proba"]  # usefull ?
        self.n_rules = len(self.all_possible_rules_list)
        end = time.time()
        print(f"Rules extraction took {end - start:.4f} seconds")

        list_probas_by_rules = []
        list_probas_outside_by_rules = []
        if sample_weight is None:
            sample_weight = np.full((len(y),), 1)  ## vector of ones
        rules_mask = self._generate_masks_rules(X=X)
        for i in range(self.n_rules):
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

    def predict_proba(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin. in classification case
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        to_add_probas_outside_rules : bool, optional (default=True)
            Whether to include probabilities from samples not satisfying the rules.
        Returns
        ----------
        y_pred_probas : array-like, shape (n_samples, n_classes)
            The predicted class probabilities for each sample.
        """
        y_pred_probas = np.zeros((len(X), self.n_classes_))
        rules_mask = self._generate_masks_rules(X=X)
        for indice in range(self.n_rules):
            final_mask = rules_mask[:, indice]
            y_pred_probas[final_mask] += self.list_probas_by_rules[
                indice
            ]  ## add the asociated rule probability

            if to_add_probas_outside_rules:  # ERWAN TIPS !!
                y_pred_probas[~final_mask] += self.list_probas_outside_by_rules[
                    indice
                ]  ## If the rule is not verified we add the probas of the training samples not verifying the rule.
        if to_add_probas_outside_rules:
            y_pred_probas = (1 / self.n_rules) * (y_pred_probas)
        else:
            scaling_coeffs = y_pred_probas.sum(axis=1)
            y_pred_probas = (
                y_pred_probas
                / np.array([scaling_coeffs, scaling_coeffs, scaling_coeffs]).T
            )

        return y_pred_probas

    def predict(self, X, to_add_probas_outside_rules=True):
        """
        Predict method for SirusMixin in classification case.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        to_add_probas_outside_rules : bool, optional (default=True)
            Whether to include probabilities from samples not satisfying the rules.
        Returns
        ----------
        y_pred : array-like, shape (n_samples,)
            The predicted classes for each sample.
        """
        y_pred_probas = self.predict_proba(
            X=X, to_add_probas_outside_rules=to_add_probas_outside_rules
        )
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

    #######################################################
    ############# Regressor fit and predict  ##############
    #######################################################
    def _fit_rules_regressor(
        self, X, y, all_possible_rules_list, sample_weight=None, to_encode_target=False
    ):
        """
        Fit method for SirusMixin in regression case.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values (real numbers).
        all_possible_rules_list : list
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
        all_possible_rules_list_str = [
            str(elem) for elem in all_possible_rules_list
        ]  # Trick for np.unique
        all_possible_rules_list, all_possible_freq_list = get_top_rules(
            all_possible_rules_list_str=all_possible_rules_list_str, p0=self.p0
        )
        if len(all_possible_rules_list) == 0:
            raise ValueError(
                "No rule found with the given p0 value. Try to decrease it."
            )

        #### APPLY POST TREATMEANT : remove redundant rules
        res = self._paths_filtering_stochastic(
            paths=all_possible_rules_list,
            proba=all_possible_freq_list,
            num_rule=self.num_rule,
        )  ## Maximum number of rule to keep=25
        self.all_possible_rules_list = res["paths"]
        self.all_possible_rules_frequency_list = res["proba"]
        self.n_rules = len(self.all_possible_rules_list)
        # list_mask_by_rules = []
        list_output_by_rules = []
        list_output_outside_by_rules = []
        gamma_array = np.zeros((X.shape[0], self.n_rules))
        rules_mask = self._generate_masks_rules(X=X)
        for rule_number, current_rules in enumerate(self.all_possible_rules_list):
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
        self.ridge.fit(gamma_array, y)  # sample_weight=sample_weight
        for indice in range(
            self.n_rules
        ):  # Scale the probabilities by the learned coefficients
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

    def _predict_regressor(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin for regression case.
        Parameters
        X : array-like, shape (n_samples, n_features)
            The input samples.
        to_add_probas_outside_rules : bool, optional (default=True)
            Whether to include probabilities from samples not satisfying the rules.
        Returns
        ----------
        y_pred : array-like, shape (n_samples,)
            The predicted values for each sample.
        1. Generate the feature matrix based on the rules for the input samples.
        2. Use the fitted Ridge regression model to predict target values.
        3. Return the predicted values.
        4. The method constructs the feature matrix by evaluating each rule on the input samples.
        5. It includes an intercept term in the feature matrix for the Ridge regression model.
        6. The predictions are made using the linear combination of the rule-based features and the learned coefficients from the Ridge model.
        7. The function supports the option to include or exclude probabilities for samples not satisfying the rules, although in this implementation it is always included in the feature matrix.
        8. The final output is a one-dimensional array of predicted values corresponding to each input sample.
        9. The method ensures that the predictions are consistent with the training process and the rules extracted from the decision trees.
        """
        gamma_array = np.zeros((X.shape[0], self.n_rules))
        rules_mask = self._generate_masks_rules(X=X)
        for indice in range(self.n_rules):
            final_mask = rules_mask[:, indice]
            gamma_array[final_mask, indice] = self.list_probas_by_rules[indice]
            if to_add_probas_outside_rules:  # ERWAN TIPS !!
                gamma_array[~final_mask, indice] = self.list_probas_outside_by_rules[
                    indice
                ]
        # y_pred = self.ridge.predict(gamma_array)
        y_pred = gamma_array.sum(axis=1) + self.ridge.intercept_

        return y_pred

    #######################################################
    ################ Fit main classiifer   ################
    #######################################################
    def _fit_quantile_classifier(self, X, y, sample_weight=None):
        """
        fit method for SirusMixin.
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
        start = time.time()
        if self.p0 > 1.0 or self.p0 < 0.0:
            raise ValueError("Invalid value for p0: p0 must be in the range (0, 1].")
        if self.num_rule <= 0:
            raise ValueError("num_rule must be a positive integer.")
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
        end = time.time()
        print(
            f"Pre-processing binarization took in fit_main_clasifier {end - start:.4f} seconds"
        )

        start = time.time()
        super().fit(
            X_bin,
            y,
            sample_weight=sample_weight,
        )
        end = time.time()
        print(f"Grow forest took {end - start:.4f} seconds")
        self._array_quantile = array_quantile
        self._list_unique_categorical_values = _list_unique_categorical_values  # list of each categorical features containing unique values for each of them
        self._list_categorical_indexes = _list_categorical_indexes  # indices of each categorical features, including the one hot encoded
