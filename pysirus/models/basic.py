from functools import reduce
from operator import and_
from math import isclose

import numpy as np
from sklearn.tree import _tree
from sklearn.tree import _splitter
import sklearn.tree._classes
from sklearn.linear_model import Ridge,RidgeCV

from ._QuantileSplitter import QuantileBestSplitter

sklearn.tree._classes.DENSE_SPLITTERS = {
    "best": _splitter.BestSplitter,
    "random": _splitter.BestSplitter,
    "quantile": QuantileBestSplitter,
}

class Node:
    """
    Tree node class

    Parameters
    ----------
    feature : int
        Current node feature indice splitting
    treshold : flaot
        Current node treshold splittting
    side : str
        Side of the rule. 'L' for Left (i.e. less or equal) and 'R' for right (i.e. gretter)
    node_id : int
       Current  Node id
    children : list of Node
        Child Nodes if not a leaf node

    """

    def __init__(self, feature=None, treshold=-1, side=None, node_id=-1, *children):
        self.node_id = node_id
        self.feature = feature
        self.treshold = treshold
        self.side = side
        if children:
            self.children = children
        else:
            self.children = []


class SirusMixin:
    """
    Mixin of SIRUS. Base of all SIRUS models.
    """
    #######################################################
    ##### Auxiliary function for path construction  #######
    #######################################################
    def explore_tree_(self, node_id, side, tree):
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
                self.explore_tree_(id_left_child, "L", tree),  # L for \leq
                self.explore_tree_(id_right_child, "R", tree),
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

    def construct_longest_paths_(self, root):
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
            is_split_node = curr_rule.feature != -2

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
                # print('c')
                continue
        return tree_structure

    def split_sub_rules_(self, path, is_removing_singleton=False):
        """
        From a multiple rule, generate the associated sub multiple/single rules.
        Auxiliar function for generate_all_possible_rules_.

        Parameters
        ----------

        Returns
        ----------
        """
        list_sub_path = []
        max_size_curr_path = len(path)
        if is_removing_singleton:
            int_to_add = 1
        else:
            int_to_add = 0
        for j in range(max_size_curr_path - int_to_add):
            curr_path = path[: (max_size_curr_path - j)]
            if len(curr_path) >= 2:
                list_sub_path.append(curr_path)
        return list_sub_path

    def generate_all_possible_rules_(self, tree_structure):
        """
        Generate all possibles rules (single and multiple) from a tree_strucre (i.e a list of node to leafs paths)

        Parameters
        ----------

        Returns
        ----------
        """
        all_paths_list = []
        for i in range(len(tree_structure)):
            curr_path = tree_structure[i]
            max_size_curr_path = len(curr_path)

            ## Single rules
            for k in range(max_size_curr_path):
                if [curr_path[k]] not in all_paths_list :
                    all_paths_list.append([curr_path[k]])

            ## We take all the rules strating from a head node
            for k in range(max_size_curr_path):
                list_sub_path = self.split_sub_rules_(
                    curr_path[k:], is_removing_singleton=True
                )
                all_paths_list.extend(list_sub_path)

            ## More complexe cases : internal rules
            if max_size_curr_path == 1:
                continue
            else: 
                curr_path_size_pair = (max_size_curr_path % 2) == 0
                if curr_path_size_pair:  ## PAIRS
                    for k in range(1, (max_size_curr_path // 2)):
                        list_sub_path = self.split_sub_rules_(
                            curr_path[k : max_size_curr_path - k],
                            is_removing_singleton=True,
                        )
                        all_paths_list.extend(list_sub_path)
                else:  ## IMPAIRS
                    for k in range(1, (max_size_curr_path // 2)):
                        list_sub_path = self.split_sub_rules_(
                            curr_path[k : max_size_curr_path - k],
                            is_removing_singleton=True,
                        )
                        all_paths_list.extend(list_sub_path)
        return all_paths_list

    def from_rules_to_constraint(self, rule):
        """
        Extract informations from a single rule.
        Auxiliar function for  generate_single_rule_mask.

        Parameters
        ----------

        Returns
        ----------
        """
        dimension = rule[0]
        treshold = rule[1]
        sign = rule[2]
        return dimension, treshold, sign

    def generate_single_rule_mask(self, X, dimension, treshold, sign):
        """
        Uses constraints of a single rule (len 1) to generatye the associated mask for data set X.

        Parameters
        ----------
        """
        if sign == "L":
            return X[:, dimension] <= treshold  # .mean()
        else:
            return X[:, dimension] > treshold  # .mean()


    def extract_single_tree_rules(self, tree):
        """
        Fit method for SirusMixin.
        """
        root = self.explore_tree_(0, "Root", tree)  ## Root node
        tree_structure = self.construct_longest_paths_(
            root
        )  ## generate the tree structure with Node instances
        all_possible_rules_list = self.generate_all_possible_rules_(
            tree_structure
        )  # Explre the tree structure to extract the longest rules (rules from root to a leaf)
        return all_possible_rules_list
    
    def generate_mask_rule(self,X,rules):
        """
        Generate the mask associated to a rule of len >=1.
        """
        list_mask=[]
        for j in range(len(rules)):
            dimension,treshold,sign = self.from_rules_to_constraint(rule=rules[j])
            mask = self.generate_single_rule_mask(X=X,dimension=dimension,treshold=treshold,sign=sign)
            list_mask.append(mask)
        final_mask = reduce(and_, list_mask)
        return final_mask
    

            
        return data
    def _related_rule(self,curr_rule, relative_rule):
        """
        Check if the current rule is related to relative_rule.
        Args:
            curr_rule (tuple): Current rule to check.
            A (tuple): First single rule.
            B (tuple): Second single rule.
        """
        
        if len(relative_rule)==1:
            A = relative_rule[0]
            if len(curr_rule) == 1: ## Both are len = 1
                return (curr_rule[0][0]== A[0])  and (curr_rule[0][1]== A[1])
            elif len(curr_rule) == 2:
                l1,l2 = curr_rule[0], curr_rule[1]
                return ((l1[0]== A[0])  and (l1[1]== A[1])) or ((l1[0]== A[0])  and (l1[1]== A[1]))
            else:
                raise ValueError(f"Rule {curr_rule} has more than two splits; this is not supported.")
        else:
            A, B = relative_rule
            if len(curr_rule) == 1:
                return ((curr_rule[0][0]== A[0])  and (curr_rule[0][1]== A[1])) or ((curr_rule[0][0]== B[0])  and (curr_rule[0][1]== B[1]))
            elif len(curr_rule) == 2:
                l1,l2 = curr_rule[0], curr_rule[1]
                return ((l1[0]== A[0])  and (l1[1]== A[1])) or ((l1[0]== B[0])  and (l1[1]== B[1])) or ((l1[0]== A[0])  and (l2[1]== A[1])) or ((l2[0]== B[0])  and (l2[1]== B[1]))
            else:
                raise ValueError(f"Rule {curr_rule} has more than two splits; this is not supported.")
        
    def paths_filtering_matrix_stochastic(self,paths, proba, num_rule):
        """
            Post-treatment for rules when tree depth is at most 2 (deterministic algorithm).
            Args:
                paths (list): List of rules (each rule is a list of splits; each split [var, thr, dir])
                proba (list): Probabilities associated with each path/rule
                num_rule (int): Max number of rules to keep
            Returns:
                dict: {'paths': filtered_paths, 'proba': filtered_proba}

        """
        paths_ftr = []
        proba_ftr = []
        #split_gen = []
        ind_max = len(paths)
        ind = 0
        num_rule_temp = 0
        
        n_samples_indep = 10000
        data_indep = np.zeros((n_samples_indep, self.array_quantile_.shape[1]), dtype=float)
        for j in range(self.array_quantile_.shape[1]):
            np.random.seed(j)
            elem_low = self.array_quantile_[:,j].min()-1
            elem_high = self.array_quantile_[:,j].max()+1
            data_indep[:,j]=np.random.uniform(low=elem_low, high=elem_high,size=n_samples_indep)
        np.random.seed(self.random_state)
            
        while num_rule_temp < num_rule and ind < ind_max:
            curr_path= paths[ind]
            if curr_path in paths_ftr: ## Avoid duplicates
                ind += 1
                num_rule_temp = len(paths_ftr)
                continue
            elif len(paths_ftr) != 0: ## If there are already filtered paths
                list_bool_related_rules = [self._related_rule(curr_path, x) for x in paths_ftr]
                related_paths_ftr = [path for path, boolean in zip(paths_ftr, list_bool_related_rules) if boolean]
                #**print('related_paths_ftr :',related_paths_ftr)
                if len(related_paths_ftr) == 0: ## If there are no related paths
                    paths_ftr.append(curr_path)
                    proba_ftr.append(proba[ind])
                else:
                    rules_ensemble = related_paths_ftr + [curr_path]
                    #related_paths_ftr = paths_ftr## WARNINGS on compare toutes les règles finalement !!!! #####
                    list_matrix = [[] for i in range(len(rules_ensemble))]
                    for i,x in enumerate(rules_ensemble):
                        mask_x = self.generate_mask_rule(X=data_indep,rules=x)
                        list_matrix[i]=mask_x                       

                    if len(list_matrix) >0:
                        # Check if the current rule is redundant with the previous ones trough matrix rank
                        matrix = np.array(list_matrix).T
                        ones_vector = np.ones((len(matrix),1))  # Vector of ones
                        matrix = np.hstack((matrix,ones_vector))
                        matrix_rank = np.linalg.matrix_rank(matrix)
                        n_rules_compared = len(rules_ensemble)
                        if matrix_rank == (n_rules_compared) + 1:  
                            # The current rule is not redundant with the previous ones
                            paths_ftr.append(curr_path)
                            proba_ftr.append(proba[ind])
                ind += 1
                num_rule_temp = len(paths_ftr)
                
            else: ## If there are no filtered paths yet
                paths_ftr.append(curr_path)
                proba_ftr.append(proba[ind])
                ind += 1
                num_rule_temp = len(paths_ftr)
        
        return {'paths': paths_ftr, 'proba': proba_ftr}
                 
        
    def paths_filtering_stochastic(self,paths, proba, num_rule):
        """
            Post-treatment for rules when tree depth is at most 2 (deterministic algorithm).
            
            Args:
                paths (list): List of rules (each rule is a list of splits; each split [var, thr, dir])
                proba (list): Probabilities associated with each path/rule
                num_rule (int): Max number of rules to keep
            
            Returns:
                dict: {'paths': filtered_paths, 'proba': filtered_proba}
        """
        #if len(paths) <= num_rule:
        #    return {'paths': paths, 'proba': proba}
        #else:
        return self.paths_filtering_matrix_stochastic(paths=paths, proba=proba, num_rule=num_rule)
    #######################################################
    ############ Classification fit and predict  ##########
    #######################################################

    def fit_forest_rules(self, X, y, all_possible_rules_list, p0=0.0,sample_weight=None):
        all_possible_rules_list_str = [
            str(elem) for elem in all_possible_rules_list
        ]  # Trick for np.unique
        unique_str_rules, indices_rules, count_rules = np.unique(
            all_possible_rules_list_str, return_counts=True, return_index=True
        )  # get the unique rules and count
        proportions_count = count_rules / len(
            count_rules
        )  # Get frequency of each rules
        proportions_count_sort = -np.sort(
            -proportions_count
        )  # Sort rules frequency by descending order
        proportions_count_sort_indices = np.argsort(
            -count_rules
        )  # Sort rules coubnt by descending order (same results as proportions)
        n_rules_to_keep = (
            proportions_count_sort > p0
        ).sum()  ## not necssary to sort proportions_count...
        all_possible_rules_list = [
            eval(unique_str_rules[i])
            for i in proportions_count_sort_indices[:n_rules_to_keep]
        ]#all possible rules reindexed 
        #### APPLY POST TREATMEANT : remove redundant rules

        #print('25 all_possible_rules_list : ',all_possible_rules_list[:25])
        #print('####'*5)
        #print('25 proportions_count_sort : ',proportions_count_sort[:25])
        #print('####'*5)
        res = self.paths_filtering_stochastic(paths=all_possible_rules_list, proba=proportions_count_sort, num_rule=25) ## Maximum number of rule to keep=25
        self.all_possible_rules_list = res['paths']
        self.n_rules = len(self.all_possible_rules_list)

        # list_mask_by_rules = []
        list_probas_by_rules = []
        list_probas_outside_by_rules = []
        if sample_weight is  None:
            sample_weight = np.full((len(y),),1)## vector of ones

        for current_rules in self.all_possible_rules_list:
            # for loop for getting all the values in train (X) passing the rules
            list_mask = []
            for j in range(
                len(current_rules)
            ):  ## iteraation on each signle rule of the potentail multiple rule
                dimension, treshold, sign = self.from_rules_to_constraint(
                    rule=current_rules[j]
                )
                mask = self.generate_single_rule_mask(
                    X=X, dimension=dimension, treshold=treshold, sign=sign
                )  # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(and_, list_mask)
            y_train_rule = y[final_mask] 
            y_train_outside_rule = y[~final_mask] * sample_weight[~final_mask]
            sample_weight_rule = sample_weight[final_mask]
            sample_weight_outside_rule = sample_weight[~final_mask]

            list_probas = []
            list_probas_outside_rules = []
            for cl in range(self.n_classes_):  # iteration on each class of the target
                if len(y_train_rule) == 0:
                    curr_probas = 0
                else:
                    curr_probas = sample_weight_rule[y_train_rule == cl].sum() / sample_weight_rule.sum()
                if len(y_train_outside_rule) == 0:
                    curr_probas_outside_rules = 0
                else:
                    curr_probas_outside_rules = sample_weight_outside_rule[y_train_outside_rule == cl].sum() / sample_weight_outside_rule.sum()
    
                list_probas.append(curr_probas)
                list_probas_outside_rules.append(curr_probas_outside_rules)

            # list_mask_by_rules.append(final_mask) # uselesss
            list_probas_by_rules.append(list_probas)
            list_probas_outside_by_rules.append(list_probas_outside_rules)

        # self.list_mask_by_rules = list_mask_by_rules
        self.list_probas_by_rules = list_probas_by_rules
        self.list_probas_outside_by_rules = list_probas_outside_by_rules
        self.type_target = y.dtype

    def predict_proba(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        y_pred_probas = np.zeros((len(X), self.n_classes_))
        for indice in range(self.n_rules):
            current_rules = self.all_possible_rules_list[indice]
            list_mask = []
            for j in range(
                len(current_rules)
            ):  ## iteration on each signle rule of the potentail multiple rule
                dimension, treshold, sign = self.from_rules_to_constraint(
                    rule=current_rules[j]
                )
                mask = self.generate_single_rule_mask(
                    X=X, dimension=dimension, treshold=treshold, sign=sign
                )  # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(
                and_, list_mask
            )  ## test samples that verify the current multiple rule
            y_pred_probas[final_mask] += self.list_probas_by_rules[
                indice
            ]  ## add the asociated rule probability

            if to_add_probas_outside_rules:  # ERWAN TIPS !!
                y_pred_probas[~final_mask] += self.list_probas_outside_by_rules[
                    indice
                ]  ## If the rule is not verified we add the probas of the training samples not verifying the rule.
        if to_add_probas_outside_rules:
            return (1 / self.n_rules) * (y_pred_probas)
        else:
            scaling_coeffs = y_pred_probas.sum(axis=1)
            y_pred_probas = y_pred_probas / np.array([scaling_coeffs,scaling_coeffs,scaling_coeffs]).T
            return y_pred_probas

    def predict(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        y_pred_probas = self.predict_proba(
            X=X, to_add_probas_outside_rules=to_add_probas_outside_rules
        )
        y_pred_numeric = np.argmax(y_pred_probas, axis=1)
        if self.type_target != int:
            y_pred = y_pred_numeric.copy().astype(self.type_target)
            for i, cls in zip(self.classes_):
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
    def fit_forest_rules_regressor(self, X, y, all_possible_rules_list, p0=0.0):
        all_possible_rules_list_str = [
            str(elem) for elem in all_possible_rules_list
        ]  # Trick for np.unique
        unique_str_rules, indices_rules, count_rules = np.unique(
            all_possible_rules_list_str, return_counts=True, return_index=True
        )  # get the unique rules and count
        proportions_count = count_rules / len(
            count_rules
        )  # Get frequency of each rules
        proportions_count_sort = -np.sort(
            -proportions_count
        )  # Sort rules frequency by descending order
        proportions_count_sort_indices = np.argsort(
            -count_rules
        )  # Sort rules coubnt by descending order (same results as proportions)
        n_rules_to_keep = (
            proportions_count_sort > p0
        ).sum()  ## not necssary to sort proportions_count...
        all_possible_rules_list = [ 
            eval(unique_str_rules[i])
            for i in proportions_count_sort_indices[:n_rules_to_keep]
        ]#all possible rules reindexed 
        #### APPLY POST TREATMEANT : remove redundant rules
        #res = self.paths_filter_2depth(paths=all_possible_rules_list, proba=proportions_count_sort, num_rule=25)
        res=self.paths_filtering_2d(paths=all_possible_rules_list, proba=proportions_count_sort, num_rule=25)
        self.all_possible_rules_list = res['paths']
        self.n_rules = len(self.all_possible_rules_list)
        # list_mask_by_rules = []
        list_output_by_rules = []
        list_output_outside_by_rules = []
        gamma_array = np.zeros((X.shape[0], self.n_rules))
        for rule_number, current_rules in enumerate(
            self.all_possible_rules_list
        ):
            # for loop for getting all the values in train (X) passing the rules
            list_mask = []
            for j in range(
                len(current_rules)
            ):  ## iteraation on each signle rule of the potentail multiple rule
                dimension, treshold, sign = self.from_rules_to_constraint(
                    rule=current_rules[j]
                )
                mask = self.generate_single_rule_mask(
                    X=X, dimension=dimension, treshold=treshold, sign=sign
                )  # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(and_, list_mask)
            y_train_rule = y[final_mask]
            y_train_outside_rule = y[~final_mask]

            if len(y_train_rule) == 0:
                output_value = 0
            else:
                output_value = np.mean(y_train_rule)

            if len(y_train_outside_rule) == 0:
                output_outside_value = 0
            else:
                output_outside_value = np.mean(y_train_outside_rule)

            list_output_by_rules.append(output_value)
            list_output_outside_by_rules.append(output_outside_value)

            gamma_array[final_mask, rule_number ] = output_value
            gamma_array[ ~final_mask, rule_number] = output_outside_value

            # list_mask_by_rules.append(final_mask) # uselesss

        # self.list_mask_by_rules = list_mask_by_rules
        self.list_probas_by_rules = list_output_by_rules
        self.list_probas_outside_by_rules = list_output_outside_by_rules
        self.type_target = y.dtype

        ## final predictor fitting :
        #self.ridge = Ridge(
        #    alpha=1, fit_intercept=True, positive=True, random_state=self.random_state
        #)
        self.ridge = RidgeCV(
            alphas=np.arange(0.01,1,0.1),cv=5,scoring='neg_mean_squared_error', fit_intercept=True,
        )
        ones_vector = np.ones((len(gamma_array),1))  # Vector of ones
        gamma_array = np.hstack((gamma_array,ones_vector))
        self.ridge.fit(gamma_array, y)
        # self.gamma_array = gamma_array

    def predict_regressor(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        # y_pred = np.zeros((len(X),self.n_classes_))
        gamma_array = np.zers((X.shape[0], self.n_rules))
        for indice in range(self.n_rules):
            current_rules = self.all_possible_rules_list[indice]
            list_mask = []
            for j in range(
                len(current_rules)
            ):  ## iteration on each signle rule of the potentail multiple rule
                dimension, treshold, sign = self.from_rules_to_constraint(
                    rule=current_rules[j]
                )
                mask = self.generate_single_rule_mask(
                    X=X, dimension=dimension, treshold=treshold, sign=sign
                )  # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(
                and_, list_mask
            )  ## test samples that verify the current multiple rule
            gamma_array[indice, final_mask] = self.list_output_by_rules[indice]
            if to_add_probas_outside_rules:  # ERWAN TIPS !!
                gamma_array[indice, final_mask] = self.list_output_outside_by_rules[
                    indice
                ]

        ones_vector = np.ones((len(gamma_array),1))  # Vector of ones
        gamma_array = np.hstack((gamma_array,ones_vector))
        y_pred = self.ridge.predict(gamma_array)

        return y_pred
    
    #######################################################
    ################ Fit main classiifer   ################
    #######################################################

    def fit_main_classifier(self, X, y, quantile=10, sample_weight=None,to_not_binarize_colindex=None):
        """
        fit method for SirusMixin. 
        
        """
        if to_not_binarize_colindex is None:
            X_bin = X.copy()
            list_quantile = [
            np.percentile(X_bin, q=i * quantile, axis=0)
            for i in range(int((100 // quantile) + 1))
            ]
            array_quantile = np.array(list_quantile)
            for dim in range(X.shape[1]):
                out = np.searchsorted(array_quantile[:, dim], X_bin[:, dim], side="left")
                X_bin[:, dim] = array_quantile[out, dim]
        else :
            list_quantile = [
            np.percentile(X_bin[:,~to_not_binarize_colindex], q=i * quantile, axis=0)
            for i in range(int((100 // quantile) + 1))
            ]
            array_quantile = np.array(list_quantile)
            for dim in range(X.shape[1]):
                if dim not in to_not_binarize_colindex:
                    out = np.searchsorted(array_quantile[:, dim], X_bin[:, dim], side="left")
                    X_bin[:, dim] = array_quantile[out, dim]
        super().fit(
            X_bin,
            y,
            sample_weight=sample_weight,
        )
        self.array_quantile_ = array_quantile

    #######################################################
    ################## Print rules   ######################
    #######################################################

    def print_rules(self, max_rules=10):
        if self.feature_names_in_ is None:
            self.feature_names_in_ = np.arange(self.n_features_in_)
        for indice in range(max_rules):
            current_rules = self.all_possible_rules_list[indice]
            print("########")
            print("Rules {} ".format(indice))
            for j in range(len(current_rules)):
                dimension, treshold, sign = self.from_rules_to_constraint(
                    rule=current_rules[j]
                )
                if sign == "L":
                    sign = "<="
                else:
                    sign = ">"
                print("       &( {} {} {} )".format(self.feature_names_in_[dimension], sign, treshold))
    
    def show_rules(self, max_rules=9, target_class_index=1):
        if not hasattr(self, 'all_possible_rules_list') or \
           not hasattr(self, 'list_probas_by_rules') or \
           not hasattr(self, 'list_probas_outside_by_rules'):
            print("Model does not have the required rule attributes. Ensure it's fitted.")
            return

        rules_all = self.all_possible_rules_list
        probas_if_true_all = self.list_probas_by_rules
        probas_if_false_all = self.list_probas_outside_by_rules

        if not (len(rules_all) == len(probas_if_true_all) == len(probas_if_false_all)):
            print("Error: Mismatch in lengths of rule attributes.")
            return

        num_rules_to_show = min(max_rules, len(rules_all))
        if num_rules_to_show == 0:
            print("No rules to display.")
            return

        # Attempt to build/use feature mapping
        feature_mapping = None
        if hasattr(self, 'feature_names_in_'): # Standard scikit-learn attribute
            # Create a mapping from index to name if feature_names_in_ is a list
            feature_mapping = {i: name for i, name in enumerate(self.feature_names_in_)}
        elif hasattr(self, 'feature_names_'): # Custom attribute for feature names
            if isinstance(self.feature_names_, dict):
                feature_mapping = self.feature_names_ # Assumes it's already index:name
            elif isinstance(self.feature_names_, list):
                feature_mapping = {i: name for i, name in enumerate(self.feature_names_)}
        # If no mapping, column_name will default to using indices.

        base_ps_text = ""
        if probas_if_false_all and probas_if_false_all[0] and len(probas_if_false_all[0]) > target_class_index:
            avg_outside_target_probas = [
                p[target_class_index] for p in probas_if_false_all if p and len(p) > target_class_index
            ]
            if avg_outside_target_probas:
                estimated_avg_target_prob = np.mean(avg_outside_target_probas) * 100
                base_ps_text = (f"Estimated average rate for target class {target_class_index} (from 'else' clauses) p_s = {estimated_avg_target_prob:.0f}%.\n"
                                f"(Note: True average rate should be P(Class={target_class_index}) from training data).\n")
        
        print(base_ps_text)
        header_condition = "IF Condition"
        header_then = f"     THEN P(C{target_class_index})" 
        header_else = f"     ELSE P(C{target_class_index})" 
        
        max_condition_len = 0
        condition_strings_for_rules = []

        for i in range(num_rules_to_show):
            current_rule_conditions = rules_all[i]
            condition_parts_str = []
            for j in range(len(current_rule_conditions)):
                
                dimension, treshold, sign_internal = self.from_rules_to_constraint(
                    rule=current_rule_conditions[j]
                )
                
                column_name = f"Feature[{dimension}]" # Default if no mapping
                if feature_mapping and dimension in feature_mapping:
                    column_name = feature_mapping[dimension]
                elif feature_mapping and isinstance(dimension, str) and dimension in feature_mapping.values():
                    # If dimension is already a name that's in the mapping's values (less common for index)
                    column_name = dimension 
                
                sign_display = "<=" if sign_internal == "L" else ">"
                treshold_display = f"{treshold:.2f}" if isinstance(treshold, float) else str(treshold)
                condition_parts_str.append(f"{column_name} {sign_display} {treshold_display}")
            
            full_condition_str = " & ".join(condition_parts_str)
            condition_strings_for_rules.append(full_condition_str)
            if len(full_condition_str) > max_condition_len:
                max_condition_len = len(full_condition_str)

        condition_col_width = max(max_condition_len, len(header_condition)) + 2

        print(f"{header_condition:<{condition_col_width}} {header_then:<15} {header_else:<15}")
        print("-" * (condition_col_width + 15 + 15 + 2))

        for i in range(num_rules_to_show):
            condition_str_formatted = condition_strings_for_rules[i]
            
            prob_if_true_list = probas_if_true_all[i]
            prob_if_false_list = probas_if_false_all[i]

            then_val_str = "N/A"
            else_val_str = "N/A"

            if prob_if_true_list and len(prob_if_true_list) > target_class_index:
                p_s_if_true = prob_if_true_list[target_class_index] * 100
                then_val_str = f"{p_s_if_true:.0f}%"
            
            if prob_if_false_list and len(prob_if_false_list) > target_class_index:
                p_s_if_false = prob_if_false_list[target_class_index] * 100
                else_val_str = f"{p_s_if_false:.0f}%"

            print(f"if   {condition_str_formatted:<{condition_col_width - 5}} then {then_val_str:<12} else {else_val_str:<12}")