from functools import reduce
from operator import and_
from math import isclose
from itertools import compress

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.ensemble._forest import ForestClassifier,ForestRegressor
from sklearn.tree import _tree
from sklearn.utils._param_validation import StrOptions
from sklearn.tree import _splitter
import sklearn.tree._classes
from sklearn.linear_model import Ridge
from sklearn.ensemble._gb import set_huber_delta, _update_terminal_regions
from sklearn._loss.loss import HuberLoss
from sklearn.tree import DecisionTreeRegressor
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
        Uses constraints of a single rule to generatye the associated mask for data set X.

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
    
    def generate_mask_of_several_rules(self,X,rules):
        list_mask=[]
        for j in range(len(rules)):
            dimension,treshold,sign = self.from_rules_to_constraint(rule=rules[j])
            mask = self.generate_single_rule_mask(X=X,dimension=dimension,treshold=treshold,sign=sign)
            list_mask.append(mask)
        final_mask = reduce(and_, list_mask)
        return final_mask
    
    def detect_redundant_rules(self,all_possible_rules_list,random_state=None,verbose=0):
        np.random.seed(random_state)
        n_uniform = 10000
        X_uniform = np.array([np.random.uniform(low=self.array_quantile_[0,i]-1,high=self.array_quantile_[-1,i]+1,size=(n_uniform)) for i in range(len(self.array_quantile_[0,:]))]).T
        #rules_to_keep = np.zeros(len(all_possible_rules_list),dtype=int)
        rules_to_keep = []
        n_rules = len(all_possible_rules_list)
        all_possible_rules_list.reverse() # reverse in order to drop the rules associated to lowest values of frequency first
        for i,rules in enumerate(all_possible_rules_list):
            bool_value_current_rule = 1
            for j,second_rules in enumerate(all_possible_rules_list[i+1:]):
                if i==j:
                    continue
                else:
                    
                    mask_rules = self.generate_mask_of_several_rules(X_uniform,rules) ## First rule proba
                    X_uniform_valid = X_uniform[mask_rules]
                    probas_rules = len(X_uniform_valid) / n_uniform

                    mask_second_rules = self.generate_mask_of_several_rules(X_uniform,second_rules) ## second rule proba
                    X_uniform_valid = X_uniform[mask_second_rules]
                    probas_second_rules = len(X_uniform_valid) / n_uniform

                    #mask = self.generate_mask_of_several_rules(X_uniform,rules+second_rules) ## second rule proba
                    mask = (mask_rules*mask_second_rules)
                    #mask[mask>1] = 1
                    X_uniform_valid = X_uniform[mask]
                    probas_intersection_borth_rules = len(X_uniform_valid) / n_uniform
                    if verbose==1:
                        print('**')
                        print('i,j = ',i,j)
                        print('rules :',rules)
                        print('second_rules :',second_rules)
                        sum_array = (mask_rules*mask_second_rules)
                        #sum_array[sum_array>1] = 1
                        #print(len(mask))
                        #print('mask.sum()',mask.sum())
                        #print('(mask == sum_array).sum() :',(mask == sum_array).sum())
                        #print('(mask_rules==mask_second_rules).sum() :', (mask_rules==mask_second_rules).sum())
                        print('probas_rules : ',probas_rules)
                        print('probas_second_rules :', probas_second_rules)
                        print('probas_intersection_borth_rules :',probas_intersection_borth_rules)
                        print('probas_rules*probas_second_rules :', probas_rules*probas_second_rules)
                        print('test egalité :',isclose(probas_intersection_borth_rules, probas_rules*probas_second_rules,rel_tol=1e-3,abs_tol=1e-3))
                    if not isclose(probas_intersection_borth_rules, probas_rules*probas_second_rules,rel_tol=1e-3,abs_tol=1e-3):
                        bool_value_current_rule=0
                        break
            if verbose==1:
                print('Curr bool_value_current_rule :', bool_value_current_rule)  
            rules_to_keep.append(bool_value_current_rule)
        
        rules_to_keep.reverse() # reverse the list because the original one was revearsed also
        return rules_to_keep
                
    def paths_filter_2(self,paths, proba, num_rule):
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
        split_gen = []
        ind_max = len(paths)
        ind = 0
        num_rule_temp = 0

        while num_rule_temp < num_rule and ind < ind_max:
            #path_ind = copy.deepcopy(paths[ind])
            path_ind = paths[ind]
            
            ## Remove empty split (variable 0)
            #split_var = [split[0] for split in path_ind]
            #if 0 in split_var:
            #    path_ind = [split for split in path_ind if split[0] != 0]
            
            # Format rule with 2 cuts on same variable and direction
            if len(path_ind) == 2:
                if (path_ind[0][0] == path_ind[1][0]) and (path_ind[0][2] == path_ind[1][2]):
                    if path_ind[0][1] > path_ind[1][1]:
                        path_ind = [path_ind[0]] if path_ind[0][2] == 1 else [path_ind[1]]
                    else:
                        path_ind = [path_ind[1]] if path_ind[0][2] == 1 else [path_ind[0]]
                    paths[ind] = path_ind
            
            split_ind = [split[:2] for split in path_ind]
            d = len(path_ind)
            
            # Avoid duplicates
            if split_ind not in split_gen:
                paths_ftr.append(path_ind)
                proba_ftr.append(proba[ind])
                num_rule_temp = len(paths_ftr)
                
                # Add generated interaction
                if d <= 2:
                    if d == 1:
                        split_gen_temp = [split_ind]
                        split_gen += split_gen_temp
                    if d == 2:
                        # get index of rules involving any similar constraint
                        bool_ind = []
                        for path in paths_ftr:
                            if len(path) <= 2:
                                bools = [
                                    any([(x[:2] == y[:2]) for y in path_ind])
                                    for x in path
                                ]
                                bool_ind.append([all(bools), any(bools)])
                            else:
                                bool_ind.append([False, False])
                        bool_all = [x[0] for x in bool_ind]
                        bool_any = [x[1] for x in bool_ind]
                        bool_mixed = [not a and b for a, b in zip(bool_all, bool_any)]
                        num_rule_all = sum(bool_all)
                        num_rule_any = sum(bool_any)
                        
                        if num_rule_all >= 2: #The currrent rule is of depth 2 and 
                        #involves to rules from path_tr. Thus, it is lineary dependant from the filtered rules paths_ftr.
                        # We add it to genrated rules only
                            split_gen.append(split_ind)
                            split_gen.extend([[split[:2]] for split in path_ind])


                        # combine path with paths_ftr
                        split_gen_temp = []
                        for j, mixed in enumerate(bool_mixed):
                            if mixed:
                                x = paths_ftr[j]
                                split_diff = [s for s in (x + path_ind) if s not in set(map(tuple, x)).intersection(map(tuple, path_ind))]
                                if len(split_diff) == 2 and split_diff[0][:2] == split_diff[1][:2]:
                                    split1 = [split_diff[0][:2]]
                                    if split1 not in split_gen:
                                        split_gen_temp.append([split_diff[0][:2]])
                        
                        # specific case: two splits on the same variable
                        if split_ind[0][0] == split_ind[1][0]:
                            bool_double = [
                                all([x in split_ind for x in split]) and len(split) == 1
                                for split in split_gen
                            ]
                            if any(bool_double):
                                for k, is_double in enumerate(bool_double):
                                    if is_double:
                                        split = split_gen[k]
                                        split_diff = [s for s in split_ind if s not in split]
                                        if len(split_diff) > 0:
                                            split_gen_temp.append(split_diff)
                        # Flatten and filter out None
                        split_gen_temp = [x for x in split_gen_temp if x is not None]
                        if split_gen_temp:
                            split_gen_1 = [x for x in split_gen if len(x) == 1]
                            more_temp = []
                            for split in split_gen_temp:
                                for split1 in split_gen_1:
                                    if len(split) == 1 and split[0][0] == split1[0][0] and split[0][1] != split1[0][1]:
                                        if split[0][1] > split1[0][1]:
                                            more_temp.append([split1 + split])
                                        else:
                                            more_temp.append([split + split1])
                            split_gen_temp += more_temp
                            # Remove already existing in split_gen
                            split_gen_temp = [x for x in split_gen_temp if x not in split_gen]
                            split_gen += split_gen_temp
            ind += 1

        return {'paths': paths_ftr, 'proba': proba_ftr}
    def paths_filter_2depth(self,paths, proba, num_rule):
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
        split_gen = []
        ind_max = len(paths)
        ind = 0
        num_rule_temp = 0

        while num_rule_temp < num_rule and ind < ind_max:
            #path_ind = copy.deepcopy(paths[ind])
            path_ind = paths[ind]
            
            split_ind = [split[:2] for split in path_ind]
            d = len(path_ind)
            
            # Avoid duplicates
            if d == 1:
                if split_ind not in split_gen:
                    paths_ftr.append(path_ind)
                    proba_ftr.append(proba[ind])
                    num_rule_temp = len(paths_ftr)
                split_gen_temp = [split_ind]
                split_gen += split_gen_temp
            if d == 2:
                list_bool_in_ftr = []
                for curr_split_ind in split_ind:
                    if [curr_split_ind] in split_gen:
                        list_bool_in_ftr.append(True)
                    else:
                        list_bool_in_ftr.append(False)
                    if (not all(list_bool_in_ftr)) and (split_ind not in split_gen):
                        paths_ftr.append(path_ind)
                        proba_ftr.append(proba[ind])
                        num_rule_temp = len(paths_ftr)
                split_gen.append(split_ind)
                if ([split_ind[0]] in split_gen) and ([split_ind[1]] not in split_gen): # if one of the sigle rule is already in split_gen 
                    # then trough alinear combination we can obtain the other one.
                    split_gen.append([split_ind[1]])
                if ([split_ind[1]] in split_gen) and ([split_ind[0]] not in split_gen):
                    split_gen.append([split_ind[0]])
                #split_gen.append([split_ind[0][:2]])
                #split_gen.append([split_ind[1][:2]])

            ind += 1

        return {'paths': paths_ftr, 'proba': proba_ftr}

    
    #######################################################
    ############ Classification fit and predict  ##########
    #######################################################

    def fit_forest_rules(self, X, y, all_possible_rules_list, p0=0.0,batch_size_post_treatment=None):
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
        #all_possible_rules_list = [
        #    all_possible_rules_list[i]
        #    for i in proportions_count_sort_indices[:n_rules_to_keep]
        #]#all possible rules reindexed 
        all_possible_rules_list = [
            eval(unique_str_rules[i])
            for i in proportions_count_sort_indices[:n_rules_to_keep]
        ]#all possible rules reindexed 

        print('n_rules before post-treatment : ', len(all_possible_rules_list))
        #### APPLY POST TREATMEANT : remove redundant rules

        print('25 all_possible_rules_list : ',all_possible_rules_list[:25])
        print('25 proportions_count_sort : ',proportions_count_sort[:25])
        res = self.paths_filter_2depth(paths=all_possible_rules_list, proba=proportions_count_sort, num_rule=25)
        self.all_possible_rules_list = res['paths']
        self.n_rules = len(self.all_possible_rules_list)
        #print('After all_possible_rules_list',res['paths'])
        #print('n_rules after post-treatment : ', self.n_rules)

        # list_mask_by_rules = []
        list_probas_by_rules = []
        list_probas_outside_by_rules = []
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
            y_train_outside_rule = y[~final_mask]

            list_probas = []
            list_probas_outside_rules = []
            for cl in range(self.n_classes_):  # iteration on each class of the target
                if len(y_train_rule) == 0:
                    curr_probas = 0
                else:
                    curr_probas = len(y_train_rule[y_train_rule == cl]) / len(
                        y_train_rule
                    )
                list_probas.append(curr_probas)
                curr_probas_outside_rules = len(
                    y_train_outside_rule[y_train_outside_rule == cl]
                ) / len(y_train_outside_rule)
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

        return (1 / self.n_rules) * y_pred_probas

    def predict(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        y_pred_probas = self.predict_proba(
            X=X, to_add_probas_outside_rules=to_add_probas_outside_rules
        )
        y_pred_numeric = np.argmax(y_pred_probas, axis=1)
        if self.type_target != int:
            y_pred = y_pred_numeric.copy().astype()
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
    def fit_forest_rules_regressor(self, X, y, all_possible_rules_list, p0=0.0,batch_size_post_treatment=None):
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
        res = self.paths_filter_2depth(paths=all_possible_rules_list, proba=proportions_count_sort, num_rule=25)
        self.all_possible_rules_list = res['paths']
        self.n_rules = len(self.all_possible_rules_list)
        # list_mask_by_rules = []
        list_output_by_rules = []
        list_output_outside_by_rules = []
        gamma_array = np.zeros((X.shape[0], n_rules_to_keep))
        for rule_number, indice in enumerate(
            proportions_count_sort_indices[:n_rules_to_keep]
        ):
            # for loop for getting all the values in train (X) passing the rules
            current_rules = all_possible_rules_list[indice]
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
        self.ridge = Ridge(
            alpha=1.0, fit_intercept=True, positive=True, random_state=self.random_state
        )
        self.ridge.fit(X, y)
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

            y_pred = self.ridge.predict(gamma_array)

        return y_pred
    
    #######################################################
    ################ Fit main classiifer   ################
    #######################################################
    
    def fit_main_classifier(self, X, y, quantile=10, sample_weight=None):
        X_bin = X.copy()
        list_quantile = [
            np.percentile(X_bin, q=i * quantile, axis=0)
            for i in range(int((100 // quantile) + 1))
        ]
        array_quantile = np.array(list_quantile)
        for dim in range(X.shape[1]):
            out = np.searchsorted(array_quantile[:, dim], X_bin[:, dim], side="left")
            X_bin[:, dim] = array_quantile[out, dim]
        super().fit(
            X_bin,
            y,
            sample_weight=sample_weight,
        )
        self.array_quantile_ = array_quantile
        print('array_quantile : ', array_quantile)

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



class SirusDTreeClassifier(SirusMixin, DecisionTreeClassifier):
    """
    SIRUS class applied with a DecisionTreeClassifier
    Parameters
    ----------

    """

    _parameter_constraints: dict = {**DecisionTreeClassifier._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True,batch_size_post_treatment=None):
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
            X, y, all_possible_rules_list, p0,batch_size_post_treatment
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

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True,batch_size_post_treatment=None):
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_:  ## extraction  of all trees rules
            tree = dtree.tree_
            all_possible_rules_list.extend(self.extract_single_tree_rules(tree))
        self.fit_forest_rules(X, y, all_possible_rules_list, p0,batch_size_post_treatment)


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
    ----------

    """

    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints}
    _parameter_constraints["splitter"] = [StrOptions({"best", "random", "quantile"})]



class SirusGBClassifier(SirusMixin, GradientBoostingClassifier):
    """
    SIRUS class applied with a RandomForestClassifier

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

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True,batch_size_post_treatment=None):
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = []
        for i in range(self.n_estimators_):  ## extraction  of all trees rules
            print('self.estimators_.shape', self.estimators_.shape)
            dtree = self.estimators_[i,0]  
            tree = dtree.tree_
            all_possible_rules_list.extend(self.extract_single_tree_rules(tree))
        self.fit_forest_rules_regressor(X, y, all_possible_rules_list, p0,batch_size_post_treatment)


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

    def fit(self, X, y, p0=0.0, quantile=10, sample_weight=None, check_input=True,batch_size_post_treatment=None):
        self.fit_main_classifier(X, y, quantile, sample_weight)
        all_possible_rules_list = []
        for i in range(self.n_outputs_):  ## extraction  of all trees rules
            dtree = self.estimators_[i]  
            tree = dtree.tree_
            all_possible_rules_list.extend(self.extract_single_tree_rules(tree))
        self.fit_forest_rules_regressor(X, y, all_possible_rules_list, p0,batch_size_post_treatment)


# TODO : filter redundant rules
# TODO : CV for ridge regressor ?
