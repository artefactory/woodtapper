from collections import Counter
from functools import reduce
from operator import and_

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree, export_text


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
    def __init__(self, feature=None,treshold=-1,side=None, node_id=-1, *children):
        
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
    def explore_tree_(self,node_id,side):
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
        if self.tree_.children_left[node_id] != _tree.TREE_LEAF: # possible to add a max_depth constraint exploration value
            id_left_child = self.tree_.children_left[node_id]
            id_right_child = self.tree_.children_right[node_id]
            children = [
                self.explore_tree_(id_left_child,'L'), # L for \leq
                self.explore_tree_(id_right_child,'R')
            ]
        else:
            return Node(feature=self.tree_.feature[node_id],treshold=self.tree_.threshold[node_id],side=side,node_id=node_id)
        
        return Node(self.tree_.feature[node_id],self.tree_.threshold[node_id],side,node_id,*children)
    def construct_longest_paths_(self,root):
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
        stack = [(root,0)]  # start with the root node id (0) and its depth (0)
        depth = 0
        while len(stack) > 0:
            curr_rule,indice_in_tree_struct = stack.pop()
            is_split_node = (curr_rule.feature != -2)
        
            if is_split_node:
                rule_left = (curr_rule.feature,curr_rule.treshold,'L')
                rule_right = (curr_rule.feature,curr_rule.treshold,'R')
                common_path_rules = tree_structure[indice_in_tree_struct].copy()
                common_path_rules.append(rule_right)
                tree_structure.append(common_path_rules) ## RIGHT : Added at the end
                tree_structure[indice_in_tree_struct].append(rule_left) ## LEFT  : Added depending on indice_in_tree_struct
        
                stack.append((curr_rule.children[0],indice_in_tree_struct))
                stack.append(( curr_rule.children[1],len(tree_structure)-1 ))
            else:
                #print('c')
                continue
        return tree_structure

    def split_sub_rules_(self,path,is_removing_singleton=False):
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
        for j in range(max_size_curr_path-int_to_add):
            list_sub_path.append(path[:(max_size_curr_path-j)])
        return list_sub_path
    
    def generate_all_possible_rules_(self,tree_structure):
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
            
            ## We take all the rules strating from a head node
            for k in range(max_size_curr_path):
                list_sub_path = self.split_sub_rules_(curr_path[k:],is_removing_singleton=False)
                all_paths_list.extend(list_sub_path)
        
            ## More complexe cases : internal rules
            if max_size_curr_path ==1:
                continue
            else:
                curr_path_size_pair = ( (max_size_curr_path%2)==0)
                if curr_path_size_pair: ## PAIRS       
                    for k in range(1, (max_size_curr_path//2) ):
                        list_sub_path = self.split_sub_rules_(curr_path[k:max_size_curr_path-k],is_removing_singleton=True)
                        all_paths_list.extend(list_sub_path)
                else: ## IMPAIRS
                    for k in range(1,(max_size_curr_path//2) ):
                        list_sub_path = self.split_sub_rules_(curr_path[k:max_size_curr_path-k],is_removing_singleton=True)
                        all_paths_list.extend(list_sub_path)
                        if k == (max_size_curr_path//2): #case odd last 
                            list_sub_path = self.split_sub_rules_(curr_path[k:max_size_curr_path-(k-1)],is_removing_singleton=True)
                            all_paths_list.extend(list_sub_path)
        return all_paths_list
    
    def from_rules_to_constraint(self,rule):
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
        return dimension,treshold,sign
    
    def generate_single_rule_mask(self,X,dimension,treshold,sign):
        """
        Uses constraints of a single rule to generatye the associated mask for data set X.

        Parameters
        ----------
        """
        if sign=='L':
            return (X[:,dimension]<=treshold) #.mean()
        else:
            return (X[:,dimension]>treshold)#.mean()
    
    def fit(self, X, y,p0=0.0,quantile=10,sample_weight=None):
        """
        Fit method for SirusMixin.
        """
        X_bin = X.copy()
        list_quantile = [np.percentile(X_bin,q=i*quantile,axis=0) for i in range(int((100//quantile)+1))]
        array_quantile = np.array(list_quantile )
        for dim in range(X.shape[1]):
            out = np.searchsorted(array_quantile[:,dim], X_bin[:,dim],side='left')
            X_bin[:,dim] = array_quantile[out,dim]
        super().fit(X=X_bin, y=y, sample_weight=sample_weight)

        root = self.explore_tree_(0,'Root') ## Root node
        tree_structure = self.construct_longest_paths_(root) ## generate the tree structure with Node instances
        all_possible_rules_list = self.generate_all_possible_rules_(tree_structure) # Explre the tree structure to extract the longest rules (rules from root to a leaf)
        all_possible_rules_list_str = [str(elem) for elem in all_possible_rules_list] # Trick for np.unique
        unique_str_rules,indices_rules,count_rules = np.unique(all_possible_rules_list_str,return_counts=True,return_index=True) # get the unique rules and count
        proportions_count = (count_rules / len(count_rules)) # Get frequency of each rules
        proportions_count_sort = -np.sort(-proportions_count) # Sort rules frequency by descending order 
        proportions_count_sort_indices = np.argsort(-count_rules) # Sort rules coubnt by descending order (same results as proportions)
        n_rules_to_keep = (proportions_count_sort > p0).sum() ## not necssary to sort proportions_count...
        
        list_mask_by_rules = []
        list_probas_by_rules = []
        list_probas_outside_by_rules = []
        # APPLY POST TREATMEANT HERE on count_sort_ind[:p0]
        for indice in proportions_count_sort_indices[:n_rules_to_keep]:
            #for loop for getting all the values in train (X) passing the rules
            current_rules = all_possible_rules_list[indice]
            list_mask=[]
            for j in range(len(current_rules)): ## iteraation on each signle rule of the potentail multiple rule
                dimension,treshold,sign = self.from_rules_to_constraint(rule=current_rules[j])
                mask = self.generate_single_rule_mask(X=X,dimension=dimension,treshold=treshold,sign=sign) # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(and_, list_mask)
            y_train_rule = y[final_mask]
            y_train_outside_rule = y[~final_mask]
            list_probas =[]
            list_probas_outside_rules = []
            for cl in range(self.n_classes_): #iteration on each class of the target
                if len(y_train_rule)==0:
                    curr_probas=0
                else:
                    curr_probas = len(y_train_rule[y_train_rule==cl]) / len(y_train_rule)
                list_probas.append(curr_probas)
                curr_probas_outside_rules = len(y_train_outside_rule[y_train_outside_rule==cl]) / len(y_train_outside_rule)
                list_probas_outside_rules.append(curr_probas_outside_rules)

            list_mask_by_rules.append(final_mask)
            list_probas_by_rules.append(list_probas)
            list_probas_outside_by_rules.append(list_probas_outside_rules)

        self.all_possible_rules_list = [all_possible_rules_list[i] for i in  proportions_count_sort_indices[:n_rules_to_keep]]
        self.n_rules = len(self.all_possible_rules_list)
        #self.list_mask_by_rules = list_mask_by_rules
        self.list_probas_by_rules = list_probas_by_rules
        self.list_probas_outside_by_rules = list_probas_outside_by_rules
        self.type_target = y.dtype 
    
    
    def predict_proba(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        y_pred_probas = np.zeros((len(X),self.n_classes_))
        for indice in range(self.n_rules):
            current_rules = self.all_possible_rules_list[indice]
            list_mask=[]
            for j in range(len(current_rules)): ## iteraation on each signle rule of the potentail multiple rule
                dimension,treshold,sign = self.from_rules_to_constraint(rule=current_rules[j])
                mask = self.generate_single_rule_mask(X=X,dimension=dimension,treshold=treshold,sign=sign) # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(and_, list_mask) ## test samples that verify the current multiple rule
            y_pred_probas[final_mask] +=  self.list_probas_by_rules[indice] ## add the asociated rule probability
            
            if to_add_probas_outside_rules: #ERWAN TIPS !!
                y_pred_probas[~final_mask] +=  self.list_probas_outside_by_rules[indice]## If the rule is not verified we add the probas of the training samples not verifying the rule.
        
        return  (1/self.n_rules)*y_pred_probas

    def predict(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        y_pred_probas = self.predict_proba(X=X,to_add_probas_outside_rules=to_add_probas_outside_rules)
        y_pred_numeric = np.argmax(y_pred_probas,axis=1) 
        if self.type_target != int:
            y_pred=y_pred_numeric.copy().astype()
            for i,cls in zip(self.classes_):
                y_pred[y_pred_numeric == i] = cls
            return y_pred.ravel().reshape(-1,)
        else:
            return y_pred_numeric.ravel().reshape(-1,)


class SirusDTreeClassifier(SirusMixin, DecisionTreeClassifier): 
    """
    SIRUS class applied with a DecisionTreeClassifier
    Parameters
    ----------

    """

class SirusRFClassifier(SirusMixin, RandomForestClassifier): #DecisionTreeClassifier
    """
    SIRUS class applied with a RandomForestClassifier

    """


#TODO : Define a splitter that split on train data values (and no longer on the mean of two values)
#TODO : DecisionTreeClassifierAbd that uses the previous splitter
#TODO : RandomForestClassifierAbd that uses the previous splitter
#TODO : Adapt SirusMixin for RandomForestClassifierAbd