from collections import Counter
from functools import reduce
from operator import and_

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import _tree, export_text

from ._QuantileSplitter import QuantileBestSplitter
from sklearn.utils._param_validation import  StrOptions
from sklearn.tree import _splitter
import sklearn.tree._classes
sklearn.tree._classes.DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.BestSplitter,"quantile":QuantileBestSplitter}

from sklearn.linear_model import Ridge

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
    def explore_tree_(self,node_id,side,tree):
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
        if tree.children_left[node_id] != _tree.TREE_LEAF: # possible to add a max_depth constraint exploration value
            id_left_child = tree.children_left[node_id]
            id_right_child = tree.children_right[node_id]
            children = [
                self.explore_tree_(id_left_child,'L',tree), # L for \leq
                self.explore_tree_(id_right_child,'R',tree)
            ]
        else:
            return Node(feature=tree.feature[node_id],treshold=tree.threshold[node_id],side=side,node_id=node_id)
        
        return Node(tree.feature[node_id],tree.threshold[node_id],side,node_id,*children)
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
        
    def fit_main_classifier(self, X, y, quantile=10, sample_weight=None):
        X_bin = X.copy()
        list_quantile = [np.percentile(X_bin,q=i*quantile,axis=0) for i in range(int((100//quantile)+1))]
        array_quantile = np.array(list_quantile )
        for dim in range(X.shape[1]):
            out = np.searchsorted(array_quantile[:,dim], X_bin[:,dim],side='left')
            X_bin[:,dim] = array_quantile[out,dim]
        super().fit(
            X_bin,
            y,
            sample_weight=sample_weight,
        )
    
    def extract_single_tree_rules(self,tree):
        """
        Fit method for SirusMixin.
        """
        root = self.explore_tree_(0,'Root',tree) ## Root node
        tree_structure = self.construct_longest_paths_(root) ## generate the tree structure with Node instances
        all_possible_rules_list = self.generate_all_possible_rules_(tree_structure) # Explre the tree structure to extract the longest rules (rules from root to a leaf)
        return all_possible_rules_list
        
        
    def fit_forest_rules(self, X, y,all_possible_rules_list,p0=0.0):
        all_possible_rules_list_str = [str(elem) for elem in all_possible_rules_list] # Trick for np.unique
        unique_str_rules,indices_rules,count_rules = np.unique(all_possible_rules_list_str,return_counts=True,return_index=True) # get the unique rules and count
        proportions_count = (count_rules / len(count_rules)) # Get frequency of each rules
        proportions_count_sort = -np.sort(-proportions_count) # Sort rules frequency by descending order 
        proportions_count_sort_indices = np.argsort(-count_rules) # Sort rules coubnt by descending order (same results as proportions)
        n_rules_to_keep = (proportions_count_sort > p0).sum() ## not necssary to sort proportions_count...
        list_mask_by_rules = []
        list_probas_by_rules = []
        list_probas_outside_by_rules = []
        #### APPLY POST TREATMEANT HERE on count_sort_ind[:n_rules_to_keep] ####
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

            #list_mask_by_rules.append(final_mask) # uselesss
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
            for j in range(len(current_rules)): ## iteration on each signle rule of the potentail multiple rule
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
        
    ################################
    ######### Regressor ############
    ################################    
    def fit_forest_rules_regressor(self, X, y,all_possible_rules_list,p0=0.0):
        all_possible_rules_list_str = [str(elem) for elem in all_possible_rules_list] # Trick for np.unique
        unique_str_rules,indices_rules,count_rules = np.unique(all_possible_rules_list_str,return_counts=True,return_index=True) # get the unique rules and count
        proportions_count = (count_rules / len(count_rules)) # Get frequency of each rules
        proportions_count_sort = -np.sort(-proportions_count) # Sort rules frequency by descending order 
        proportions_count_sort_indices = np.argsort(-count_rules) # Sort rules coubnt by descending order (same results as proportions)
        n_rules_to_keep = (proportions_count_sort > p0).sum() ## not necssary to sort proportions_count...
        list_mask_by_rules = []
        list_output_by_rules = []
        list_output_outside_by_rules = []
        gamma_array = np.zers((X.shape[0],n_rules_to_keep))
        #### APPLY POST TREATMEANT HERE on count_sort_ind[:n_rules_to_keep] ####
        for rule_number,indice in enumerate(proportions_count_sort_indices[:n_rules_to_keep]):
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
    
            if len(y_train_rule)==0:
                output_value = 0
            else:
                output_value = np.mean(y_train_rule)

            if len(y_train_outside_rule)==0:
                output_outside_value = 0
            else:
                output_outside_value = np.mean(y_train_outside_rule)

            list_output_by_rules.append(output_value)
            list_output_outside_by_rules.append(output_outside_value)
            
            gamma_array[rule_number,final_mask] = output_value
            gamma_array[rule_number,~final_mask] = output_value

            #list_mask_by_rules.append(final_mask) # uselesss

        ## all_possible_rules_list reindexed
        self.all_possible_rules_list = [all_possible_rules_list[i] for i in  proportions_count_sort_indices[:n_rules_to_keep]]
        self.n_rules = len(self.all_possible_rules_list)
        #self.list_mask_by_rules = list_mask_by_rules
        self.list_probas_by_rules = list_output_by_rules
        self.list_probas_outside_by_rules = list_output_outside_by_rules
        self.type_target = y.dtype 

        self.ridge = Ridge(alpha=1.0,fit_intercept=True, positive=True, random_state=self.random_stat)
        self.ridge.fit(X,y)
        #self.gamma_array = gamma_array
    
    def predict_regressor(self, X, to_add_probas_outside_rules=True):
        """
        predict_proba method for SirusMixin.
        """
        #y_pred = np.zeros((len(X),self.n_classes_))
        gamma_array = np.zers((X.shape[0],self.n_rules))
        for indice in range(self.n_rules):
            current_rules = self.all_possible_rules_list[indice]
            list_mask=[]
            for j in range(len(current_rules)): ## iteration on each signle rule of the potentail multiple rule
                dimension,treshold,sign = self.from_rules_to_constraint(rule=current_rules[j])
                mask = self.generate_single_rule_mask(X=X,dimension=dimension,treshold=treshold,sign=sign) # I do it on X and not on X_bin
                list_mask.append(mask)
            final_mask = reduce(and_, list_mask) ## test samples that verify the current multiple rule
            gamma_array[indice,final_mask] = self.list_output_by_rules[indice]
            if to_add_probas_outside_rules: #ERWAN TIPS !!
                gamma_array[indice,final_mask] =  self.list_output_outside_by_rules[indice]

            y_pred = self.ridge.predict(gamma_array)
        
        return  y_pred
    
        
    def print_rules(self,max_rules=10):
        for indice in range(max_rules):
            current_rules = self.all_possible_rules_list[indice]
            print("########")
            print('Rules {} '.format(indice))
            for j in range(len(current_rules)):
                dimension,treshold,sign = self.from_rules_to_constraint(rule=current_rules[j])
                if sign=='L':
                    sign='<='
                else:
                    sign='>'
                print("       &( X[:,{}] {} {} )".format(dimension,sign,treshold))


class SirusDTreeClassifier(SirusMixin, DecisionTreeClassifier): 
    """
    SIRUS class applied with a DecisionTreeClassifier
    Parameters
    ----------

    """
   
    _parameter_constraints: dict = {
        **DecisionTreeClassifier._parameter_constraints
    }
    _parameter_constraints["splitter"] = [StrOptions({"best", "random","quantile"})]

    def fit(self, X, y,p0=0.0,quantile=10, sample_weight=None, check_input=True):
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
        self.fit_main_classifier(X, y,quantile,sample_weight)
        all_possible_rules_list = self.extract_single_tree_rules(self.tree_)
        self.fit_forest_rules(X, y,all_possible_rules_list,p0) ## Checker que cx'est bien sur X et non le X_bin
        return self


class SirusRFClassifier(SirusMixin, RandomForestClassifier): #DecisionTreeClassifier
    """
    SIRUS class applied with a RandomForestClassifier

    """
    _parameter_constraints: dict = {
        **RandomForestClassifier._parameter_constraints
    }
    _parameter_constraints["splitter"] = [StrOptions({"best", "random","quantile"})]
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
        super(ForestClassifier,self).__init__(
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
                "splitter"
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

    def fit(self, X, y,p0=0.0,quantile=10, sample_weight=None, check_input=True):
        self.fit_main_classifier(X, y,quantile,sample_weight)
        all_possible_rules_list = []
        for dtree in self.estimators_: ## extraction  of all trees rules
            tree = dtree.tree_
            all_possible_rules_list.extend( self.extract_single_tree_rules(tree) )
        self.fit_forest_rules(X, y,all_possible_rules_list,p0)

######### Regressor ############

from sklearn.ensemble._gb import set_huber_delta,_update_terminal_regions
from sklearn._loss.loss import HuberLoss
from sklearn.tree import DecisionTreeRegressor

class SirusDTreeRegressor(SirusMixin, DecisionTreeRegressor): 
    """
    SIRUS class applied with a DecisionTreeClassifier
    Parameters
    ----------

    """
    _parameter_constraints: dict = {
        **DecisionTreeRegressor._parameter_constraints
    }
    _parameter_constraints["splitter"] = [StrOptions({"best", "random","quantile"})]

    def fit_forest_rules_regressor(self, X, y,p0=0.0,quantile=10, sample_weight=None, check_input=True):
        """Build a decision tree classifier from the training set (X, y).
        """
        self.fit_main_classifier(X, y,quantile,sample_weight)
        all_possible_rules_list = self.extract_single_tree_rules(self.tree_)
        self.fit_forest_rules(X, y,all_possible_rules_list,p0) ## Checker que cx'est bien sur X et non le X_bin
        return self

    def predict(self, X, to_add_probas_outside_rules=True):
        return self.predict_regressor(X, to_add_probas_outside_rules)

class SirusGBClassifier(SirusMixin, GradientBoostingClassifier): 
    """
    SIRUS class applied with a RandomForestClassifier

    """
    _parameter_constraints: dict = {
        **GradientBoostingClassifier._parameter_constraints
    }
    _parameter_constraints["splitter"] = [StrOptions({"best", "random","quantile"})]

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
            tree = SirusDTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter, ## ici
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

    def fit(self, X, y,p0=0.0,quantile=10, sample_weight=None, check_input=True):
        self.fit_main_classifier(X, y,quantile,sample_weight)
        all_possible_rules_list = []
        for i in range(self.n_estimators_): ## extraction  of all trees rules
            dtree = self.estimators_[i,1] ## Y 1-d
            tree = dtree.tree_
            all_possible_rules_list.extend( self.extract_single_tree_rules(tree) )
        self.fit_forest_rules(X, y,all_possible_rules_list,p0)


#TODO : filter redundant rules
#TODO : 
        


