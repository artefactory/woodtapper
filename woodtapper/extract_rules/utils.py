from functools import reduce

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import get_scorer


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
    children : list of Node or None
        Child Nodes if not a leaf node
    Returns
    ----------
    Node: Node
        The current Node instance
    """

    def __init__(self, feature=None, treshold=-1, side=None, node_id=-1, *children):
        self.node_id = node_id
        self.feature = feature
        self.treshold = treshold
        self.side = side
        self.children = children if children else []


def get_top_rules(rules_str, p0):
    """
    Get the top rules with frequency greater than p0
    Parameters
    ----------
    rules_str : list of str
        List of all possible rules in string format
    p0 : float
        Minimum frequency threshold
    Returns
    ----------
    rules_ : list of list of tuples
        List of all possible rules in tuple format
    rules_freq_ : list of float
        List of frequencies corresponding to the rules
    Raises
    ----------
    ValueError
        If no rule is found with the given p0 value
    """
    if len(rules_str) == 0 or len(rules_str[0]) == 0:
        raise ValueError("The input list of rules is empty.")
    unique_str_rules, indices_rules, frequence_rules = np.unique(
        rules_str, return_counts=True, return_index=True
    )  # get the unique rules and count
    frequence_rules = frequence_rules / frequence_rules.sum()  # convert to frequency
    unique_str_rules_and_freq = zip(
        unique_str_rules, frequence_rules
    )  # combine rules and frequency
    all_rules_sorted = sorted(
        unique_str_rules_and_freq, key=lambda x: x[1], reverse=True
    )  # sort by frequency
    all_possible_rules_and_freq_list = [
        (eval(unique_str_rule), freq)
        for unique_str_rule, freq in all_rules_sorted
        if freq > p0
    ]  # filter by p0
    if len(all_possible_rules_and_freq_list) == 0:
        if len(all_possible_rules_and_freq_list[0]) == 0:
            raise ValueError(
                "No rule found with the given p0 value. Try to decrease it."
            )
    rules_, rules_freq_ = zip(*all_possible_rules_and_freq_list)  # unzip
    return rules_, rules_freq_


def compute_staibility_criterion(model):
    """
    Compute the stability criterion for a given model.
    Parameters
    ----------
    model : SirusDTreeClassifier, SirusRFClassifier, SirusGBClassifier
        The model instance with attributes n_estimators and all_possible_rules_frequency_list
    Returns
    -------
    None
        Prints the stability criterion value
    """
    M = model.n_estimators
    list_p0 = np.arange(0.1, 1, 0.08)
    list_epsilon = []
    print("Computing stability criterion...")
    for p0_curr in list_p0:
        epsilon_numerator = np.sum(
            [
                binom.cdf(k=p0_curr * M, n=M, p=pm)
                * (1 - binom.cdf(k=p0_curr * M, n=M, p=pm))
                for pm in model.all_possible_rules_frequency_list
            ]
        )
        epsilon_denominator = np.sum(
            [
                (1 - binom.cdf(k=p0_curr * M, n=M, p=pm))
                for pm in model.all_possible_rules_frequency_list
            ]
        )
        epsilon = (
            epsilon_numerator / epsilon_denominator if epsilon_denominator > 0 else 0
        )
        list_epsilon.append(epsilon)
    print("***** \n Stability criterion value:", np.mean(list_epsilon), "\n*****")


def ridge_cv_positive(
    X,
    y,
    alphas=np.linspace(0, 1, 25),
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=None,
):
    """
    Cross-validate Ridge regression with positive=True manually,
    and return the best model fitted on all data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,)
        Target values.

    alphas : list or array-like
        List of alpha (regularization strength) values to test.

    scoring : str or callable
        A scoring function name (from sklearn) or a callable with signature
        `scoring(estimator, X_val, y_val)` returning a scalar score.

    cv : int or cross-validation generator, default=5
        Number of CV folds or a CV splitter.

    random_state : int, optional
        Random state for reproducibility (if cv is an integer).

    Returns
    -------
    best_alpha : float
        The alpha value that yields the best mean CV score.

    best_model : sklearn.linear_model.Ridge
        Ridge model (positive=True) trained on all data with best_alpha.

    results : dict
        Dictionary mapping alpha -> mean CV score.
    """

    # Handle scoring
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring

    # Handle CV splitter
    if isinstance(cv, int):
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    results = {}

    # Cross-validation loop
    for alpha in alphas:
        model = Ridge(
            alpha=alpha, fit_intercept=True, positive=True, random_state=random_state
        )
        scores = []

        for train_idx, val_idx in cv_splitter.split(X, y):
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            y_pred = m.predict(X[val_idx])
            score = scorer._score_func(y[val_idx], y_pred)
            scores.append(score)

        results[alpha] = np.mean(scores)

    # Select best alpha
    best_alpha = min(results, key=results.get)

    return best_alpha, results


#######################################################
##### Auxiliary function for path construction  #######
#######################################################
def _explore_tree(node_id, side, tree):
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
    if tree.children_left[node_id] != _tree.TREE_LEAF:
        # possible to add a max_depth constraint exploration value
        id_left_child = tree.children_left[node_id]
        id_right_child = tree.children_right[node_id]
        children = [
            _explore_tree(id_left_child, "L", tree),  # L for \leq
            _explore_tree(id_right_child, "R", tree),
        ]
        starting_node = Node(
            tree.feature[node_id], tree.threshold[node_id], side, node_id, *children)
    else:
        starting_node = Node(
            tree.feature[node_id], tree.threshold[node_id], side, node_id)

    return starting_node


def _construct_longest_paths(root):
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
            tree_structure.append(common_path_rules)
            ## RIGHT : Added at the end
            tree_structure[indice_in_tree_struct].append(rule_left)
            ## LEFT  : Added depending on indice_in_tree_struct

            stack.append((curr_rule.children[0], indice_in_tree_struct))
            stack.append((curr_rule.children[1], len(tree_structure) - 1))
        else:
            continue

    return tree_structure


def _split_sub_rules(path):
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
    for j in range(len(path), 0, -1):
        curr_path = path[:j]
        if len(curr_path) >= 2:
            list_sub_path.append(curr_path)

    return list_sub_path


def _generate_all_possible_rules(tree_structure):
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
        list_sub_path = _split_sub_rules(curr_path)
        all_paths_list.extend(list_sub_path)
        all_paths_list.append([curr_path[0]])

    return all_paths_list


def _extract_single_tree_rules(tree):
    """
    Extract all possible rules (single and multiple) from a single tree.
    Parameters
    ----------
    tree : sklearn DecisionTree instance
        The tree from which to extract rules.
    Returns
    ----------
    rules_ : list
        List of all possible rules (single and multiple) extracted from the tree.
    1. Explore the tree structure and create Node instances.
    2. Generate the tree structure with Node instances.
    3. Explore the tree structure to extract the longest rules (rules from root to a leaf).
    4. Generate all possible rules (single and multiple) from the tree structure.
    5. Return the list of all possible rules.
    """
    root = _explore_tree(0, "Root", tree)  ## Root node
    tree_structure = _construct_longest_paths(root)
    # generate the tree structure with Node instances
    if len(tree_structure[0]) == 0 and root.feature == -2:
        # case where root node is also a leaf (-2 means leaf node in sklearn)
        rules_ = [[]]  
        # Tree with only one leaf
    else:
        rules_ = _generate_all_possible_rules(tree_structure)
        # Explore the tree structure to extract the longest rules (rules from root to a leaf)
    return rules_


def _generate_single_rule_mask(X, dimension, treshold, sign):
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
    single_mask_rule = X[:, dimension] > treshold
    if sign == "L":
        single_mask_rule = ~single_mask_rule

    return single_mask_rule


def _from_rules_to_constraint(rule):
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


def _generate_mask_rule(X, rules):
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
        dimension, treshold, sign = _from_rules_to_constraint(rule=rules[j])
        mask = _generate_single_rule_mask(
            X=X, dimension=dimension, treshold=treshold, sign=sign
        )
        list_mask.append(mask)
    final_mask = reduce(and_, list_mask)
    return final_mask

