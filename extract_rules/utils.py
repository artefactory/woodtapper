import numpy as np
from scipy.stats import binom


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
        if children:
            self.children = children
        else:
            self.children = []

def get_top_rules(all_possible_rules_list_str,p0):
    unique_str_rules, indices_rules, frequence_rules = np.unique(
    all_possible_rules_list_str, return_counts=True, return_index=True
    )  # get the unique rules and count
    frequence_rules = frequence_rules / frequence_rules.sum()  # convert to frequency
    unique_str_rules_and_freq = zip(unique_str_rules, frequence_rules) # combine rules and frequency
    all_rules_sorted = sorted(unique_str_rules_and_freq, key=lambda x: x[1], reverse=True) # sort by frequency
    all_possible_rules_and_freq_list = [(eval(unique_str_rule),freq) for unique_str_rule, freq in all_rules_sorted if freq > p0] # filter by p0
    if len(all_possible_rules_and_freq_list) == 0:
        if len(all_possible_rules_and_freq_list) == 0:
            raise ValueError(
                "No rule found with the given p0 value. Try to decrease it."
            )
    all_possible_rules_list, all_possible_freq_list =zip(*all_possible_rules_and_freq_list) # unzip
    return all_possible_rules_list, all_possible_freq_list


def compute_staibility_criterion(model):
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
            epsilon_numerator / epsilon_denominator
            if epsilon_denominator > 0
            else 0
        )
        list_epsilon.append(epsilon)
    print("***** \n Stability criterion value:", np.mean(list_epsilon), "\n*****")
