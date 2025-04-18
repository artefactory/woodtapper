import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree, export_text

class Node:
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
        if self.tree_.children_left[node_id] != _tree.TREE_LEAF: # possible to add a max_depth constraint exploration value
            id_left_child = self.tree_.children_left[node_id]
            id_right_child = self.tree_.children_right[node_id]
            children = [
                self.explore_tree_(id_left_child,'L'),
                self.explore_tree_(id_right_child,'R')
            ]
        else:
            return Node(feature=self.tree_.feature[node_id],treshold=self.tree_.threshold[node_id],side=side,node_id=node_id)
        
        return Node(self.tree_.feature[node_id],self.tree_.threshold[node_id],side,node_id,*children)
    def construct_longest_paths_(self,root):
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
                tree_structure.append(common_path_rules) ## DROITE : Ajouté à la fin
                tree_structure[indice_in_tree_struct].append(rule_left) ## GAUCHE  : On le rajoute selon indice_in_tree_struct
        
                stack.append((curr_rule.children[0],indice_in_tree_struct))
                stack.append(( curr_rule.children[1],len(tree_structure)-1 ))
            else:
                #print('c')
                continue
        return tree_structure

    def split_sub_rules_(self,path,is_removing_singleton=False):
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
        all_paths_list = []
        for i in range(len(tree_structure)):
            print('*****'*10)
            curr_path = tree_structure[i]
            max_size_curr_path = len(curr_path)
            
            for k in range(max_size_curr_path):
                list_sub_path = self.split_sub_rules_(curr_path[k:],is_removing_singleton=False)
                print('ici : ', list_sub_path)
                all_paths_list.extend(list_sub_path)
        
            ## CAS plus complexes :
            if max_size_curr_path ==1:
                #all_paths_list.append(curr_path)
                print("SKIIIIP")
                continue
            else:
                curr_path_size_pair = ( (max_size_curr_path%2)==0)
                print('TAIIIILLE : ',max_size_curr_path)
        
                if curr_path_size_pair:
                    print('aa')
                    #list_sub_path = self.split_sub_rules_(curr_path)
                    #all_paths_list.extend(list_sub_path)          
                    for k in range(1, (max_size_curr_path//2) ):
                        print('curr path : ',curr_path[k:max_size_curr_path-k])
                        list_sub_path = self.split_sub_rules_(curr_path[k:max_size_curr_path-k],is_removing_singleton=True)
                        print('list_sub_path : ',list_sub_path)
                        all_paths_list.extend(list_sub_path)
                else:
                    print('b')
                    #list_sub_path = self.split_sub_rules_(curr_path)
                    #all_paths_list.extend(list_sub_path)
                    for k in range(1,(max_size_curr_path//2) ):
                        print('curr path : ',curr_path[k:max_size_curr_path-k])
                        list_sub_path = self.split_sub_rules_(curr_path[k:max_size_curr_path-k],is_removing_singleton=True)
                        all_paths_list.extend(list_sub_path)
                        print('list_sub_path : ',list_sub_path)
                        if k == (max_size_curr_path//2): #case odd last 
                            list_sub_path = self.split_sub_rules_(curr_path[k:max_size_curr_path-(k-1)],is_removing_singleton=True)
                            print('list_sub_path : ',list_sub_path)
                            all_paths_list.extend(list_sub_path)
            print('end iteration all_paths_list : ',all_paths_list)
        return all_paths_list
    
    def fit(self, X, y, sample_weight=None):
        super().fit(X=X, y=y, sample_weight=sample_weight)

        root = self.explore_tree_(0,'Root')
        tree_structure = self.construct_longest_paths_(root)
        all_possible_rules_list = self.generate_all_possible_rules_(tree_structure)
        all_possible_rules_list_str = [str(elem) for elem in all_possible_rules_list]
        unique_str_rules,indices_rules,count_rules = np.unique(all_possible_rules_list_str,return_counts=True,return_index=True)
        return unique_str_rules,indices_rules,count_rules

class SirusDTreeClassifier(SirusMixin, DecisionTreeClassifier):
    """
    SIRUS class applied with a DecisionTreeClassifier

    """

class SirusRFClassifier(SirusMixin, RandomForestClassifier):
    """
    SIRUS class applied with a RandomForestClassifier

    """
