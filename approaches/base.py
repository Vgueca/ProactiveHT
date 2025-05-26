from abc import ABC, abstractmethod
from river.tree import *
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import NumericBinaryBranch
from river.tree.nodes.branch import NominalBinaryBranch


_APPROACH_REGISTRY = {}

def register_approach(cls):
    """Decorator: registers the class in the global dictionary according to its ID attribute."""
    aid = getattr(cls, "ID", None)
    if aid is None:
        raise ValueError(f"Class {cls.__name__} must define a static attribute `ID`")
    _APPROACH_REGISTRY[aid] = cls
    return cls

def get_approach(approach_id, window_size):
    """Returns an instance of the class whose ID == approach_id."""
    cls = _APPROACH_REGISTRY.get(approach_id)
    if cls is None:
        raise KeyError(f"No approach with ID {approach_id} registered")
    return cls(window_size=window_size, approach=approach_id)


class ProactiveHTBase(ABC):
    def __init__(self, window_size=1000, approach_id=None, stats_reset_mode = 0):

        if approach_id == 20 or approach_id == 22:
            self.model = HoeffdingAdaptiveTreeClassifier(leaf_prediction='mc')
        elif approach_id == 21 or approach_id == 23:
            self.model = ExtremelyFastDecisionTreeClassifier(leaf_prediction='mc')
        else:
            self.model = HoeffdingTreeClassifier(leaf_prediction='mc')
        
        
        self.window_size = window_size
        self.approach_id = approach_id
        self.stats_reset_mode = stats_reset_mode
        
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def update(self, X, y, idx, decision_boundaries):
        pass

    def traverse_tree(self, node):
        boundaries=[]
        if hasattr(node, 'feature'):  
            feature = node.feature
            threshold = node.threshold

            boundaries.append((feature, threshold))

            if hasattr(node, 'children') and isinstance(node.children, list):
                for child in node.children:
                    child_result = self.traverse_tree(child)
                    if child_result:
                        boundaries.extend(child_result)

        return boundaries

    def extract_rules(self, model):
        decision_boundaries = self.traverse_tree(model._root)
        return decision_boundaries

    def update_tree(self, target_feature, target_threshold, new_threshold, new_feature=None, rotation=None):
        nodes_to_visit = [self.model._root]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)

            if hasattr(node, 'feature'):
                if node.feature == target_feature and node.threshold == target_threshold:
                    node.threshold = new_threshold
          
                    if new_feature:
                        node.feature = new_feature
                        
                    if rotation:
                        aux = node.children[0]
                        node.children[0] = node.children[1] 
                        node.children[1] = aux

                    for child in node.children:
                        if isinstance(child, HTLeaf):
                            if bool(child.stats):

                                if self.stats_reset_mode == 0:
                                    self.reset_stats_NO(child)
                                elif self.stats_reset_mode == 1:
                                    self.reset_stats(child)
                                elif self.stats_reset_mode == 2:
                                    self.reset_stats_predominant(child)
                                elif self.stats_reset_mode == 3:
                                    self.reset_stats_predominant_difference(child)
                                elif self.stats_reset_mode == 4:
                                    self.reset_stats(child)
                                else:
                                    self.reset_stats_NO(child)
                                    
                        else:
                            pass
                    break

            if hasattr(node, 'children'):
                nodes_to_visit.extend(node.children)
    



    def reset_stats(self, leaf):
        for key in list(leaf.stats.keys()):
            leaf.stats[key] = 0
        return

    def reset_stats_predominant_difference(self, leaf):
    
        if len(leaf.stats) < 2:
            predominant_class = max(leaf.stats, key=leaf.stats.get)
            for key in list(leaf.stats.keys()):
                if key != predominant_class:
                    leaf.stats[key] = 0
            return

        sorted_classes = sorted(leaf.stats.items(), key=lambda x: x[1], reverse=True)
        predominant_class, predominant_count = sorted_classes[0]
        second_predominant_class, second_class_count = sorted_classes[1]

        leaf.stats[predominant_class] = max(0, predominant_count - second_class_count)

        for key in list(leaf.stats.keys()):
            if key != predominant_class:
                leaf.stats[key] = 0

    def reset_stats_predominant(self, leaf):
        if not leaf.stats:
            return

        predominant_class = max(leaf.stats, key=leaf.stats.get)
        for key in list(leaf.stats.keys()):
            if key != predominant_class:
                leaf.stats[key] = 0

    def reset_stats_NO(self, leaf):
        return
    

        
    def find_node_and_restrictions(self, current_node, target_feature, target_threshold, path_restrictions=None):
        if path_restrictions is None:
            path_restrictions = []

        if (hasattr(current_node, 'feature') and 
            current_node.feature == target_feature and 
            current_node.threshold == target_threshold):
            return current_node, path_restrictions

        if isinstance(current_node, NumericBinaryBranch) or isinstance(current_node, NominalBinaryBranch):
            left_child, right_child = current_node.children

            if left_child is not None:
                found_left, restrictions_left = self.find_node_and_restrictions(
                    left_child, 
                    target_feature, 
                    target_threshold, 
                    path_restrictions + [(current_node.feature, '<=', current_node.threshold)]
                )
                if found_left is not None:
                    return found_left, restrictions_left

            if right_child is not None:
                found_right, restrictions_right = self.find_node_and_restrictions(
                    right_child, 
                    target_feature, 
                    target_threshold, 
                    path_restrictions + [(current_node.feature, '>', current_node.threshold)]
                )
                if found_right is not None:
                    return found_right, restrictions_right

        return None, []


    def closest_points(self, X, y, num_points, DB, n_subprocesses=4):
        return self.find_filtered_closest_points(X, y, num_points, DB)


    def find_filtered_closest_points(self, X, y, num_points, DB, restrictions=None): #with filtering
        feature, threshold = DB

        filtered_X, filtered_y = self.data_filter(X, y, restrictions)
        
        distances = []
        for index, row in filtered_X.iterrows():
            distance = abs(row[feature] - threshold)  
            region = 0 if row[feature] <= threshold else 1  
            distances.append((index, distance, region))

        distances.sort(key=lambda x: (x[1], x[2]))

        closest_points = []
        region_counts = {0: 0, 1: 0}
        for index, distance, region in distances:
            if region_counts[region] < num_points:
                point = filtered_X.loc[index]
                label = filtered_y.loc[index]
                closest_points.append((point, label, distance, region))
                region_counts[region] += 1

        return closest_points
                     

    # def find_closest_points(self, X, y, num_points, DB): #no filtering necessary
    
    #     feature, threshold = DB

    #     distances = []
    #     for index, row in X.iterrows():
    #         distance = abs(row[feature] - threshold)  
    #         region = 0 if row[feature] <= threshold else 1  
    #         distances.append((index, distance, region))

    #     distances.sort(key=lambda x: (x[1], x[2]))

    #     closest_points = []
    #     class_counts = {}
    #     for index, distance, region in distances:
    #         label = y.loc[index]
    #         if label not in class_counts:
    #             class_counts[label] = 0
    #         if class_counts[label] < num_points:
    #             point = X.loc[index]
    #             closest_points.append((point, label, distance))
    #             class_counts[label] += 1

    #     return closest_points

    def data_filter(self, X, y, restrictions=None):
        filtered_X = X.copy()
        filtered_y = y.copy()
        
        if restrictions:
            for feature_restriction, operator, threshold_restriction in restrictions:
                if operator == '<=':
                    filtered_X = filtered_X[filtered_X[feature_restriction] <= threshold_restriction]
                    filtered_y = filtered_y.loc[filtered_X.index]
                elif operator == '>':
                    filtered_X = filtered_X[filtered_X[feature_restriction] > threshold_restriction]
                    filtered_y = filtered_y.loc[filtered_X.index]

        return filtered_X, filtered_y