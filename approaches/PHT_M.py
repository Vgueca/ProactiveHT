from approaches.base import ProactiveHTBase, register_approach
import numpy as np
import copy
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch



@register_approach
class ProactiveHT_M(ProactiveHTBase):
    ID = 2
    NAME = "PHT-M"

    def __init__(self, window_size=1000, approach = 2, stats_reset_mode = 0, alpha=0.01):
        super().__init__(window_size, approach, stats_reset_mode)
        self.n_closest_points = alpha*window_size

    
    def train(self, X, y ):

        acumulative_DB = []
        list_models = []
        predictions = []

        for idx, (index, row) in enumerate(X.iterrows()):
            x_dict = row.to_dict()
            label = y[index]

            y_hat = self.model.predict_one(x_dict)

            predictions.append((label, y_hat))
            self.model.learn_one(x_dict, label)


            # stats_reset_mode is set to 0 by default in our experimentation
            if self.stats_reset_mode == 7:
                leaf = self.find_leaf_node(x_dict)
                if leaf is not None:
                    self.fade_leaf_stats(leaf)       
            if self.stats_reset_mode == 5:
                self.fade_all_stats(self.model._root)       
            if self.stats_reset_mode == 4 or self.stats_reset_mode == 6:
                self.last_window_samples.append((x_dict, label))
                if len(self.last_window_samples) > self.window_size:
                    self.last_window_samples.pop(0)
            
            
            if (idx+1) % self.window_size == 0:
                decision_boundaries = self.extract_rules(self.model)
                for feature, threshold in decision_boundaries:
                    node, restrictions = self.find_node_and_restrictions(self.model._root, feature, threshold)
                    
                    closest_points = self.find_filtered_closest_points(
                        X.iloc[idx-self.window_size:idx, :], 
                        y.iloc[idx-self.window_size:idx], 
                        self.n_closest_points, 
                        (feature, threshold), 
                        restrictions
                    )
                    
                    new_threshold = self.update(closest_points, feature, threshold)

                decision_boundaries = self.extract_rules(self.model)
                acumulative_DB.append(copy.deepcopy(decision_boundaries))
                list_models.append(copy.deepcopy(self.model))

                if self.stats_reset_mode == 4 or self.stats_reset_mode == 6:
                    self.window_sample_remind()

        decision_boundaries = self.extract_rules(self.model)
        for feature, threshold in decision_boundaries:
            node, restrictions = self.find_node_and_restrictions(self.model._root, feature, threshold)
            closest_points = self.find_filtered_closest_points(X.iloc[idx-(idx % self.window_size):idx, :], y.iloc[idx-(idx % self.window_size):idx], self.n_closest_points, (feature, threshold), restrictions)
            new_threshold = self.update(closest_points, feature, threshold)
        
        decision_boundaries = self.extract_rules(self.model)
        acumulative_DB.append(copy.deepcopy(decision_boundaries))
        list_models.append(copy.deepcopy(self.model))

        return acumulative_DB, list_models, predictions
                   



    def update(self, closest_points, feature, threshold, learning_rate=0.1):

        points_region1 = [point for point, label, _, _ in closest_points if point[feature] > threshold]
        points_region2 = [point for point, label, _, _ in closest_points if point[feature] <= threshold]


        mean_region1 = np.mean([point[feature] for point in points_region1]) if points_region1 else threshold
        mean_region2 = np.mean([point[feature] for point in points_region2]) if points_region2 else threshold

        mean = (mean_region1 + mean_region2) / 2
        
        new_threshold = mean

        self.update_tree(feature, threshold, new_threshold)

        return new_threshold
    



    #=== ABLATION EXPERIMENT FUNCTIONS ===#
    def fade(self, d, fade_factor=0.9999):
        for k, v in d.items():
            if isinstance(v, dict):
                self.fade(v, fade_factor)
            else:
                d[k] = v * fade_factor

    def fade_all_stats(self, node, fade_factor=0.9999):
        if isinstance(node, HTLeaf):
            self.fade(node.stats, fade_factor)
        else:
            for child in node.children:
                self.fade_all_stats(child, fade_factor)

    def fade_leaf_stats(self, node, fade_factor=0.9999):
        self.fade(node.stats, fade_factor)

    def window_sample_remind(self): 
        if self.stats_reset_mode == 4:
            for x_dict, label in self.last_window_samples:
                self.model.learn_one(x_dict, label)

    def print_leaf_stats(self, node):
        if isinstance(node, HTLeaf):
            print(node.stats)
        else:
            for child in node.children:
                self.print_leaf_stats(child)

    def find_leaf_node(self, x):
        if self.model._root is None:
            return None

        if isinstance(self.model._root, DTBranch):
            return self.model._root.traverse(x, until_leaf=True)
        return self.model._root