from abc import ABC
import copy
import numpy as np


from approaches.base import ProactiveHTBase, register_approach
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import NumericBinaryBranch, NominalBinaryBranch

@register_approach  
class EFDT_M(ProactiveHTBase):
    ID = 23
    NAME = "EFDT"

    def __init__(self, window_size=1000, approach=23, stats_reset_mode=0, alpha=0.01):
        super().__init__(window_size, approach, stats_reset_mode)
        self.window_size = 1000

        self.n_closest_points = int(alpha * window_size)
    
    def train(self, X, y):

        acumulative_DB = []
        list_models = []
        predictions = []

        for idx, (index, row) in enumerate(X.iterrows()):
            x_dict = row.to_dict()
            label = y[index]
    
            y_hat = self.model.predict_one(x_dict)
            predictions.append((label, y_hat))
            self.model.learn_one(x_dict, label)
    
            if (idx+1) % self.window_size == 0 :
                self.proactive_update(
                    X.iloc[idx-self.window_size : idx],
                    y.iloc[idx-self.window_size : idx]
                )

                decision_boundaries = self.extract_rules(self.model)
                acumulative_DB.append(decision_boundaries)
                list_models.append(copy.deepcopy(self.model))
        

        decision_boundaries = self.extract_rules(self.model)
        acumulative_DB.append(decision_boundaries)
        list_models.append(copy.deepcopy(self.model))

        return acumulative_DB, list_models, predictions
    
    def proactive_update(self, X_win, y_win):

        for feature, threshold in self.extract_rules(self.model):
            
            node, restrictions = self.find_node_and_restrictions(
                self.model._root,
                feature,
                threshold
            )
            if node is None:
                continue

            closest = self.find_filtered_closest_points(
                X_win, y_win,
                self.n_closest_points,
                (feature, threshold),
                restrictions
            )

            new_thr = self.update(closest, feature, threshold)

            
    def update(self, closest_points, feature, threshold):

        region1 = [p for p, _, _, _ in closest_points if p[feature] >  threshold]
        region2 = [p for p, _, _, _ in closest_points if p[feature] <= threshold]

        m1 = np.mean([p[feature] for p in region1]) if region1 else threshold
        m2 = np.mean([p[feature] for p in region2]) if region2 else threshold

        new_threshold = (m1 + m2) / 2.0

        self.update_tree(feature, threshold, new_threshold)
        return new_threshold