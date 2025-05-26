from abc import ABC
import copy
import numpy as np

from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import NumericBinaryBranch, NominalBinaryBranch
from approaches.base import ProactiveHTBase, register_approach

@register_approach
class ProactiveHAT_M(ProactiveHTBase):
    ID   = 22
    NAME = "PHAT-M"

    def __init__(self, window_size=1000, approach=22, stats_reset_mode=0, alpha=0.01):

        super().__init__(window_size, approach_id=approach, stats_reset_mode=stats_reset_mode)
        
        self.n_closest_points = int(alpha * window_size)

    def train(self, X, y):

        accum_db     = []
        models       = []
        predictions  = []

        for idx, (_, row) in enumerate(X.iterrows(), start=1):
            x_dict = row.to_dict()
            true_y = y.iloc[idx-1]

            y_pred = self.model.predict_one(x_dict)
            predictions.append((true_y, y_pred))
            self.model.learn_one(x_dict, true_y)

            if idx % self.window_size == 0:
                self._proactive_update(
                    X.iloc[idx-self.window_size : idx],
                    y.iloc[idx-self.window_size : idx]
                )
                accum_db.append(copy.deepcopy(self.extract_rules(self.model)))
                models.append(copy.deepcopy(self.model))

        remainder = len(X) % self.window_size
        if remainder:
            self._proactive_update(
                X.iloc[-remainder:],
                y.iloc[-remainder:]
            )
        accum_db.append(copy.deepcopy(self.extract_rules(self.model)))
        models.append(copy.deepcopy(self.model))

        return accum_db, models, predictions

    def _proactive_update(self, X_win, y_win):

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

            
            if hasattr(node, 'children'):
                for child in node.children:
                    if isinstance(child, HTLeaf) and hasattr(child, 'drift_detector'):
                        child.drift_detector = child.drift_detector.clone()

    def update(self, closest_points, feature, threshold):

        region1 = [p for p, _, _, _ in closest_points if p[feature] >  threshold]
        region2 = [p for p, _, _, _ in closest_points if p[feature] <= threshold]

        m1 = np.mean([p[feature] for p in region1]) if region1 else threshold
        m2 = np.mean([p[feature] for p in region2]) if region2 else threshold

        new_threshold = (m1 + m2) / 2.0

        self.update_tree(feature, threshold, new_threshold)
        return new_threshold
