from approaches.base import ProactiveHTBase, register_approach

import copy
import numpy as np
import pandas as pd
from river.tree import HoeffdingTreeClassifier


@register_approach
class ProactiveHT_S(ProactiveHTBase):
    ID = 8
    NAME = "PHT-S"
    def __init__(self, window_size=1000, approach=8, stats_reset_mode=0, alpha=0.01):
        
        super().__init__(window_size, approach)
        
        self.n_closest_points = int(alpha * window_size)
        self.stats_reset_mode = stats_reset_mode

        self.shadow_forest = {}
        self.improvement_history = []
        
        self.node_id_counter = 0
        self.node_region_alerts = {}

        self.region_threshold = 0.5

        if self.model._root is not None:
            self.assign_node_ids(self.model._root)

    def assign_node_ids(self, node): #just applied to split nodes
        if not hasattr(node, 'node_id'):
            node.node_id = self.node_id_counter
            self.node_id_counter += 1

            self.node_region_alerts[node.node_id] = {
                0: {
                    "in_alert": False,
                    "shadow_model": None,
                    "centroid_history": [],
                    "smoothed_vector": None,
                    "points_movements": {}
                },
                1: {
                    "in_alert": False,
                    "shadow_model": None,
                    "centroid_history": [],
                    "smoothed_vector": None,
                    "points_movements": {}
                }
            }

        if hasattr(node, 'children') and isinstance(node.children, list):
            for child in node.children:
                self.assign_node_ids(child)

    def get_node_by_id(self, current_node, target_id):
        if current_node.node_id == target_id:
            return current_node
        if hasattr(current_node, 'children') and isinstance(current_node.children, list):
            for child in current_node.children:
                found = self.get_node_by_id(child, target_id)
                if found:
                    return found
        return None
    
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
         
            if (idx + 1) % self.window_size == 0:
                start_idx = idx + 1 - self.window_size
                end_idx = idx

                decision_boundaries = self.extract_rules(self.model)
                
                for feature, threshold in decision_boundaries:
                    node, restrictions = self.find_node_and_restrictions(self.model._root, feature, threshold)
                    if node is None:
                        continue
                    self.assign_node_ids(self.model._root)
        
                    node_id = getattr(node, 'node_id', None)
                    if node_id is None:
                        continue

                    filtered_X, filtered_y = self.data_filter(
                        X.iloc[start_idx:end_idx+1, :],
                        y.iloc[start_idx:end_idx+1],
                        restrictions
                    )
                    if filtered_X.empty:
                        continue

                    self.analyze_movement(node, feature, threshold, filtered_X, filtered_y, restrictions)

                    self.check_proactive_replacement(node, feature, threshold, filtered_X, filtered_y, restrictions)

                decision_boundaries = self.extract_rules(self.model)
                acumulative_DB.append(decision_boundaries)
                list_models.append(copy.deepcopy(self.model))

        decision_boundaries = self.extract_rules(self.model)
        acumulative_DB.append(decision_boundaries)
        list_models.append(copy.deepcopy(self.model))

        return acumulative_DB, list_models, predictions

    def update_threshold_and_rotate(self, node, feature, threshold, filtered_X, filtered_y):
        points_region1 = filtered_X[filtered_X[feature] <= threshold]
        points_region2 = filtered_X[filtered_X[feature] > threshold]

        if points_region1.empty or points_region2.empty:
            return threshold, feature

        centroid_region1 = np.mean(points_region1.values, axis=0)
        centroid_region2 = np.mean(points_region2.values, axis=0)
        vector_centroids = centroid_region1 - centroid_region2
        norm_centroids = np.linalg.norm(vector_centroids)

        if norm_centroids == 0:
            return threshold, feature

        normal_vector = np.zeros(filtered_X.shape[1])
        idx_feature = filtered_X.columns.get_loc(feature)
        normal_vector[idx_feature] = 1

        if np.dot(vector_centroids, normal_vector) < 0:
            normal_vector = -normal_vector

        cos_theta = np.dot(vector_centroids, normal_vector) / (norm_centroids * np.linalg.norm(normal_vector))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angle_hyperplane = 90 - angle

        new_feature = feature
        new_threshold = threshold

        if angle_hyperplane < 45:
            variance_per_dim = np.var(filtered_X.values, axis=0)
            new_feature_idx = np.argmax(variance_per_dim)
            new_feature = filtered_X.columns[new_feature_idx]
            new_threshold = (centroid_region1[new_feature_idx] + centroid_region2[new_feature_idx]) / 2

        closest_points = self.find_filtered_closest_points(filtered_X, filtered_y, self.n_closest_points, (new_feature, new_threshold))
        region1 = [p for p,l,_,_ in closest_points if p[new_feature] > new_threshold]
        region2 = [p for p,l,_,_ in closest_points if p[new_feature] <= new_threshold]
        mean1 = np.mean([p[new_feature] for p in region1]) if region1 else new_threshold
        mean2 = np.mean([p[new_feature] for p in region2]) if region2 else new_threshold
        final_thresh = (mean1 + mean2)/2

        if final_thresh != threshold or new_feature != feature:
            node.feature = new_feature
            node.threshold = final_thresh

        return final_thresh, new_feature

    
    def analyze_movement(self, node, feature, threshold, filtered_X, filtered_y, restrictions):
        node_id = node.node_id

        closest_points = self.find_filtered_closest_points(
            filtered_X, filtered_y, self.n_closest_points,
            (feature, threshold),
            restrictions
        )

        closest_points_region0 = [(point,label) for point, label, _, _ in closest_points if point[feature] <= threshold]
        closest_points_region1 = [(point,label) for point, label, _, _ in closest_points if point[feature] > threshold]
        

        if closest_points_region0:
            centroid_region0 = np.mean([point for point, _ in closest_points_region0], axis=0)
        else:
            centroid_region0 = None

        if closest_points_region1:
            centroid_region1 = np.mean([point for point, _ in closest_points_region1], axis=0)
        else:
            centroid_region1 = None

        centroids = { 0: centroid_region0, 1: centroid_region1 }
        for r in [0,1]:
            if centroids[r] is not None:
                self.update_centroids_stats(node_id, r, centroids[r])

        for r in [0,1]:      
            smoothed_vector = self.node_region_alerts[node_id][r]["smoothed_vector"]
            idx = int(feature.split("_")[1])
            if smoothed_vector is None:
                continue
            if (smoothed_vector[idx] > 0 and r == 1) or (smoothed_vector[idx] <= 0 and r == 0):
                continue

            if centroids[r] is None:
                continue
            in_alert = False
            if not self.node_region_alerts[node_id][r]["in_alert"]:
                in_alert = self.check_if_in_alert(centroids[r], feature, threshold)

            if in_alert and not self.node_region_alerts[node_id][r]["in_alert"]:
                self.node_region_alerts[node_id][r]["in_alert"] = True
                self.node_region_alerts[node_id][r]["shadow_model"] = HoeffdingTreeClassifier(leaf_prediction='mc')

            if self.node_region_alerts[node_id][r]["in_alert"]:
                
                shadow_model = self.node_region_alerts[node_id][r]["shadow_model"]

                if r == 0:
                    invader_points = closest_points_region0
                    other_region_mask = (filtered_X[feature] > threshold) 
                else:
                    invader_points = closest_points_region1
                    other_region_mask = (filtered_X[feature] <= threshold)

                if invader_points:
                    invader_indices = [pt.name for (pt, lbl) in invader_points]
                    X_invad = pd.DataFrame(
                        [pt for (pt, lbl) in invader_points],
                        columns=filtered_X.columns,
                        index=invader_indices
                    )
                    y_invad = filtered_y.loc[invader_indices]
                else:
                    X_invad = pd.DataFrame(columns=filtered_X.columns)
                    y_invad = pd.Series(dtype=filtered_y.dtype)

                X_other = filtered_X[other_region_mask]
                y_other = filtered_y.loc[X_other.index]

                X_invad_pushed = self.shift_invaders(X_invad, feature, threshold, centroids[r], region=r)
                
                X_shadow = pd.concat([X_invad_pushed, X_other], axis=0)
                y_shadow = pd.concat([y_invad, y_other], axis=0).loc[X_shadow.index]

                X_shadow = X_shadow.reset_index(drop=True)
                y_shadow = y_shadow.reset_index(drop=True)

                for i, row_data in X_shadow.iterrows():
                    label = y_shadow[i]  
                    shadow_model.learn_one(row_data.to_dict(), label)

    def shift_invaders(self, X_invad, feature, threshold, region_centroid, region):

        epsilon = 0.1

        if X_invad.empty or region_centroid is None:
            return X_invad

        X_pushed = X_invad.copy()

        feat_idx = int(feature.split("_")[1])

        offset = threshold - region_centroid[feat_idx]
        if region == 1:
            offset = offset - epsilon
        else:
            offset = offset + epsilon

        for i_row, row_data in X_pushed.iterrows():
            old_val = row_data[feature]
            new_val = old_val + offset
            X_pushed.at[i_row, feature] = new_val

        return X_pushed
                
    def update_centroids_stats(self, node_id, region, last_centroid):

        centroid_history = self.node_region_alerts[node_id][region]["centroid_history"]

        if len(centroid_history) == 0:
            previous_centroid = None
        else:
            previous_centroid = centroid_history[-1]

        centroid_history.append(last_centroid)

        smoothed_vector = self.node_region_alerts[node_id][region]["smoothed_vector"]
        if smoothed_vector is None:
            smoothed_vector = np.zeros(len(last_centroid))

        if previous_centroid is not None:
            displacement = last_centroid - previous_centroid
            new_smoothed_vector = 0.8 * displacement + 0.2 * smoothed_vector 
        else:
            new_smoothed_vector = smoothed_vector

        self.node_region_alerts[node_id][region]["smoothed_vector"] = new_smoothed_vector

    def check_proactive_replacement(self, node, feature, threshold, filtered_X, filtered_y, restrictions):
        node_id = node.node_id

        for r in [0,1]:
            in_alert = self.node_region_alerts[node_id][r]["in_alert"]
            if not in_alert:
                continue

            crossing = self.check_if_crossing(node_id, r, feature, threshold)

            if crossing:
                shadow_model = self.node_region_alerts[node_id][r]["shadow_model"]
                if shadow_model is not None:
                    sibling_node = node.children[1-r]
                    self.replace_node_with_subtree(sibling_node, shadow_model)
                    self.assign_node_ids(self.model._root)
                else:
                    print("No hay shadow_model para region r, no se reemplaza.")

    def check_if_in_alert(self, centroid, feature, threshold):
            
            idx = int(feature.split("_")[1])
            dist_to_thr = np.abs(centroid[idx] - threshold)
        
            return (dist_to_thr < self.region_threshold)


    def check_if_crossing(self, node_id, r, feature, threshold, alpha=0.75):
        idx = int(feature.split("_")[1])
        smoothed_vector = self.node_region_alerts[node_id][r]["smoothed_vector"]

        if smoothed_vector is None:
            return False
        
        last_centroid = self.node_region_alerts[node_id][r]["centroid_history"][-1]

        if last_centroid is None:
            return False
        
        predicted_centroid = last_centroid + smoothed_vector

        if r == 0:
            return predicted_centroid[idx] > threshold
        else:
            return predicted_centroid[idx] <= threshold
    
    def replace_node_with_subtree(self, node, subtree_model):
        if self.model._root is node:
            self.model._root = subtree_model._root
            return True
        success = self.replace_node_recursive(self.model._root, node, subtree_model._root)
        return success

    def replace_node_recursive(self, current_node, old_node, new_node):
        if not hasattr(current_node, 'children') or not isinstance(current_node.children, list):
            return False

        for i, child in enumerate(current_node.children):
            if child is old_node:
                current_node.children[i] = new_node
                return True
            replaced = self.replace_node_recursive(child, old_node, new_node)
            if replaced:
                return True
        return False

    def update(self, X, y, idx, decision_boundaries):
        pass

    def get_leaves(self, node):
        leaves = []
        if hasattr(node, 'children') and isinstance(node.children, list) and len(node.children) > 0:
            for child in node.children:
                leaves.extend(self.get_leaves(child))
        else:
            leaves.append(node)
        return leaves

    def find_leaf_for_region(self, root, X_region, y_region):
        leaves = self.get_leaves(root)
        for leaf in leaves:
            restrictions = self.get_region_restrictions(root, leaf)
            filtered_X, _ = self.data_filter(X_region, y_region, restrictions)
            if len(filtered_X) == len(X_region):
                return leaf
        return None
    
    def get_region_restrictions(self, root, leaf):
        path = []
        self.find_path_to_leaf(root, leaf, path)
        restrictions = []
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i+1]
            if hasattr(node, 'feature') and hasattr(node, 'threshold'):
                if len(getattr(node, 'children', [])) >= 2:
                    if node.children[0] is next_node:
                        restrictions.append((node.feature, '<=', node.threshold))
                    else:
                        restrictions.append((node.feature, '>', node.threshold))
        return restrictions

    def find_path_to_leaf(self, current_node, target_leaf, path):
        if current_node is target_leaf:
            path.append(current_node)
            return True
        if hasattr(current_node, 'children') and isinstance(current_node.children, list):
            for child in current_node.children:
                if self.find_path_to_leaf(child, target_leaf, path):
                    path.insert(0, current_node)
                    return True
        return False
