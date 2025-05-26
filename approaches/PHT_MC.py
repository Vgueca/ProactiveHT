from approaches.base import ProactiveHTBase, register_approach
import numpy as np
import pandas as pd
import copy
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import NumericBinaryBranch
from sklearn.feature_selection import mutual_info_classif

@register_approach
class ProactiveHT_MC(ProactiveHTBase):
    ID = 10
    NAME = "PHT-MC"

    def __init__(self, window_size=1000, approach = 10, stats_reset_mode=0, alpha=0.01):

        super().__init__(window_size,  approach)
        
        self.n_closest_points = alpha*window_size
        self.stats_reset_mode = stats_reset_mode
        
        self.max_history = 5
        self.threshold_move = 0.08
        self.leaf_histories = {}
    
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
            
            if (idx+1) % self.window_size == 0 :
                decision_boundaries = self.extract_rules(self.model)
                X_window = X.iloc[idx - self.window_size:idx, :]
                Y_window = y.iloc[idx - self.window_size:idx]


                for feature, threshold in decision_boundaries:
                    node, restrictions = self.find_node_and_restrictions(self.model._root, feature, threshold)
                    
                    new_threshold = self.update(
                        X_window,
                        Y_window,
                        feature,
                        threshold, 
                        restrictions
                    )

                leaves = self.update_leaf_histories(
                        X_window,
                        Y_window
                    )
                
                self.check_proactive_subtrees(X_window, Y_window, leaves)

                decision_boundaries = self.extract_rules(self.model)
                acumulative_DB.append(decision_boundaries)
                list_models.append(copy.deepcopy(self.model))

        decision_boundaries = self.extract_rules(self.model)
        acumulative_DB.append(decision_boundaries)
        list_models.append(copy.deepcopy(self.model))

        return acumulative_DB, list_models, predictions
                   
    def update(self, X, y, feature, threshold, restrictions=None):
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
    
        if filtered_X.empty or filtered_y.empty:
            return feature, threshold 
        
        
        closest_points = self.find_filtered_closest_points(filtered_X, filtered_y, self.n_closest_points, (feature, threshold))
        
        points_region1 = [point for point, label, _, _ in closest_points if point[feature] > threshold]
        points_region2 = [point for point, label, _, _ in closest_points if point[feature] <= threshold]


        mean_region1 = np.mean([point[feature] for point in points_region1]) if points_region1 else threshold
        mean_region2 = np.mean([point[feature] for point in points_region2]) if points_region2 else threshold

        mean = (mean_region1 + mean_region2) / 2
        
        new_threshold = mean

        self.update_tree(feature, threshold, new_threshold)

        return new_threshold
    

    def update_leaf_histories(self, X, y):
        
        leaves = self.get_leaves()  
        
        for leaf in leaves:
            leaf_id = id(leaf)

            if leaf_id not in self.leaf_histories:
                self.leaf_histories[leaf_id] = {
                    "history_centroids": [],
                    "last_centroid": None,
                    "smoothed_vector": None
                }
            
            feature, threshold, restrictions, side = self.find_leaf_boundary(leaf)

            if feature is None:
                continue

            closest_points = self.find_filtered_closest_points(
                X, y, self.n_closest_points,
                (feature, threshold),
                restrictions
            )

            if not closest_points:
                continue
            
            #points that belong to the region of the leaf
            region_points = [(pt, lbl, dist, reg) 
                         for (pt, lbl, dist, reg) in closest_points
                         if reg == side]
            
            if not region_points:
                continue

            points = np.array([p[0].values for p in region_points]) 
            centroid = np.mean(points, axis=0)

            hist_info = self.leaf_histories[leaf_id]
            hist_info["history_centroids"].append(centroid)
            if len(hist_info["history_centroids"]) > self.max_history:
                hist_info["history_centroids"].pop(0)
            hist_info["last_centroid"] = centroid

        return leaves
        
    def check_proactive_subtrees(self, X, y, leaves):

        for leaf in leaves:
            leaf_id = id(leaf)
            if leaf_id not in self.leaf_histories:
                continue

            leaf_history = self.leaf_histories[leaf_id]
            centroids = leaf_history["history_centroids"]
            if len(centroids) < 2:
                continue

            old_smooth = leaf_history.get("smoothed_vector", None)
            predicted_centroid, new_smooth = self.predict_next_centroid_displacement(
                centroids, old_smooth, alpha=0.75
            )
            
            self.leaf_histories[leaf_id]["smoothed_vector"] = new_smooth

            feature, threshold, restrictions, side = self.find_leaf_boundary(leaf)
            if feature is None:
                continue
            
            normal_vector = np.zeros(X.shape[1])
            feature_idx = X.columns.get_loc(feature)
            normal_vector[feature_idx] = 1

            last_centroid = centroids[-1]

            movement_vector = predicted_centroid - last_centroid

            if threshold - self.threshold_move <= predicted_centroid[feature_idx] <= threshold + self.threshold_move: #the estimated centroid will be close to the threshold
                parent = self.find_parent(self.model._root, leaf)
                if parent is None:
                    continue

                sibling, suboperator_leaf = (parent.children[0],"<=") if parent.children[1] is leaf else (parent.children[1],">")
                if not isinstance(sibling, HTLeaf):
                    continue

                _, _, restrictions, _ = self.find_leaf_boundary(sibling)

                self.create_proactive_split(X, y, parent, sibling, suboperator_leaf, feature, threshold, restrictions, side, movement_vector)


    def find_parent(self, current_node, target_node):

        if isinstance(current_node, NumericBinaryBranch):
            if current_node.children[0] is target_node or current_node.children[1] is target_node:
                return current_node
            parent = self.find_parent(current_node.children[0], target_node)
            if parent is not None:
                return parent
            parent = self.find_parent(current_node.children[1], target_node)
            if parent is not None:
                return parent
        elif hasattr(current_node, 'children'):
            
            for child in current_node.children:
                if child is target_node:
                    return current_node
                else:
                    parent = self.find_parent(child, target_node)
                    if parent is not None:
                        return parent
            return None
        return None


    def create_proactive_split(self, X, y, parent, leaf, suboperator_leaf, feature, threshold, restrictions, side,  movement_vector):
        closest_points = self.find_filtered_closest_points(
            X, y, self.n_closest_points,
            (feature, threshold),
            restrictions
        )

        if not closest_points:
            return
        bool_points_region1 = True

        if side == 0:
            closest_points_region0 = [(point,label) for point, label, _, _ in closest_points if point[feature] <= threshold]
            closest_points_region1 = [(point,label) for point, label, _, _ in closest_points if point[feature] > threshold]
            if not closest_points_region0:
                return 
            
            labels = [label for point, label, _, _ in closest_points if point[feature] <= threshold]
            invader_class = max(set(labels), key=labels.count)
        else:
            closest_points_region0 = [(point,label) for point, label, _, _ in closest_points if point[feature] > threshold]
            closest_points_region1 = [(point,label) for point, label, _, _ in closest_points if point[feature] <= threshold]
            if not closest_points_region0:
                return 
            
            labels = [label for point, label, _, _ in closest_points if point[feature] > threshold]
            invader_class = max(set(labels), key=labels.count)

        
        if not closest_points_region1:
            bool_points_region1 = False


        centroid_region0 = np.mean([point for point, _ in closest_points_region0], axis=0)
        if bool_points_region1:
            centroid_region1 = np.mean([point for point, _ in closest_points_region1], axis=0)

        indices = [pt.name for (pt, lbl, dist, reg) in closest_points]
        region_df = pd.DataFrame([pt for (pt, lbl, dist, reg) in closest_points],
                                columns=X.columns,
                                index=indices)
        region_labels = y.loc[indices]

        features = X.columns.tolist()
        if feature in features:
            features.remove(feature)

        if not features:
            return

        if region_df.empty or region_labels.empty:
            print("Empty region")
            return
        df_subset = region_df[features]

        if len(df_subset) < 3 or region_labels.nunique() < 2:
            return None 
        
        best_feature = self.select_best_feature(df_subset, region_labels, selection_method="variance_centroid")  
        if isinstance(best_feature, tuple):
            best_feature = best_feature[0]  
        best_feature = str(best_feature)

        best_feature_col = X.columns.get_loc(best_feature)

        
        
        if not bool_points_region1:
            new_threshold = min(point[best_feature] for point, label in closest_points_region0)
            new_suboperator = "<="
        else:
            if centroid_region0[best_feature_col] >= centroid_region1[best_feature_col]:
                new_threshold = (centroid_region0[best_feature_col] + centroid_region1[best_feature_col]) / 2
                new_suboperator = ">"
            else:
                new_threshold = (centroid_region0[best_feature_col] + centroid_region1[best_feature_col]) / 2
                new_suboperator = "<="

        leaves = tuple(
            self.model._new_leaf(initial_stats=None, parent=leaf) 
            for _ in range(2)
        )

        stats = {}
        branch = NumericBinaryBranch(stats, best_feature, new_threshold, leaf.depth, leaves[0], leaves[1])
        self.replace_node(leaf, branch)

        leaf_id = id(leaf)
        if leaf_id in self.leaf_histories:
            del self.leaf_histories[leaf_id]

        new_leaves = branch.children
        for new_leaf in new_leaves:
            new_leaf_id = id(new_leaf)
            self.leaf_histories[new_leaf_id] = {
                "history_centroids": [],
                "last_centroid": None,
                "smoothed_vector": None
            }

        invader_indices = [pt.name for (pt, lbl) in closest_points_region0]
        df_invaders = pd.DataFrame(
            [pt for (pt, lbl) in closest_points_region0],
            columns=X.columns,
            index=invader_indices
        )
        labels_invaders = y.loc[invader_indices]

        invader_centroid = np.mean([pt for (pt, lbl) in closest_points_region0], axis=0)
  
        new_restrictions = restrictions + [(best_feature, new_suboperator, new_threshold)]
        if not restrictions:
            subX = X.copy()
            subY = y.copy()

            if suboperator_leaf == ">":
                subX = subX[subX[feature] > threshold]
                if new_suboperator == ">":
                    subX = subX[subX[best_feature] > new_threshold]
                else:
                    subX = subX[subX[best_feature] <= new_threshold]
                subY = subY.loc[subX.index]
            else:
                subX = subX[subX[feature] <= threshold]
                if new_suboperator == ">":
                    subX = subX[subX[best_feature] > new_threshold]
                else:
                    subX = subX[subX[best_feature] <= new_threshold]
                subY = subY.loc[subX.index]

        else:
            subX, subY = self.data_filter(X, y, new_restrictions)

        df_union = pd.concat(
            [subX.reset_index(drop=True), df_invaders.reset_index(drop=True)], 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        lbl_union = pd.concat(
            [subY.reset_index(drop=True), labels_invaders.reset_index(drop=True)], 
            axis=0
        ).drop_duplicates().reset_index(drop=True)

        if len(subY) == 0:
            return
        classes_sub = subY.unique()

        if len(classes_sub) == 0:
            stats = {}
            for label in labels_invaders.unique():
                stats[label] = labels_invaders[labels_invaders == label].count()
            
            if new_suboperator == ">":
                new_leaves[1].stats = stats
            else:
                new_leaves[0].stats = stats
            return
        elif len(classes_sub) == 1:
            if invader_class in classes_sub:
                stats = {}
                for label in labels_invaders.unique():
                    stats[label] = labels_invaders[labels_invaders == label].count()
                for label in classes_sub:
                    stats[label] += subY[subY == label].count()

                if new_suboperator == ">":
                    new_leaves[1].stats = stats
                else:
                    new_leaves[0].stats = stats
                return
            else:
                if new_suboperator == ">":
                    self.create_child_split(df_union, lbl_union, branch.children[1], new_restrictions, best_feature, new_threshold, new_suboperator)
                else:
                    self.create_child_split(df_union, lbl_union, branch.children[0], new_restrictions, best_feature, new_threshold, new_suboperator)
        else:
            if new_suboperator == ">":
                self.create_child_split(df_union, lbl_union, branch.children[1], new_restrictions, best_feature, new_threshold, new_suboperator)
            else:
                self.create_child_split(df_union, lbl_union, branch.children[0], new_restrictions, best_feature, new_threshold, new_suboperator)


    def create_child_split(self, X_region, y_region, target_leaf, restrictions, feature, threshold, new_suboperator):
        if len(X_region) == 0:
            return
        
        X_region, y_region = X_region.align(y_region, join='inner', axis=0)

        closest_points = self.find_filtered_closest_points(
            X_region, y_region, self.n_closest_points,
            (feature, threshold),
            None
        )
        
        if not closest_points:
            return


        closest_points_region0 = [point for point, label, _, _ in closest_points if point[feature] <= threshold]
        closest_points_region1 = [point for point, label, _, _ in closest_points if point[feature] > threshold]

        if not closest_points_region0 or not closest_points_region1:
            return

        region0_df = pd.DataFrame(closest_points_region0, columns=X_region.columns)
        region1_df = pd.DataFrame(closest_points_region1, columns=X_region.columns)

        best_var = -1
        best_feat = None
        for feat in X_region.columns:
            var_col = region0_df[feat].var() + region1_df[feat].var()
            if var_col > best_var:
                best_var = var_col
                best_feat = feat

        if best_feat is None:
            return

        thr = (region0_df[best_feat].mean() + region1_df[best_feat].mean()) / 2

        leaves = tuple(
            self.model._new_leaf(initial_stats=None, parent=target_leaf)
            for _ in range(2)
        )

        if region0_df[best_feat].mean() > region1_df[best_feat].mean():
            stats = {}
            for point, label, _, _ in closest_points_region1:
                if label not in stats:
                    stats[label] = 0
                stats[label] += 1

            leaves[0].stats = stats

            stats = {}
            for point, label, _, _ in closest_points_region0:
                if label not in stats:
                    stats[label] = 0
                stats[label] += 1
            leaves[1].stats = stats
        else:
            stats = {}
            for point, label, _, _ in closest_points_region0:
                if label not in stats:
                    stats[label] = 0
                stats[label] += 1

            leaves[0].stats = stats

            stats = {}
            for point, label, _, _ in closest_points_region1:
                if label not in stats:
                    stats[label] = 0
                stats[label] += 1
            leaves[1].stats = stats

        child_branch = NumericBinaryBranch(stats, best_feat, thr, target_leaf.depth, leaves[0], leaves[1])

        self.replace_node(target_leaf, child_branch)

        for new_leaf in leaves:
            new_leaf_id = id(new_leaf)
            self.leaf_histories[new_leaf_id] = {
            "history_centroids": [],
            "last_centroid": None,
            "smoothed_vector": None
            }


    def get_leaves(self):
        return self.collect_leaves(self.model._root)

    def collect_leaves(self, node):
        if isinstance(node, HTLeaf):
            return [node]

        leaves = []
        
        if hasattr(node, 'children'):
            for child in node.children:
                leaves.extend(self.collect_leaves(child))

        return leaves

    def find_leaf_boundary(self, leaf):
        found, restrictions = self.find_leaf_path_restrictions(self.model._root, leaf)

        if not found or not restrictions:
            return None, None, [], None
        
        feature, operator, threshold = restrictions[-1]
        restrictions = restrictions[:-1]

        region = 0 if operator == "<=" else 1

        return feature, threshold, restrictions, region


    def find_leaf_path_restrictions(self, current_node, target_leaf, path_restrictions=None):
        if path_restrictions is None:
            path_restrictions = []

        if current_node is target_leaf:
            return True, path_restrictions

        if hasattr(current_node, 'children'):
            for i, child in enumerate(current_node.children):
                operator = '<=' if i == 0 else '>'
                new_path = path_restrictions + [(current_node.feature, operator, current_node.threshold)]

                found, found_restrictions = self.find_leaf_path_restrictions(child, target_leaf, new_path)
                if found:
                    return True, found_restrictions

        return False, []

        
        return False, []

    def replace_node(self, old_node, new_node):

        def traverse_and_replace(current_node):
            if hasattr(current_node, 'children'):
                for i, child in enumerate(current_node.children):
                    if child is old_node:
                        current_node.children[i] = new_node
                        return True
                    else:
                        replaced = traverse_and_replace(child)
                        if replaced:
                            return True
            return False

        replaced = traverse_and_replace(self.model._root)
        return replaced
    

    def predict_next_centroid_displacement(self, centroids, old_smoothed_vector=None, alpha=0.75):
        n = len(centroids)
        if n == 0:
            return None, np.zeros(0)
        if n == 1:
            return centroids[-1], np.zeros_like(centroids[-1])

        last_centroid = centroids[-1]  
        prev_centroid = centroids[-2]   

        if old_smoothed_vector is None:
            old_smoothed_vector = np.zeros_like(last_centroid)

        displacement = last_centroid - prev_centroid
        new_smoothed = alpha * displacement + (1 - alpha) * old_smoothed_vector
        next_centroid = last_centroid + new_smoothed

        return next_centroid, new_smoothed
    
    def select_best_feature(self, X, y, selection_method="combined"):
        features = X.columns.tolist()
        
        if selection_method == "combined":
            mutual_info = mutual_info_classif(X[features], y, discrete_features='auto', n_neighbors=1)
            
            centroids = {label: X[y == label].mean().values for label in y.unique()}
            centroid_distances = np.zeros(len(features))

            for i, feature in enumerate(features):
                if len(centroids) > 1:
                    labels = list(centroids.keys())
                    dist = np.linalg.norm(centroids[labels[0]] - centroids[labels[1]]) 
                    centroid_distances[i] = dist

            mi_norm = mutual_info / np.max(mutual_info) if np.max(mutual_info) > 0 else mutual_info
            centroid_distances_norm = centroid_distances / np.max(centroid_distances) if np.max(centroid_distances) > 0 else centroid_distances

            scores = 0.6 * mi_norm + 0.4 * centroid_distances_norm
            best_feature_idx = np.argmax(scores)
        
        elif selection_method == "mutual_info":
            mutual_info = mutual_info_classif(X[features], y, discrete_features='auto', n_neighbors=1)
            best_feature_idx = np.argmax(mutual_info)
        
        elif selection_method == "variance":
            variances = X.var()
            best_feature_idx = np.argmax(variances)
        
        elif selection_method == "variance_centroid":
            variances = X.var().values
            
            centroids = {label: X[y == label].mean().values for label in y.unique()}
            centroid_distances = np.zeros(len(features))

            for i, feature in enumerate(features):
                if len(centroids) > 1:
                    labels = list(centroids.keys())
                    dist = np.linalg.norm(centroids[labels[0]] - centroids[labels[1]]) 
                    centroid_distances[i] = dist
            
            var_norm = variances / np.max(variances) if np.max(variances) > 0 else variances
            centroid_distances_norm = centroid_distances / np.max(centroid_distances) if np.max(centroid_distances) > 0 else centroid_distances
            
            scores = 0.6 * var_norm + 0.4 * centroid_distances_norm
            best_feature_idx = np.argmax(scores)
        
        else:
            raise ValueError("Invalid selection method. Choose 'combined', 'mutual_info', 'variance', or 'variance_centroid'.")
        
        best_feature = features[best_feature_idx]
        return best_feature, best_feature_idx
