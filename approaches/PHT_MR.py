from approaches.base import ProactiveHTBase, register_approach
import numpy as np
import copy


@register_approach
class ProactiveHT_MR(ProactiveHTBase):
    ID = 3
    NAME = "PHT-MR"
    def __init__(self, window_size=1000, approach=3, stats_reset_mode=0, alpha=0.01):
        super().__init__(window_size, approach)
        
        self.n_closest_points = alpha*window_size
        self.stats_reset_mode = stats_reset_mode

    
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
                for feature, threshold in decision_boundaries:
                    node, restrictions = self.find_node_and_restrictions(self.model._root, feature, threshold)

                    new_threshold = self.update(
                        X.iloc[idx - self.window_size:idx, :],
                        y.iloc[idx - self.window_size:idx],
                        feature,
                        threshold, 
                        restrictions
                    )
                
                decision_boundaries = self.extract_rules(self.model)
                acumulative_DB.append(decision_boundaries)
                list_models.append(copy.deepcopy(self.model))
                


        decision_boundaries = self.extract_rules(self.model)
        for feature, threshold in decision_boundaries:
            node, restrictions = self.find_node_and_restrictions(self.model._root, feature, threshold)
            new_threshold = self.update(
                X.iloc[idx - self.window_size:idx, :],
                y.iloc[idx - self.window_size:idx],
                feature,
                threshold,
                restrictions
            )
        decision_boundaries = self.extract_rules(self.model)
        acumulative_DB.append(decision_boundaries)
        list_models.append(copy.deepcopy(self.model))



        return acumulative_DB, list_models, predictions
                   

    def update(self, X, y, feature, threshold, restrictions=None):
        rotation = False
        new_threshold = threshold
        new_feature = feature

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
            return threshold  

        points_region1 = []
        points_region2 = []
        for idx, (index, row) in enumerate(filtered_X.iterrows()):
            if row[feature] <= threshold:
                points_region1.append(row)
            else:
                points_region2.append(row)

        if not points_region1 or not points_region2:
            return threshold 

        centroid_region1 = np.mean(points_region1, axis=0)
        centroid_region2 = np.mean(points_region2, axis=0)

        vector_centroids = centroid_region1 - centroid_region2
        norm_centroids = np.linalg.norm(vector_centroids)

        if norm_centroids == 0:
            return threshold  

        normal_vector = np.zeros(filtered_X.shape[1])
        normal_vector[filtered_X.columns.get_loc(feature)] = 1

        if np.dot(vector_centroids, normal_vector) < 0:
            normal_vector = -normal_vector  

        cos_theta = np.dot(vector_centroids, normal_vector) / (norm_centroids * np.linalg.norm(normal_vector))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        angle_hyperplane = 90 - angle

        if angle_hyperplane < 45:
            variance_per_dim = np.var(filtered_X.values, axis=0)
            new_feature_idx = np.argmax(variance_per_dim)
            new_feature = filtered_X.columns[new_feature_idx]

            
            new_threshold = (centroid_region1[new_feature_idx] + centroid_region2[new_feature_idx]) / 2

            if centroid_region1[new_feature_idx] > centroid_region2[new_feature_idx]:
                rotation = True

        # print("Decision boundary: ", new_feature, new_threshold)
        closest_points = self.find_filtered_closest_points(filtered_X, filtered_y, self.n_closest_points, (new_feature, new_threshold))

        closest_points_region1 = [point for point, label, _, _ in closest_points if point[new_feature] > new_threshold]
        # print("Number of points in region 1: ", len(closest_points_region1))
        closest_points_region2 = [point for point, label, _, _ in closest_points if point[new_feature] <= new_threshold]
        # print("Num of points in region 2: ", len(closest_points_region2))


        mean_region1 = np.mean([point[new_feature] for point in closest_points_region1]) if closest_points_region1 else threshold
        mean_region2 = np.mean([point[new_feature] for point in closest_points_region2]) if closest_points_region2 else threshold

        # print("Mean region 1: ", mean_region1)
        # print("Mean region 2: ", mean_region2)

        mean = (mean_region1 + mean_region2) / 2
        new_threshold = mean

 
        self.update_tree(feature, threshold, new_threshold, new_feature, rotation)

        # print("New decision boundary: ", new_feature, new_threshold)

        return new_threshold



    