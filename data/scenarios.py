from .csv import save_stream
import utils.utils as ut
from mpi4py import MPI
import numpy as np
import pandas as pd
import os
from utils.utils import Centroid
from .generators import RandomRBFMC


class DataDrift:
    def __init__(self, file, n_class, n_cluster_per_class, ratio_affected, dimension, type, save, size, seed = 3,centroids = None):
        self.output_file = file
        self.n_class = n_class
        self.n_cluster_per_class = n_cluster_per_class
    
        if ratio_affected*n_cluster_per_class - int(ratio_affected*n_cluster_per_class) != 0:
            ratio_affected = (int(ratio_affected*n_cluster_per_class)+1)/n_cluster_per_class

        self.ratio_affected = ratio_affected
        self.dimension = dimension
        self.type = type
        self.save = save
        self.size = size
        self.centroids = centroids
        self.seed = seed
        np.random.seed(seed)
        

    def set_stream(self, stream):
        self.stream = stream
    
    def reset_stream(self, centroids=None):
        self.stream = RandomRBFMC(32, 42, n_classes=2, n_features=self.dimension, n_centroids=2, min_distance=1, std_dev=3, manual_centroid=centroids)

    def save_stream(self):
        if self.save:
            print(f"Stream saved in {self.output_file}")
        
        return save_stream(self.stream, self.output_file, self.size, self.save)


    def generate_centroids(self, min_distance=2.0, coord_range=(-3, 3), std_dev=0.1):
        centroids = []

        for class_label in range(self.n_class):
            class_centroids = []
            while len(class_centroids) < self.n_cluster_per_class:
                centroid_coords = [np.random.uniform(*coord_range) for _ in range(self.dimension)]
                is_valid = True
          
                for existing_centroid in class_centroids:
                    distance = np.linalg.norm(np.array(centroid_coords) - np.array(existing_centroid.centre))
                    if distance < min_distance:
                        is_valid = False
                        break

                if is_valid:
                    class_centroids.append(Centroid(centroid_coords, class_label, std_dev))

            centroids.extend(class_centroids)

        return centroids
    
    def chase(self):
        n = np.random.randint(1, 6)
        
        vectors = []
        for i in range(n):
            direction = np.random.normal(0, 1, self.dimension)
            direction /= np.linalg.norm(direction)
            module = np.random.uniform(1, 3)
            vector = direction * module
            vectors.append(vector)
        
        
        centroids = self.generate_centroids()

        n_affected = int(self.n_cluster_per_class * self.ratio_affected)
        if n_affected == 0:
            n_affected = 1
            
        positions = []
        initial_positions = []
        for i in range(self.n_class):
            for j in range(n_affected):
                initial_positions.append(centroids[i * self.n_cluster_per_class + j].centre.copy())

        positions.append(initial_positions)

        for vector in vectors:
            new_positions = []
            for j in range(self.n_class * n_affected):
                new_position = positions[-1][j] + vector  
                new_positions.append(new_position.tolist())
            positions.append(new_positions)

        df = pd.DataFrame()

        WIDTH = (self.size // self.n_class) // self.n_cluster_per_class

        for i in range(n):
            if i == 0:
                self.stream = RandomRBFMC(
                    32, 42, n_classes=self.n_class, n_features=self.dimension, n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1, std_dev=3, manual_centroid=centroids
                )
                for j in range(self.n_class):
                    self.stream.incremental_moving_fixed_clusters(j, n_affected, width= WIDTH, destination=positions[i+1][j*n_affected:(j)*n_affected+n_affected])

                df = self.save_stream()
            else:
                new_centroids = []
                for j in range(self.n_class):
                    for k in range(self.n_cluster_per_class):
                        if k < n_affected:
                            centroid = Centroid(positions[i][j*n_affected+k], j, 0.1)
                        else:
                            centroid = centroids[j*self.n_cluster_per_class+k]
                        new_centroids.append(centroid)
            
                aux_stream = RandomRBFMC(
                    32, 42, n_classes=self.n_class, n_features=self.dimension, n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1, std_dev=3, manual_centroid=new_centroids
                )

                for j in range(self.n_class):
                    aux_stream.incremental_moving_fixed_clusters(j, n_affected, width= WIDTH, destination=positions[i+1][j*n_affected:(j)*n_affected+n_affected])

                df_aux = save_stream(aux_stream, "results/aux_"+str(i)+".csv", self.size, save=True)
                df = pd.concat([df, df_aux], axis=0)

        df.to_csv(self.output_file, index=False, header=False)

        for i in range(n):
            file = "results/aux_"+str(i)+".csv"
            if os.path.exists(file):
                os.remove(file)

        return df

    def cross(self):

        centroids = self.generate_centroids(min_distance=1.5)
        
        n_affected = int(self.n_cluster_per_class * self.ratio_affected)

        initial_positions = []
        for i in range(self.n_class):
            for j in range(n_affected):
                initial_positions.append(centroids[i * self.n_cluster_per_class + j].centre.copy())

        center = np.zeros(self.dimension)
        for i in range(self.n_class * n_affected):
            center += initial_positions[i]
        center /= self.n_class * n_affected

        destinations = []

        WIDTH = (self.size // self.n_class) // self.n_cluster_per_class
        for i in range(self.n_class * n_affected):
            destination = 2 * center - initial_positions[i]
            destinations.append(destination)


        self.stream = RandomRBFMC(
                    32, 42, n_classes=self.n_class, n_features=self.dimension, n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1, std_dev=3, manual_centroid=centroids
                )
        
        for i in range(self.n_class):
            self.stream.incremental_moving_fixed_clusters(i, n_affected, width= WIDTH, destination=destinations[i*n_affected:(i)*n_affected+n_affected])

        df = self.save_stream()
    
        return df

    def split(self):

        n = np.random.randint(1, 6)
        while n == 1:
            n = np.random.randint(1, 6)


        shift = np.random.uniform(1,6)
        
        centroids = self.generate_centroids(min_distance=1)

        self.stream =  RandomRBFMC(
                    32, 42, n_classes=self.n_class, n_features=self.dimension, n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1, std_dev=3, manual_centroid=centroids
                )
        
        n_affected = int(self.n_cluster_per_class * self.ratio_affected)
        
        for i in range(self.n_class):
            shift = np.random.uniform(1, 6)
            self.stream.split_cluster_fixed_clusters(i, i, shift, 10000, n_affected, 3)

        df = self.save_stream()

        return df

    def merge(self):
        centroids = self.generate_centroids(min_distance=1.5)
        self.stream = RandomRBFMC(
                    32, 42, n_classes=self.n_class, n_features=self.dimension, n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1, std_dev=3, manual_centroid=centroids
                )
        
        n_affected = int(self.n_cluster_per_class * self.ratio_affected)
        
        WIDTH = (self.size // self.n_class) // self.n_cluster_per_class
        for i in range(self.n_class):   
            self.stream.merge_cluster_fixed_clusters(i, i, WIDTH, n_affected)

        df = self.save_stream()

        return df
    
    def ghosting(self):

        df = self.save_stream()
        
        self.stream.remove_cluster(1, 1)

        df_aux = self.save_stream()
        df = pd.concat([df, df_aux], axis=0)

        return df

    def birth(self):
        df = self.save_stream()
        
        self.stream.add_cluster(2,0.4,[1.75, -1.75])

        df_aux = self.save_stream()
        df = pd.concat([df, df_aux], axis=0)

        return df


    def clock(self, steps=6):
        df = pd.DataFrame()
        centroids = self.generate_centroids(min_distance=1.2)
        N = np.random.randint(0, len(centroids))
        center_to_move = np.array(centroids[N].centre)
        center_of_rotation = np.array(centroids[1 - N].centre)

        
        vector_to_rotate = center_to_move - center_of_rotation
        distance = np.linalg.norm(vector_to_rotate)  

        random_vector = np.random.randn(self.dimension)
        random_vector -= random_vector.dot(vector_to_rotate) / np.linalg.norm(vector_to_rotate) ** 2 * vector_to_rotate
        random_vector /= np.linalg.norm(random_vector)  

        angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        points_on_circle = []

        for angle in angles:
            rotated_vector = (
                np.cos(angle) * vector_to_rotate
                + np.sin(angle) * random_vector
            )
            new_position = center_of_rotation + rotated_vector
            points_on_circle.append(new_position)

        points_on_circle.append(points_on_circle[0])

        for i, new_position in enumerate(points_on_circle):
            aux_stream = RandomRBFMC(
                32,
                42,
                n_classes=2,
                n_features=self.dimension,
                n_centroids=2,
                min_distance=1,
                std_dev=3,
                manual_centroid=centroids,
            )
            aux_stream.incremental_moving(
                centroids[N].class_label, proportions=1, width=25000, destination=new_position.tolist()
            )

            df_aux = save_stream(aux_stream, "results/clock_drift/df_clock_drift.csv", self.size, save=False)
            df = pd.concat([df, df_aux], axis=0) if not df.empty else df_aux

            centroids[N] = Centroid(new_position.tolist(), centroids[N].class_label, 0.1)

        return df
        


    '''
    Depending on the type of scenario provided we can generate 5 different simple drifts:
        0. (Chase) Clusters moving following the same path.
        1. (Cross) Clusters moving in opposite directions and meeting in the middle. 
        2. (Split) Clusters splitting in several portions with random directions.
        3. (Merge) Clusters from the same class merging into one, in the middle of the path.
        5. (Clock*) Clusters moving in a circle around a point. (*) This scenarios was not included in the research but it was used for development purposes.
    '''
    def drifting(self):
        if self.type == 0:
            self.drifted_dataframe = self.chase()
        elif self.type == 1:
            self.drifted_dataframe = self.cross()
        elif self.type == 2:
            self.drifted_dataframe = self.split()
        elif self.type == 3:
            self.drifted_dataframe = self.merge()
        elif self.type == 4:
            self.drifted_dataframe = self.clock(steps=5)


        '''
        # Appearance of a new cluster (abrupt drift)
        elif self.type == 5:
            centroids = [Centroid([-1.75, 1.75], 0, 0.1), Centroid([0, 0], 1, 0.1), Centroid([1.75, -1.75], 2, 0.1)]
            self.set_stream(RandomRBFMC(32, 42, n_classes=3, n_features=self.dimension, n_centroids=3, min_distance=1, std_dev=0.1, manual_centroid=centroids))

            self.drifted_dataframe = self.ghosting()
        # Birth of a new cluster (abrupt drift)
        elif self.type == 6:
            centroids = [Centroid([-1.75, 1.75], 0, 0.1), Centroid([0, 0], 1, 0.1)]
            self.set_stream(RandomRBFMC(32, 42, n_classes=3, n_features=self.dimension, n_centroids=2, min_distance=1, std_dev=0.1, manual_centroid=centroids))

            self.drifted_dataframe = self.birth()
        '''

        return self.drifted_dataframe
    





def generate_scenarios(config):
    
    experiment_cfg = config['experiment']
    seeds = experiment_cfg['seeds']
    dims = experiment_cfg['dims']
    n_classes = experiment_cfg['n_classes']
    n_clusters_per_class = experiment_cfg['n_clusters_per_class']
    ratio_affected = experiment_cfg['ratio_affected']
    drift_types = experiment_cfg['drift_types']
    window_size = experiment_cfg['window_size']

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return

    for seed in seeds:
        for dimension in dims:
            for n_class in n_classes:
                for n_cluster_per_class in n_clusters_per_class:
                    for ratio in ratio_affected:
                        path = f"datasets/seed{seed}/dim{dimension}/class{n_class}/cluster{n_cluster_per_class}/ratio{ratio}/"
                        if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)

                        for scenario_type in drift_types:
                            scenario_path = path + f"scenario_{scenario_type}/"
                            if not os.path.exists(scenario_path):
                                os.makedirs(scenario_path, exist_ok=True)

                            output_file = scenario_path + "data.csv"

                            print(f"Process {rank}: generating scenario {scenario_type} for seed={seed}, dim={dimension}, class={n_class}, cluster={n_cluster_per_class}, ratio={ratio}.")

                            drift = DataDrift(output_file, n_class, n_cluster_per_class, ratio, dimension, scenario_type, True, 50000, seed)
                            drifted_dataframe = drift.drifting()

                            drifted_dataframe.to_csv(output_file, index=False, header=False)
                            print(f"Process {rank}: scenario {scenario_type} saved in {output_file}.")