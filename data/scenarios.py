import sys
import os

# Add parent directory to path to allow imports when running as script
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.csv_utils import save_stream
    from .generators import RandomRBFMC
except ImportError:
    # When running as script, use absolute imports
    from utils.csv_utils import save_stream
    from data.generators import RandomRBFMC

import utils.utils as ut
from mpi4py import MPI
import numpy as np
import pandas as pd
from utils.utils import Centroid


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

                df_aux = save_stream(aux_stream, f"temp_aux_{i}.csv", self.size, save=True)
                df = pd.concat([df, df_aux], axis=0)

        df.to_csv(self.output_file, index=False, header=False)

        for i in range(n):
            file = f"temp_aux_{i}.csv"
            if os.path.exists(file):
                os.remove(file)

        return df

    def multi_chase_drift(self):
        """
        Generate a stream with multiple Chase drifts occurring at random positions.
        Each drift lasts for 250 samples and there are between 4-6 drifts total.
        """
        n_drifts = np.random.randint(4, 7)
        drift_duration = 250
        
        total_stream_size = self.size
        min_gap = 100 
        
        drift_positions = []
        max_start_position = total_stream_size - drift_duration
        
        for i in range(n_drifts):
            while True:
                start_pos = np.random.randint(0, max_start_position)
                end_pos = start_pos + drift_duration
                
                valid_position = True
                for existing_start, existing_end in drift_positions:
                    if not (end_pos < existing_start - min_gap or start_pos > existing_end + min_gap):
                        valid_position = False
                        break
                
                if valid_position:
                    drift_positions.append((start_pos, end_pos))
                    break
                    
                if len([pos for pos in drift_positions]) > 0:
                    min_gap = max(50, min_gap - 10)
        
        drift_positions.sort()
        
        print(f"Generating {n_drifts} Chase drifts at positions: {drift_positions}")
        
        centroids = self.generate_centroids()
        
        self.stream = RandomRBFMC(
            32, 42, n_classes=self.n_class, n_features=self.dimension, 
            n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1, 
            std_dev=3, manual_centroid=centroids
        )
        
        df_complete = pd.DataFrame()
        current_position = 0
        current_centroids = [centroid for centroid in centroids]  
        
        n_affected = int(self.n_cluster_per_class * self.ratio_affected)
        if n_affected == 0:
            n_affected = 1
        
        for drift_idx, (drift_start, drift_end) in enumerate(drift_positions):
            if current_position < drift_start:
                stable_size = drift_start - current_position
                stable_stream = RandomRBFMC(
                    32, 42, n_classes=self.n_class, n_features=self.dimension,
                    n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1,
                    std_dev=3, manual_centroid=current_centroids
                )
                df_stable = save_stream(stable_stream, f"temp_stable_{drift_idx}.csv", stable_size, save=False)
                df_complete = pd.concat([df_complete, df_stable], axis=0)
                current_position = drift_start
            
            direction = np.random.normal(0, 1, self.dimension)
            direction /= np.linalg.norm(direction)
            module = np.random.uniform(1, 3)
            drift_vector = direction * module
            
            new_centroids = []
            for i in range(self.n_class):
                for j in range(self.n_cluster_per_class):
                    if j < n_affected:
                        old_position = np.array(current_centroids[i * self.n_cluster_per_class + j].centre)
                        new_position = old_position + drift_vector
                        new_centroid = Centroid(new_position.tolist(), i, 0.1)
                    else:
                        new_centroid = current_centroids[i * self.n_cluster_per_class + j]
                    new_centroids.append(new_centroid)
            
            drift_stream = RandomRBFMC(
                32, 42, n_classes=self.n_class, n_features=self.dimension,
                n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1,
                std_dev=3, manual_centroid=current_centroids
            )
            
            for class_idx in range(self.n_class):
                destinations = []
                for j in range(n_affected):
                    centroid_idx = class_idx * self.n_cluster_per_class + j
                    dest_position = new_centroids[centroid_idx].centre
                    destinations.append(dest_position)
                
                drift_stream.incremental_moving_fixed_clusters(
                    class_idx, n_affected, width=drift_duration, destination=destinations
                )
            
            df_drift = save_stream(drift_stream, f"temp_drift_{drift_idx}.csv", drift_duration, save=False)
            df_complete = pd.concat([df_complete, df_drift], axis=0)
    
            current_centroids = new_centroids
            current_position = drift_end
        
        if current_position < total_stream_size:
            remaining_size = total_stream_size - current_position
            final_stream = RandomRBFMC(
                32, 42, n_classes=self.n_class, n_features=self.dimension,
                n_centroids=self.n_cluster_per_class*self.n_class, min_distance=1,
                std_dev=3, manual_centroid=current_centroids
            )
            df_final = save_stream(final_stream, f"temp_final.csv", remaining_size, save=False)
            df_complete = pd.concat([df_complete, df_final], axis=0)
        
        for drift_idx in range(n_drifts):
            for temp_file in [f"temp_stable_{drift_idx}.csv", f"temp_drift_{drift_idx}.csv"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        if os.path.exists("temp_final.csv"):
            os.remove("temp_final.csv")

        if self.save:
            df_complete.to_csv(self.output_file, index=False, header=False)
            print(f"Multi-Chase drift dataset saved in {self.output_file}")
        
        return df_complete

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

        WIDTH = (self.size // self.n_class) // self.n_cluster_per_class
        #WIDTH = 10000
        if hasattr(self, 'width_factor'):
            WIDTH = int(WIDTH * self.width_factor)

        for i in range(self.n_class):
            shift = np.random.uniform(1, 6)
            self.stream.split_cluster_fixed_clusters(i, i, shift, WIDTH, n_affected, 3)

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

            df_aux = save_stream(aux_stream, "temp_clock_drift.csv", self.size, save=False)
            df = pd.concat([df, df_aux], axis=0) if not df.empty else df_aux

            centroids[N] = Centroid(new_position.tolist(), centroids[N].class_label, 0.1)

        return df
        


    '''
    Depending on the type of scenario provided we can generate 6 different simple drifts:
        0. (Chase) Clusters moving following the same path.
        1. (Cross) Clusters moving in opposite directions and meeting in the middle. 
        2. (Split) Clusters splitting in several portions with random directions.
        3. (Merge) Clusters from the same class merging into one, in the middle of the path.
        4. (Clock*) Clusters moving in a circle around a point. (*) This scenarios was not included in the research but it was used for development purposes.
        5. (Multi-Chase) Multiple Chase drifts occurring at random positions in the stream. Abrupt scenario
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
        elif self.type == 5:
            self.drifted_dataframe = self.multi_chase_drift()


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
    





def generate_scenarios(config, width_factor=1.0, output_dir="datasets"):

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
                        path = f"{output_dir}/seed{seed}/dim{dimension}/class{n_class}/cluster{n_cluster_per_class}/ratio{ratio}/"
                        if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)

                        for scenario_type in drift_types:
                            scenario_path = path + f"scenario_{scenario_type}/"
                            if not os.path.exists(scenario_path):
                                os.makedirs(scenario_path, exist_ok=True)

                            output_file = scenario_path + "data.csv"

                            print(f"Process {rank}: generating scenario {scenario_type} for seed={seed}, dim={dimension}, class={n_class}, cluster={n_cluster_per_class}, ratio={ratio}, velocity_factor={width_factor}.")

                            base_size = 50000
                            adjusted_size = int(base_size * width_factor) if width_factor != 1.0 else base_size
                            drift = DataDrift(output_file, n_class, n_cluster_per_class, ratio, dimension, scenario_type, True, adjusted_size, seed)
                            drifted_dataframe = drift.drifting()

                            drifted_dataframe.to_csv(output_file, index=False, header=False)
                            print(f"Process {rank}: scenario {scenario_type} saved in {output_file}.")


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Generate datasets with different drift velocities")
    parser.add_argument("--config", default="config.yml", help="Configuration file")
    parser.add_argument("--width_factor", type=float, default=1.0, 
                       help="Drift velocity factor (0.5= fast, 1.0= normal)")
    parser.add_argument("--output_dir", default="datasets", 
                       help="Output directory for datasets")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate scenarios with custom parameters
    generate_scenarios(config, width_factor=args.width_factor, output_dir=args.output_dir)
    
    print(f"Datasets generated with velocity factor={args.width_factor} in {args.output_dir}/")