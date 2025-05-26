from __future__ import annotations
from river.datasets.synth import RandomRBF
from river.datasets.synth.random_rbf import Centroid, random_index_based_on_weights
import random
import math
import numpy as np
from typing import List


class RandomRBFMC(RandomRBF):
    def __init__(
        self,
        seed_model: int = None,
        seed_sample: int = None,
        n_classes: int = 2,
        n_features: int = 10,
        n_centroids: int = 50,
        min_distance: float = 0.3,
        std_dev: float = 0.1,
        manual_centroid: list[Centroid] = None,
    ):
        super().__init__(seed_model, seed_sample, n_classes, n_features, n_centroids)
        self.min_distance = min_distance
        self.std_dev = std_dev
        self.rng_model = random.Random(self.seed_model)
        self.moving_centroids: list[MovingCentroid] = []
        if not manual_centroid:
            self._generate_centroids()
        else:
            self.centroids = manual_centroid
            self.centroid_weights = [1 for _ in range(len(self.centroids))]

    def _compute_nearest(self, centroid: List[float]):
        import math

        if centroid == None:
            return False
        for c in self.centroids:
            if c.centre != None:
                dist = math.dist(centroid, c.centre)
                if dist < self.min_distance:
                    return False
        return True

    def _generate_centroids(self):
        """Generates centroids

        Sequentially creates all the centroids, choosing at random a center,
        a label, a standard deviation and a weight.

        """

        self.centroids = []
        self.centroid_weights = []
        classes_assinged = [i for i in range(self.n_classes)]
        for i in range(self.n_centroids):
            self.centroids.append(Centroid())
            rand_centre = None
            while not self._compute_nearest(rand_centre):
                rand_centre = []
                for j in range(self.n_num_features):
                    rand_centre.append(self.rng_model.uniform(-1, 1))

            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = classes_assinged.pop(
                self.rng_model.randint(0, len(classes_assinged) - 1)
            )
            if len(classes_assinged) == 0:
                classes_assinged = [i for i in range(self.n_classes)]
            self.centroids[i].std_dev = self.std_dev
            self.centroid_weights.append(1)

    def _generate_sample(self, rng_sample: random.Random):
        idx = random_index_based_on_weights(self.centroid_weights, rng_sample)
        current_centroid = self.centroids[idx]

        moving_centers = [m.c for m in self.moving_centroids]
        if current_centroid in moving_centers:
            # print("moving centroid selected")
            self.moving_centroids[moving_centers.index(current_centroid)].update()
        att_vals = dict()
        magnitude = 0.0
        for i in range(self.n_features):
            att_vals[i] = (rng_sample.uniform(-1, 1) * 2.0) - 1.0
            magnitude += att_vals[i] * att_vals[i]
        magnitude = magnitude**0.5
        desired_mag = rng_sample.gauss(0, 1) * current_centroid.std_dev
        scale = desired_mag / magnitude
        x = {
            i: current_centroid.centre[i] + att_vals[i] * scale
            for i in range(self.n_features)
        }
        y = current_centroid.class_label
        return x, y

    def swap_clusters(self, class_1: int, class_2: int, proportions: float = 0.5):
        class_1_centroids = [c for c in self.centroids if c.class_label == class_1]
        class_2_centroids = [c for c in self.centroids if c.class_label == class_2]

        class_1_centroid = self.rng_model.sample(
            class_1_centroids, k=int(len(class_1_centroids) * proportions)
        )
        class_2_centroid = self.rng_model.sample(
            class_2_centroids, k=int(len(class_2_centroids) * proportions)
        )

        for c in class_1_centroid:
            c.class_label = class_2
        for c in class_2_centroid:
            c.class_label = class_1

    def remove_cluster(self, class_1: int, proportions: float = 0.5):
        class_1_centroids = [c for c in self.centroids if c.class_label == class_1]
        to_be_removed_clusters = self.rng_model.sample(
            class_1_centroids, k=int(len(class_1_centroids) * proportions)
        )
        for c in to_be_removed_clusters:
            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)

    def add_cluster(self, class_1: int, weight: float = 1.0, manual_centroid: list = None):
        self.centroids.append(Centroid())
        i = len(self.centroids) - 1
        rand_centre = None
        while not self._compute_nearest(rand_centre):
            rand_centre = []
            for j in range(self.n_num_features):
                rand_centre.append(self.rng_model.uniform(-1, 1))

        if manual_centroid:
            rand_centre = manual_centroid
            
        self.centroids[i].centre = rand_centre
        self.centroids[i].class_label = class_1
        self.centroids[i].std_dev = self.std_dev
        self.centroid_weights.append(weight)

    def shift_cluster(self, class_1: int, proportions: float = 1.0):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        for i in range(int(len(class_centroids) * proportions)):
            self.add_cluster(class_1=class_1)
        to_be_removed_clusters = self.rng_model.sample(
            class_centroids, k=int(len(class_centroids) * proportions)
        )
        for c in to_be_removed_clusters:
            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)

    def split_cluster(
        self,
        class_1: int,
        class_2: int,
        shift_mag: float = 0.3,
        width: int = 1,
        proportion: float = 0.5,
        num_splits: int = 2  
    ):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        for j in range(0, int(len(class_centroids) * proportion)):
            c = self.rng_model.choice(class_centroids)
            class_centroids.remove(c)
            centroid = c.centre
            shift = self.rng_model.uniform(0.1, shift_mag)

            start_center = centroid.copy()
            
            new_centers = []
            for i in range(num_splits):
                direction = np.random.normal(size=len(start_center))
                direction /= np.linalg.norm(direction) 

                new_center = [att + direction[j] * shift for j, att in enumerate(centroid)]
                new_centers.append(new_center)

            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)

            for i, center in enumerate(new_centers):
                new_centroid = Centroid()
                new_centroid.centre = start_center  
                new_centroid.class_label = class_1 if i % 2 == 0 else class_2  
                new_centroid.std_dev = c.std_dev  
                self.centroids.append(new_centroid)
                self.centroid_weights.append(1)

                self.moving_centroids.append(MovingCentroid(new_centroid, new_centroid.centre, center, width))

    def generate_directions_2d(self, num_splits, rng=None, random_offset=True):
        if rng is None:
            rng = np.random

        angles = np.linspace(0, 2 * math.pi, num_splits, endpoint=False)
        
        offset = rng.uniform(0, 2 * math.pi) if random_offset else 0.0
        
        directions = []
        for angle in angles:
            a = angle + offset
            directions.append(np.array([math.cos(a), math.sin(a)]))
        
        return directions

    def split_cluster_fixed_clusters(
        self,
        class_1: int,
        class_2: int,
        shift_mag: float = 0.3,
        width: int = 1,
        n_affected_clusters: int = 1,
        num_splits: int = 2
    ):
        
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        for j in range(n_affected_clusters):
            c = class_centroids[j]
            centroid = c.centre
            shift = self.rng_model.uniform(1, shift_mag)

            start_center = centroid.copy()
           
            if self.n_num_features == 2:
                directions = self.generate_directions_2d(num_splits, rng=self.rng_model)
            else:
                directions = []
                for i in range(num_splits):
                    direction = np.random.normal(size=len(start_center))
                    direction /= np.linalg.norm(direction) 
                    directions.append(direction)

            new_centers = []
            for direction in directions:
                new_center = [att + direction[j] * shift for j, att in enumerate(centroid)]
                new_centers.append(new_center)

            index = self.centroids.index(c)
            self.centroids.remove(c)
            self.centroid_weights.pop(index)

            for i, center in enumerate(new_centers):
                new_centroid = Centroid()
                new_centroid.centre = start_center  
                new_centroid.class_label = class_1 
                new_centroid.std_dev = c.std_dev  
                self.centroids.append(new_centroid)
                self.centroid_weights.append(1)

                self.moving_centroids.append(MovingCentroid(new_centroid, new_centroid.centre, center, width))

    def merge_cluster(
        self, class_1: int, class_2: int, width: int = 1, proportion: float = 0.5
    ):
        class_centroids_1 = [c for c in self.centroids if c.class_label == class_1]
        if len(class_centroids_1) % 2 != 0:
            class_centroids_1.pop(0)

        class_centroids_2 = [c for c in self.centroids if c.class_label == class_2]
        if len(class_centroids_2) % 2 != 0:
            class_centroids_2.pop(0)

        n_merges = int(len(class_centroids_1) * proportion / 2)

        for j in range(0, n_merges):
            c_1 = self.rng_model.choice(class_centroids_1)
            class_centroids_1.remove(c_1)
            if class_1 == class_2:
                c_2 = self.rng_model.choice(class_centroids_1)
                class_centroids_1.remove(c_2)
            else:
                c_2 = self.rng_model.choice(class_centroids_2)
                class_centroids_2.remove(c_2)

            center = [
                (c_1.centre[i] + c_2.centre[i]) / 2 for i in range(0, len(c_1.centre))
            ]

            self.moving_centroids.append(MovingCentroid(c_1, c_1.centre, center, width))
            self.moving_centroids.append(MovingCentroid(c_2, c_2.centre, center, width))


    def merge_cluster_fixed_clusters(
        self, class_1: int, class_2: int, width: int = 1, n_affected_clusters: int = 1
    ):
        if class_1 != class_2:
            raise ValueError("Merged operation is only allowed for the same class")
        
        class_centroids_1 = [c for c in self.centroids if c.class_label == class_1]

        to_be_merged = class_centroids_1[:n_affected_clusters]
        i = 0
        if n_affected_clusters % 2 != 0 and n_affected_clusters >= 3:
            c1, c2, c3 = to_be_merged[0], to_be_merged[1], to_be_merged[2]
            center_triple = [
                (c1.centre[k] + c2.centre[k] + c3.centre[k]) / 3
                for k in range(len(c1.centre))
            ]
            self.moving_centroids.append(MovingCentroid(c1, c1.centre, center_triple, width))
            self.moving_centroids.append(MovingCentroid(c2, c2.centre, center_triple, width))
            self.moving_centroids.append(MovingCentroid(c3, c3.centre, center_triple, width))
            i = 3  

        while i < len(to_be_merged) - 1:
            c_1 = to_be_merged[i]
            c_2 = to_be_merged[i + 1]
            center_pair = [
                (c_1.centre[k] + c_2.centre[k]) / 2 for k in range(len(c_1.centre))
            ]
            self.moving_centroids.append(MovingCentroid(c_1, c_1.centre, center_pair, width))
            self.moving_centroids.append(MovingCentroid(c_2, c_2.centre, center_pair, width))
            i += 2


    def incremental_moving(
        self, class_1: int, proportions: float = 1.0, width: int = 100, destination=None
    ):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]
        to_be_moved = self.rng_model.sample(
            class_centroids, k=int(len(class_centroids) * proportions)
        )

        # print(to_be_moved)
        for i in range(int(len(to_be_moved))):
            rand_centre = None
            if destination:
                rand_centre = destination
            else:
                while not self._compute_nearest(rand_centre):
                    rand_centre = []
                    for j in range(self.n_num_features):
                        rand_centre.append(self.rng_model.uniform(-1, 1))

            # print(rand_centre)

            self.moving_centroids.append(
                MovingCentroid(
                    to_be_moved[i], to_be_moved[i].centre, rand_centre, width
                )
            )

        # print(self.moving_centroids)

    def incremental_moving_fixed_clusters(
        self, class_1: int, n_affected_clusters, width: int = 100, destination=None
    ):
        class_centroids = [c for c in self.centroids if c.class_label == class_1]

        to_be_moved = class_centroids[:n_affected_clusters]

        if len(to_be_moved) != len(destination):
            print(len(to_be_moved))
            print(len(destination))
            raise ValueError("The number of centroids to be moved and the number of destination centroids must be the same")
        
        # print(to_be_moved)
        for i in range(int(len(to_be_moved))):
            rand_centre = None
            if destination:
                rand_centre = destination[i]
            else:
                while not self._compute_nearest(rand_centre):
                    rand_centre = []
                    for j in range(self.n_num_features):
                        rand_centre.append(self.rng_model.uniform(-1, 1))

            # print(rand_centre)

            self.moving_centroids.append(
                MovingCentroid(
                    to_be_moved[i], to_be_moved[i].centre, rand_centre, width
                )
            )

        # print(self.moving_centroids)


    def __iter__(self):
        rng_sample = random.Random(self.seed_sample)

        while True:
            x, y = self._generate_sample(rng_sample)
            yield x, y


class MovingCentroid:
    def __init__(self, c: Centroid, centre_1: list, centre_2: list, width: int) -> None:
        self.c = c
        self.centre_1 = centre_1.copy()
        self.centre_2 = centre_2
        self.width = width
        self.instanceCount = 0

    def update(self):
        try:
            factor = min(self.instanceCount / self.width, 1)
        except:
            factor = 0

        for i in range(0, len(self.centre_1)):
            self.c.centre[i] = (1 - factor) * self.centre_1[i] + (
                factor
            ) * self.centre_2[i]

        self.instanceCount += 1


# 0.99 -> 0
# 0.01 -> self.width
