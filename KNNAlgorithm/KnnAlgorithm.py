import random
from collections import Counter

import numpy as np

from KNNAlgorithm.CalculateDistance.Factory.EuclideanDistanceCreator import EuclideanDistanceCreator


class KnnAlgorithm:
    def __init__(self, k, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def _get_neighbors(self, x):
        distance_creator = EuclideanDistanceCreator()
        distance_strategy = distance_creator.create_distance()
        distances = [distance_strategy.calculate(x, x_train_sample) for x_train_sample in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        return self.y_train[k_indices]

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            neighbors = self._get_neighbors(x)
            label_counts = Counter(neighbors)
            most_common = label_counts.most_common()
            max_count = most_common[0][1]
            candidates = [label for label, count in most_common if count == max_count]
            predictions.append(random.choice(candidates))
        return np.array(predictions)