import random
from collections import Counter

import numpy as np

from KNNAlgorithm.CalculateDistance.Factory.EuclideanDistanceCreator import EuclideanDistanceCreator


class KnnAlgorithm:
    """
    Initialize the KNN class with training data and the number of neighbors (k).
    :param k: Number of neighbors to consider.
    :param x_train: Training data (features).
    :param y_train: Training data (labels).
    """
    def __init__(self, k, x_train, y_train):
        self.x_train = x_train              # Store the training data (features).
        self.y_train = y_train              # Store the labels for the training data.
        self.k = k                          # Number of neighbors to consider.

    """
    Find the k nearest neighbors of a given point.
    :param x: The point for which to find neighbors.
    :return: The labels of the k nearest neighbors.
    """
    def get_neighbors(self, x):
        distance_creator = EuclideanDistanceCreator()                   # Create an instance of the factory for calculating Euclidean distance.
        distance_strategy = distance_creator.create_distance()          # Create the distance calculation strategy (EuclideanDistance).
        distances = [distance_strategy.calculate(x, x_train_sample)     # Calculate the distances between point x and all training samples.
                     for x_train_sample in self.x_train]
        k_indices = np.argsort(distances)[:self.k]                      # Get the indices of the k closest points, sorted by ascending distance.
        return self.y_train[k_indices]                                  # Return the labels of the k closest points.

    """
    Predict labels for a set of test data.
    :param x_test: Test data for which to predict labels.
    :return: Array of predicted labels.
    """
    def predict(self, x_test):
        predictions = []                                                                    # Initialize an empty list to store predictions.
        for x in x_test:                                                                    # For each point in the test data
            neighbors = self.get_neighbors(x)  # Get the labels of the nearest neighbors.
            label_counts = Counter(neighbors)                                               # Count the frequency of each label among the neighbors.
            most_common = label_counts.most_common()                                        # Get the most common labels sorted by frequency.
            max_count = most_common[0][1]                                                   # Determine the highest frequency among the labels.
            candidates = [label for label, count in most_common if count == max_count]      # Select all labels with the highest frequency (in case of ties).
            predictions.append(random.choice(candidates))                                   # If there's a tie, randomly choose one of the candidate labels.
        return np.array(predictions)                                                        # Return the predictions as a NumPy array.