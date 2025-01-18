import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

from KNNAlgorithm.CalculateDistance.Factory.EuclideanDistanceCreator import EuclideanDistanceCreator


class KnnAlgorithm:
    """
    Initializes the KNN classifier with the number of neighbors (k) and the training data.
    The training data will later be used to find the nearest neighbors for test points.
    """

    def __init__(self, k: int, x_train: pd.DataFrame, y_train: pd.Series):
        self.x_train = x_train      # Store the training data (features).
        self.y_train = y_train      # Store the labels for the training data.
        self.k = k                  # Number of neighbors to consider.

    """
    Identifies the k nearest neighbors for a given test point.
    The method calculates the distance between the test point and all training points
    and selects the k points with the smallest distances. This provides the labels
    of the closest training points, which will be used to determine the label of the test point.
    """

    def get_neighbors(self, x: pd.Series):
        try:
            distance_creator = EuclideanDistanceCreator()               # Create an instance of the factory for calculating Euclidean distance.
            distance_strategy = distance_creator.create_distance()      # Create the distance calculation strategy (EuclideanDistance).
            distances = []

            for _, x_train_sample in self.x_train.iterrows():           # Iterating through each row in the training data (x_train is a DataFrame)
                distance = distance_strategy.calculate(x,
                                                       x_train_sample)  # Calculate distance between the test point `x` and the current training sample `x_train_sample`
                distances.append(distance)

            k_indices = np.argsort(distances)[:self.k]                  # Sort distances and get the indices of the k smallest
        except Exception as e:                                          # handles  exception
            print(e)                                                    # prints the exception
            sys.exit('Error to get the neighbors.')
        return self.y_train.iloc[k_indices]                         # Return the labels of the k closest points

    """
    Predicts the labels for a set of test points.
    For each test point, the method identifies its k closest neighbors and assigns a label
    based on the majority label among these neighbors. If there is a tie in the label frequencies,
    one of the tied labels is chosen randomly.
    """

    def predict(self, x_test: pd.DataFrame):
        try:
            predictions = []                                # Initialize an empty list to store predictions

            for _, x in x_test.iterrows():                  # Iterating through each test point in the dataset
                neighbors = self.get_neighbors(x)           # Get the labels of the nearest neighbors

                label_counts = Counter(neighbors)           # Count the frequency of each label among the neighbors
                most_common = label_counts.most_common()    # Get the most common labels sorted by frequency
                max_count = most_common[0][1]               # Determine the highest frequency among the labels


                candidates = [label for label, count in most_common if count == max_count]  # Select all labels with the highest frequency (in case of ties)
                predictions.append(random.choice(candidates))                               # If there's a tie, randomly choose one of the candidate labels
        except Exception as e:                               # handles  exception
            print(e)                                         # prints the exception
            sys.exit('Error to predict the label.')

        return np.array(predictions)
