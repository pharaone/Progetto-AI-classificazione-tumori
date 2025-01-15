from math import sqrt

import pandas as pd
import numpy as np

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm


class Evaluator:
    __features : pd.DataFrame = None
    __targets : pd.Series = None

    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        self.__features = features
        self.__targets = targets

    """
    Performs a model validation using the holdout data split method. 
    Divides the data into a training set and a test set, trains a KNN model, 
    and calculates the evaluation metrics on the test set.

    Args:
    training_percentage (float): The percentage of data to use for training (0-1).
    k_neighbors (int): The number of neighbors to consider in the KNN model.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    def holdout_validation(self, training_percentage: float, k_neighbors: int):
        x_train, y_train, x_test, y_test = self._split_data(training_percentage)    # Split the data into training and test sets

        knn = KnnAlgorithm(k_neighbors, x_train, y_train)                           # Initialize the KNN classifier

        y_pred = knn.predict(x_test)                                                # Make predictions on the test data

        return self.calculate_metrics(y_test, y_pred)                               # Calculate and return the evaluation metrics

    def k_fold_cross_validation(self, k_times: int):
        features_subsets : [pd.DataFrame] = np.array_split(self.__features, k_times)
        targets_subsets : [pd.Series] = np.array_split(self.__targets, k_times)

        accuracy_rate_list = []
        error_rate_list = []
        sensitivity_rate_list = []
        specificity_rate_list = []
        geometric_mean_rate_list = []
        area_under_the_curve_rate_list = []

    def stratified_cross_validation(self, k_times: int):
        pass

    """
    Calculates the main evaluation metrics for the KNN model: 
    Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean, 
    and Area Under the Curve (AUC).

    Args:
    y_test (pd.Series): The true target values in the test set.
    y_pred (pd.Series): The predicted values from the KNN model.

    Returns:
    dict: A dictionary containing the calculated evaluation metrics.
    """
    def calculate_metrics(self, y_test, y_pred):
        true_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 4)                               # Initialize variables for metric calculations
        true_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 2)
        false_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 4)
        false_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 2)

        accuracy = (true_positive + true_negative) / len(y_test)                                                        # Calculate accuracy
        error_rate = 1 - accuracy                                                                                       # Calculate error rate
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0   # Calculate sensitivity (true positive rate)
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0   # Calculate specificity (true negative rate)
        geometric_mean = sqrt(sensitivity * specificity)                                                                # Calculate the geometric mean of sensitivity and specificity
        auc = (sensitivity + specificity) / 2                                                                           # Calculate the Area Under the Curve (AUC)

        metrics = {                                                                                                     # Create a dictionary with the calculated metrics
            "Accuracy Rate": accuracy,
            "Error Rate": error_rate,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Geometric Mean": geometric_mean,
            "Area Under the Curve": auc,
        }

        return metrics

    """
    Splits the data into training and test sets based on the training percentage.

    Args:
    training_percentage (float): The percentage of data to use for training (0-1).

    Returns:
    tuple: The training (x_train, y_train) and test (x_test, y_test) data sets.
    """
    def _split_data(self, training_percentage: float):
        split_index = int(len(self.__features) * training_percentage)   # Calculate the index for splitting based on the training percentage

        x_train = self.__features.iloc[:split_index]                    # Extract the training and test data
        y_train = self.__targets.iloc[:split_index]
        x_test = self.__features.iloc[split_index:]
        y_test = self.__targets.iloc[split_index:]

        return x_train, y_train, x_test, y_test
