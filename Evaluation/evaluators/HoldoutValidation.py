import sys
from math import sqrt

import pandas as pd
import Evaluation.EvaluatorUtilities as utilities

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

"""
This class contains all the code needed to evaluate and export the evaluation data of the model
using different methods of validation:
Holdout validation
"""
class HoldoutValidation:
    __features : pd.DataFrame = None
    __targets : pd.Series = None
    """
    __metrics_map = {
        "1": "Accuracy Rate",
        "2": "Error Rate",
        "3": "Sensitivity",
        "4": "Specificity",
        "5": "Geometric Mean",
        "6": "Area Under the Curve"
        "7": "All the above"
    }
    """

    def __init__(self, features: pd.DataFrame, targets: pd.Series, metrics: list[str]):
        self.__features = features
        self.__targets = targets
        self.__metrics = metrics

    """
    Divides the dataset into training and test sets using the given percentage. 
    Trains the KNN model on the training set and evaluates its predictions on the test set.
    At the end calculate metrics that are crucial to understanding the model's overall performance
    """
    def holdout_validation(self, training_percentage: float, k_neighbors: int, distance_strategy: int):
        try:
            x_train, y_train, x_test, y_test = self.split_data(
                training_percentage)                                    # Split the data into training and test sets

            knn = KnnAlgorithm(k_neighbors, x_train, y_train, distance_strategy)           # Initialize the KNN classifier
            y_pred = knn.predict(x_test)                                # Make predictions on the test data
        except Exception as e:                                          # handles every other exception
            print(e)                                                    # prints the exception
            sys.exit('Error to use holdout validation.')

        utilities.save_metrics(self.calculate_metrics(y_test, y_pred))       # calculate and return the evaluation metrics
        utilities.plot_save_confusion_matrix(utilities.calculate_confusion_matrix(y_test, y_pred), "output/plot.jpg")     # plot and save the confusion matrix


    """
    Splits the data into training and test sets based on the training percentage.
    """
    def split_data(self, training_percentage: float):
        try:
            split_index = int(len(self.__features) * training_percentage)   # Calculate the index for splitting based on the training percentage

            x_train = self.__features.iloc[:split_index]                    # Extract the training and test data
            y_train = self.__targets.iloc[:split_index]
            x_test = self.__features.iloc[split_index:]
            y_test = self.__targets.iloc[split_index:]
        except Exception as e:                              # handles exception
            print(e)                                        # prints the exception
            sys.exit('Error to split data.')

        return x_train, y_train, x_test, y_test

    """
        Calculates the main evaluation metrics for the KNN model: 
        Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean,and Area Under the Curve (AUC).
        The calculation is based on the specific metrics selected by the user, 
        identified through numerical keys provided in the class.
        Depending on the selected numbers, the corresponding metrics are computed.
        """
    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series):
        try:
            true_positive = sum(
                1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 4)  # Calculate the confusion matrix
            true_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 2)
            false_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 4)
            false_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 2)

            metrics = {}  # Dictionary to store computed metrics

            total = true_positive + true_negative + false_positive + false_negative

            if self.__metrics.__contains__("1") or self.__metrics.__contains__("7"):  # Calculate accuracy if selected
                accuracy = (true_positive + true_negative) / total if total > 0 else 0
                metrics['Accuracy'] = accuracy

            if self.__metrics.__contains__("2") or self.__metrics.__contains__("7"):  # Calculate error rate if selected
                error_rate = (false_positive + false_negative) / total if total > 0 else 0
                metrics['Error Rate'] = error_rate
            if self.__metrics.__contains__("3") or self.__metrics.__contains__(
                    "7"):  # Calculate sensitivity if selected
                sensitivity = true_positive / (true_positive + false_negative) if (
                                                                                              true_positive + false_negative) > 0 else 0
                metrics['Sensitivity'] = sensitivity
            if self.__metrics.__contains__("4") or self.__metrics.__contains__(
                    "7"):  # Calculate specificity if selected
                specificity = true_negative / (true_negative + false_positive) if (
                                                                                              true_negative + false_positive) > 0 else 0
                metrics['Specificity'] = specificity
            if self.__metrics.__contains__("5") or self.__metrics.__contains__(
                    "7"):  # Calculate geometric mean if selected
                sensitivity = true_positive / (true_positive + false_negative) if (
                                                                                              true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (
                                                                                              true_negative + false_positive) > 0 else 0
                geometric_mean = sqrt(sensitivity * specificity)
                metrics['Geometric Mean'] = geometric_mean
            if self.__metrics.__contains__("6") or self.__metrics.__contains__("7"):  # Calculate AUC if selected
                sensitivity = true_positive / (true_positive + false_negative) if (
                                                                                              true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (
                                                                                              true_negative + false_positive) > 0 else 0
                auc = (sensitivity + specificity) / 2
                metrics['Area Under The Curve Rate'] = auc

            return metrics

        except Exception as e:  # handles exception
            print(e)  # prints the exception
            sys.exit('Error to calculate metrics.')