import sys
from math import sqrt

import numpy as np
import pandas as pd
from Evaluation.Evaluator import Evaluator

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

"""
This class extends the class Evaluator and contains all the code needed to evaluate and export the evaluation data of the model
using Holdout validation
"""
class HoldoutEvaluator(Evaluator):
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

    def __init__(self, features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                 k_neighbors: int, distance_strategy :int, training_percentage: float):
        self.__features = features
        self.__targets = targets
        self.__metrics = metrics
        self.__k_neighbors = k_neighbors
        self.__distance_strategy = distance_strategy
        self.__training_percentage = training_percentage

    """
    Divides the dataset into training and test sets using the given percentage. 
    Trains the KNN model on the training set and evaluates its predictions on the test set.
    At the end calculate metrics that are crucial to understanding the model's overall performance
    """
    def evaluate(self):
        try:
            x_train, y_train, x_test, y_test = self.split_data(
                self.__training_percentage)                                    # Split the data into training and test sets

            knn = KnnAlgorithm(self.__k_neighbors, x_train, y_train, self.__distance_strategy)           # Initialize the KNN classifier
            y_pred = knn.predict(x_test)                                # Make predictions on the test data
        except Exception as e:                                          # handles every other exception
            print(e)                                                    # prints the exception
            sys.exit('Error to use holdout validation.')

        self._save_metrics(self.calculate_metrics(y_test, y_pred))       # calculate and return the evaluation metrics
        self._plot_save_confusion_matrix(self._calculate_confusion_matrix(y_test, y_pred), "output/mean_confusion_matrix.jpg")     # plot and save the confusion matrix


    """
    Splits the data into training and test sets based on the training percentage.
    """
    def split_data(self, training_percentage: float):
        try:
            if not (0 < training_percentage < 1):                           # Check if training_percentage is between 0 e 1
                raise ValueError("Training percentage must be between 0 and 1 (exclusive).")
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
            valid_metrics = {"1", "2", "3", "4", "5", "6", "7"}
            if not set(self.__metrics).issubset(valid_metrics):                                                                 # Check if the metrics selected is correct
                raise ValueError("Invalid metric. Allowed values are: 1, 2, 3, 4, 5, 6, 7.")
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
            if self.__metrics.__contains__("6") or self.__metrics.__contains__("7"):
                fpr, tpr = [], []  # Initialize empty lists for False Positive Rate (FPR) and True Positive Rate (TPR)
                thresholds = np.linspace(2, 4, 30)  # Beetween [2, 4]

                for m in thresholds:
                    tp = sum(1 for y, pred in zip(y_test, y_pred) if
                             y == 4 and pred >= m)  # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) based on the threshold
                    tn = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred < m)
                    fp = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred >= m)
                    fn = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred < m)
                    tpr.append(tp / (tp + fn) if (
                                                             tp + fn) > 0 else 0)  # Calculate TPR and FPR for the current threshold and append to the respective lists
                    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

                sorted_indices = np.argsort(fpr)  # Sort FPR and TPR based on ascending FPR values
                fpr = np.array(fpr)[sorted_indices]
                tpr = np.array(tpr)[sorted_indices]

                auc_value = np.trapz(tpr, fpr)  # Calculate Area Under the Curve (AUC) using the trapezoidal rule

                metrics['Area Under The Curve Rate'] = auc_value  # Store the AUC value in the metrics dictionary
            return metrics

        except Exception as e:  # handles exception
            print(e)  # prints the exception
            sys.exit('Error to calculate metrics.')