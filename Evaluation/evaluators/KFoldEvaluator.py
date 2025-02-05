import sys
from math import sqrt

import pandas as pd
import numpy as np
from Evaluation.Evaluator import Evaluator

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

"""
This class contains all the code needed to evaluate and export the evaluation data of the model
using different methods of validation:
K-fold cross validation
"""
class KFoldEvaluator(Evaluator):
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
                 k_neighbors: int, distance_strategy :int, k_times: int):
        self.__features = features
        self.__targets = targets
        self.__metrics = metrics
        self.__k_neighbors = k_neighbors
        self.__distance_strategy = distance_strategy
        self.__k_times = k_times
        print("uykfkuyfkuy,giylfv")

    """
    This method divides the data into k (user inserted parameter) subsets for k-fold cross-validation.
    It trains the KNN model on k-1 subsets and evaluates on the remaining one for k times
    calculating metrics each time and averaging them across all folds.
    """
    def evaluate(self):
        print("K-fold cross validation")
        # divide the features and targets into k subsets
        features_subsets : [pd.DataFrame] = np.array_split(self.__features, self.__k_times)
        targets_subsets : [pd.Series] = np.array_split(self.__targets, self.__k_times)

        metrics = []                                                            # created an empty metrics array

        classes = [4, 2]
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)       # create the confusion matrix

        for index in range(self.__k_times):
            # create a feature and a target set without the current index fold
            features_set = features_subsets.copy().pop(index)
            targets_set = targets_subsets.copy().pop(index)

            knn = KnnAlgorithm(self.__k_neighbors, features_set, targets_set, self.__distance_strategy)          # creates an instance of the knn model
            y_prediction = knn.predict(features_subsets[index])                 # runs the prediction

            metrics.append(self.calculate_metrics(targets_set, y_prediction))   # calculates the requested metrics for this evaluation and appends them to the list
            confusion_matrix += self._calculate_confusion_matrix(targets_set, y_prediction)      # add the data from the current run to the confusion matrix

        self._plot_save_confusion_matrix(confusion_matrix, "output/mean_confusion_matrix.jpg")      # plot and save the confusion matrix
        self._save_metrics_from_metrics_list(metrics)                            # saves to file the requested metrics

    """
    Calculates the main evaluation metrics for the KNN model: 
    Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean,and Area Under the Curve (AUC).
    The calculation is based on the specific metrics selected by the user, 
    identified through numerical keys provided in the class.
    Depending on the selected numbers, the corresponding metrics are computed.
    """
    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series):
        try:
            true_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 4)                                   # Calculate the confusion matrix
            true_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 2)
            false_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 4)
            false_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 2)

            metrics = {}                                                                                                        # Dictionary to store computed metrics

            total = true_positive + true_negative + false_positive + false_negative

            if self.__metrics.__contains__("1") or self.__metrics.__contains__("7"):                                            # Calculate accuracy if selected
                accuracy = (true_positive + true_negative) / total if total > 0 else 0
                metrics['Accuracy'] = accuracy

            if self.__metrics.__contains__("2") or self.__metrics.__contains__("7"):                                            # Calculate error rate if selected
                error_rate = (false_positive + false_negative) / total if total > 0 else 0
                metrics['Error Rate'] = error_rate
            if self.__metrics.__contains__("3") or self.__metrics.__contains__("7"):                                            # Calculate sensitivity if selected
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                metrics['Sensitivity'] = sensitivity
            if self.__metrics.__contains__("4") or self.__metrics.__contains__("7"):                                            # Calculate specificity if selected
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                metrics['Specificity'] = specificity
            if self.__metrics.__contains__("5") or self.__metrics.__contains__("7"):                                            # Calculate geometric mean if selected
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                geometric_mean = sqrt(sensitivity * specificity)
                metrics['Geometric Mean'] = geometric_mean
            if self.__metrics.__contains__("6") or self.__metrics.__contains__("7"):                                            # Calculate AUC if selected
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                auc = (sensitivity + specificity) / 2
                metrics['Area Under The Curve Rate'] = auc

            return metrics

        except Exception as e:                                                                   # handles exception
            print(e)                                                                             # prints the exception
            sys.exit('Error to calculate metrics.')


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
