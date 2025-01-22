import sys
from math import sqrt

import pandas as pd
import numpy as np

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

"""
This class contains all the code needed to evaluate and export the evaluation data of the model
using different methods of validation:
Holdout, K-fold cross validation and Stratified cross validation
"""
class Evaluator:
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
    def holdout_validation(self, training_percentage: float, k_neighbors: int):
        try:
            x_train, y_train, x_test, y_test = self.split_data(
                training_percentage)                                    # Split the data into training and test sets

            knn = KnnAlgorithm(k_neighbors, x_train, y_train)           # Initialize the KNN classifier
            y_pred = knn.predict(x_test)                                # Make predictions on the test data
        except Exception as e:                                          # handles every other exception
            print(e)                                                    # prints the exception
            sys.exit('Error to use holdout_validation.')

        self.save_metrics(self.calculate_metrics(y_test, y_pred))       # Calculate and return the evaluation metrics

    """
    This method divides the data into k (user inserted parameter) subsets for k-fold cross-validation.
    It trains the KNN model on k-1 subsets and evaluates on the remaining one for k times
    calculating metrics each time and averaging them across all folds.
    """
    def k_fold_cross_validation(self, k_times: int, k_neighbors: int):
        # divide the features and targets into k subsets
        features_subsets : [pd.DataFrame] = np.array_split(self.__features, k_times)
        targets_subsets : [pd.Series] = np.array_split(self.__targets, k_times)

        metrics = []                                                            # created an empty metrics array

        for index in range(k_times):
            # create a feature and a target set without the current index fold
            features_set = features_subsets.copy().pop(index)
            targets_set = targets_subsets.copy().pop(index)

            knn = KnnAlgorithm(k_neighbors, features_set, targets_set)          # creates an instance of the knn model
            y_prediction = knn.predict(features_subsets[index])                 # runs the prediction

            metrics.append(self.calculate_metrics(targets_set, y_prediction))   # calculates the requested metrics for this evaluation and appends them to the list

        self.save_metrics_from_metrics_list(metrics)                            # saves to file the requested metrics

    """
    This method performs the stratified k-fold (k user defined parameter) cross-validation.
    It trains the KNN model on k-1 folds each with the same class distribution as the original set 
    and evaluates on the remaining one calculating metrics each time and averaging them across all folds.
    """
    def stratified_cross_validation(self, k_times: int, k_neighbors: int):
        unified_features_and_targets = self.__features.copy()                           # creates a copy of the features
        unified_features_and_targets['targets'] = self.__targets.copy()                 # adds the target column to the copy
        unified_features_and_targets.sort_values(by=['targets'], inplace=True)          # sorts everything by class

        benign_dataset = unified_features_and_targets[unified_features_and_targets['targets'] == 2.0]       # extracts the benign dataset
        malign_dataset = unified_features_and_targets[unified_features_and_targets['targets'] == 4.0]       # extracts the malign dataset

        benign_targets = np.array_split(benign_dataset['targets'], k_times)             # splits the benign targets in k folds
        malign_targets = np.array_split(malign_dataset['targets'], k_times)             # splits the malign targets in k folds

        benign_dataset = benign_dataset.drop(columns=['targets'])                       # drops the targets column only used to split the benign and malingn
                                                                                        # sets more easily from the dataset
        malign_dataset = malign_dataset.drop(columns=['targets'])

        benign_subsets: [pd.DataFrame] = np.array_split(benign_dataset, k_times)        # creates the subsets of the dataset
        malign_subsets: [pd.DataFrame] = np.array_split(malign_dataset, k_times)        # creates the subsets of the dataset

        metrics = []                                                                    # created an empty metrics array

        for index in range(k_times):
            features_set = pd.concat([benign_subsets.copy().pop(index), malign_subsets.copy().pop(index)])          # unites benign and maling sets and removes the current index fold
            targets_set = pd.concat([benign_targets.copy().pop(index), malign_targets.copy().pop(index)])           # unites benign and maling sets and removes the current index fold

            knn = KnnAlgorithm(k_neighbors, features_set, targets_set)                                  # creates an instance of the knn model
            y_prediction = knn.predict(pd.concat([benign_subsets[index], malign_subsets[index]]))       # runs the prediction

            metrics.append(self.calculate_metrics(targets_set, y_prediction))           # calculates the requested metrics for this evaluation and appends them to the list


        self.save_metrics_from_metrics_list(metrics)                                    # saves to file the requested metrics

    """
    Calculates the main evaluation metrics for the KNN model: 
    Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean, 
    and Area Under the Curve (AUC), it's possible choose the matrics or all the matrics.
    """
    def calculate_metrics(self, y_test: pd.Series, y_pred: pd.Series):
        try:
            true_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 4)   # Calculate the confusion matrix
            true_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 2)
            false_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 4)
            false_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 2)

            metrics = {}

            total = true_positive + true_negative + false_positive + false_negative

            if self.__metrics.__contains__("1") or self.__metrics.__contains__("7"):
                accuracy = (true_positive + true_negative) / total if total > 0 else 0
                metrics['Accuracy'] = accuracy

            if self.__metrics.__contains__("2") or self.__metrics.__contains__("7"):
                error_rate = (false_positive + false_negative) / total if total > 0 else 0
                metrics['Error Rate'] = error_rate
            if self.__metrics.__contains__("3") or self.__metrics.__contains__("7"):
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                metrics['Sensitivity'] = sensitivity
            if self.__metrics.__contains__("4") or self.__metrics.__contains__("7"):
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                metrics['Specificity'] = specificity
            if self.__metrics.__contains__("5") or self.__metrics.__contains__("7"):
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                geometric_mean = sqrt(sensitivity * specificity)
                metrics['Geometric Mean'] = geometric_mean
            if self.__metrics.__contains__("6") or self.__metrics.__contains__("7"):
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                auc = (sensitivity + specificity) / 2
                metrics['Area Under The Curve Rate'] = auc

            return metrics

        except Exception as e:                                                                   # handles exception
            print(e)                                                                             # prints the exception
            sys.exit('Error to split data.')

    """
    This method saves the evaluation metrics to a CSV file. 
    The metrics are provided as a dictionary, where keys represent metric names 
    and values represent their corresponding calculated values as described in the calculate_metrics method.
    """
    def save_metrics(self, metrics: dict):
        import csv
        with open('output/result.csv', 'w') as fp:                                  # open the file in write mode
            csv.writer(fp).writerows(metrics.items())                               # write the dictionary items as rows in the CSV file

    """
    This method saves averaged evaluation metrics to a CSV file. 
    It takes as an input is a list of metric dictionaries (one for each fold or evaluation run) and
    then it calculates the mean of each metric across all runs and writes the resulting averages to a CSV file.
    """
    def save_metrics_from_metrics_list(self, metrics_list: list[dict]):
        metric_sum = {}                                                             # dictionary to store the cumulative sum of each metric
        metric_count = {}                                                           # dictionary to store the count of values for each metric

        for metrics in metrics_list:                                                # iterate over the list of metrics
            for key, value in metrics.items():                                      # iterate over each metric dictionary
                metric_sum[key] = metric_sum.get(key, 0) + value                    # update the cumulative sum for each metric
                metric_count[key] = metric_count.get(key, 0) + 1                    # update the count for each metric

        metrics_mean_list = {key: metric_sum[key] / metric_count[key] for key in metric_count}              # calculate the mean for each metric
        import csv
        with open('output/result.csv', 'w') as fp:                                  # open the file in write mode
            csv.writer(fp).writerows(metrics_mean_list.items())                     # write the dictionary items as rows in the CSV file

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
