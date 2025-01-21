import sys
from math import sqrt

import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm


class Evaluator:
    __features : pd.DataFrame = None
    __targets : pd.Series = None
    __metrics: list = None

    def __init__(self, features: pd.DataFrame, targets: pd.Series, metrics: list = None):
        self.__features = features
        self.__targets = targets
        self.__metrics = metrics if metrics is not None else [
            "Accuracy Rate", "Error Rate", "Sensitivity", "Specificity",
            "Geometric Mean", "Area Under the Curve"
        ]

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

        return self.calculate_metrics(y_test, y_pred, self.__metrics)   # Calculate and return the evaluation metrics

    def k_fold_cross_validation(self, k_times: int, k_neighbors: int):
        features_subsets : [pd.DataFrame] = np.array_split(self.__features, k_times)
        targets_subsets : [pd.Series] = np.array_split(self.__targets, k_times)

        accuracy_rate_list = []
        error_rate_list = []
        sensitivity_rate_list = []
        specificity_rate_list = []
        geometric_mean_rate_list = []
        area_under_the_curve_rate_list = []

        for index in range(k_times):
            features_set = features_subsets.copy().pop(index)
            targets_set = targets_subsets.copy().pop(index)

            knn = KnnAlgorithm(k_neighbors, features_set, targets_set)
            y_prediction = knn.predict(features_subsets[index])

            metrics = self.calculate_metrics(targets_set, y_prediction)

            accuracy_rate_list.append(metrics['Accuracy Rate'])
            error_rate_list.append(metrics['Error Rate'])
            sensitivity_rate_list.append(metrics['Sensitivity'])
            specificity_rate_list.append(metrics['Specificity'])
            geometric_mean_rate_list.append(metrics['Geometric Mean'])
            area_under_the_curve_rate_list.append(metrics['Area Under the Curve'])

        metrics_dict = {'Accuracy Rate List': accuracy_rate_list,
                        'Error Rate List': error_rate_list,
                        'Sensitivity Rate List': sensitivity_rate_list,
                        'Specificity Rate List': specificity_rate_list,
                        'Geometric Mean Rate List': geometric_mean_rate_list,
                        'Area Under The Curve Rate List': area_under_the_curve_rate_list
                        }

        self.save_metrics(metrics_dict)

    def stratified_cross_validation(self, k_times: int, k_neighbors: int):
        unified_features_and_targets = self.__features.copy()
        unified_features_and_targets['targets'] = self.__targets.copy()
        unified_features_and_targets.sort_values(by=['targets'], inplace=True)

        benign_dataset = unified_features_and_targets[unified_features_and_targets['targets'] == 2.0]
        malign_dataset = unified_features_and_targets[unified_features_and_targets['targets'] == 4.0]

        benign_targets = np.array_split(benign_dataset['targets'], k_times)
        malign_targets = np.array_split(malign_dataset['targets'], k_times)

        benign_dataset = benign_dataset.drop(columns=['targets'], inplace=True)
        malign_dataset = malign_dataset.drop(columns=['targets'], inplace=True)

        benign_subsets: [pd.DataFrame] = np.array_split(benign_dataset, k_times)
        malign_subsets: [pd.DataFrame] = np.array_split(malign_dataset, k_times)

        accuracy_rate_list = []
        error_rate_list = []
        sensitivity_rate_list = []
        specificity_rate_list = []
        geometric_mean_rate_list = []
        area_under_the_curve_rate_list = []

        for index in range(k_times):
            features_set = benign_subsets.copy().pop(index) + malign_subsets.copy().pop(index)
            targets_set = benign_targets.copy().pop(index) + malign_targets.copy().pop(index)

            knn = KnnAlgorithm(k_neighbors, features_set, targets_set)
            y_prediction = knn.predict(benign_subsets[index] + malign_subsets[index])

            metrics = self.calculate_metrics(targets_set, y_prediction)

            accuracy_rate_list.append(metrics['Accuracy Rate'])
            error_rate_list.append(metrics['Error Rate'])
            sensitivity_rate_list.append(metrics['Sensitivity'])
            specificity_rate_list.append(metrics['Specificity'])
            geometric_mean_rate_list.append(metrics['Geometric Mean'])
            area_under_the_curve_rate_list.append(metrics['Area Under The Curve'])

        metrics_dict = {'Accuracy Rate List': accuracy_rate_list,
                        'Error Rate List': error_rate_list,
                        'Sensitivity Rate List': sensitivity_rate_list,
                        'Specificity Rate List': specificity_rate_list,
                        'Geometric Mean Rate List': geometric_mean_rate_list,
                        'Area Under The Curve Rate List': area_under_the_curve_rate_list
                        }

        self.save_metrics(metrics_dict)

    """
    Calculates the main evaluation metrics for the KNN model: 
    Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean, 
    and Area Under the Curve (AUC), it's possible choose the matrics or all the matrics.
    """
    def calculate_metrics(self, y_test, y_pred, selected_metrics=None):
        try:
            true_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 4)   # Calculate the confusion matrix
            true_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 2)
            false_positive = sum(1 for y, pred in zip(y_test, y_pred) if y == 2 and pred == 4)
            false_negative = sum(1 for y, pred in zip(y_test, y_pred) if y == 4 and pred == 2)

            total = true_positive + true_negative + false_positive + false_negative
            accuracy = (true_positive + true_negative) / total if total > 0 else 0
            error_rate = (false_positive + false_negative) / total if total > 0 else 0
            sensitivity = true_positive / (true_positive + false_negative) \
                if (true_positive + false_negative) > 0 else 0
            specificity = true_negative / (true_negative + false_positive) \
                if (true_negative + false_positive) > 0 else 0
            geometric_mean = sqrt(sensitivity * specificity)
            auc = (sensitivity + specificity) / 2

            all_metrics = {                                                                      # Store all computed metrics in a dictionary
                "Accuracy Rate": accuracy,
                "Error Rate": error_rate,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Geometric Mean": geometric_mean,
                "Area Under the Curve": auc,
            }

            if selected_metrics:                                                                 # If specific metrics are requested, filter and return them
                result_metrics = {}                                                              # Initialize empty dictionary
                for metric in selected_metrics:
                    if metric in all_metrics:
                        result_metrics[metric] = all_metrics[metric]                             # Add selected metrics
                return result_metrics                                                            # Return selected metrics
            return all_metrics                                                                   # Return all metrics if none specified
        except Exception as e:                                                                   # handles exception
            print(e)                                                                             # prints the exception
            sys.exit('Error to split data.')

    def save_metrics(self, metrics: dict):
        import json
        with open('result.json', 'w') as fp:
            json.dump(metrics, fp)

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
