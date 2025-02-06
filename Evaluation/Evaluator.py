from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, dtype

"""
This class contains the definition of the abstract method evaluate and many useful methods for evaluation.
Every Evaluator (Holdout, K-fold, Stratified) should extend it and override the evaluate method.
"""
class Evaluator(ABC):

    """
    This is the method every evaluator extends and which starts the evaluation process.
    It does not take any input, all inputs are given to the constructors.
    """
    @abstractmethod
    def evaluate(self):
        pass

    """
       This method calculates the confusion matrix from the data of the predictions and the real target data.
    """

    def _calculate_confusion_matrix(self, y_test: pd.Series, y_pred: pd.Series) -> ndarray[tuple[int, int], dtype[any]]:
        classes = [4, 2]  # define class labels (4 for malign and 2 for benign)

        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)  # initialize confusion matrix

        for true, pred in zip(y_test,
                              y_pred):  # iterate for each target value (a row contains the real value and the prediction)
            true_index = classes.index(true)
            pred_index = classes.index(pred)
            confusion_matrix[
                true_index, pred_index] += 1  # depending on the value of the prediction and the real data add 1 to the correct cell of the matrix

        return confusion_matrix  # return the matrix

    """
        This method takes a confusion matrix as input, plots it in a new figure and saves it as a png file.
    """

    def _plot_save_confusion_matrix(self, confusion_matrix: ndarray[tuple[int, int], dtype[any]], output_path: str):
        classes = [4, 2]  # Define class labels (4 for malign and 2 for benign)
        class_labels = ['Malign (4)', 'Benign (2)']

        try:
            fig, ax = plt.subplots(figsize=(8, 6))  # creates a new figure and an axis
            im = ax.imshow(confusion_matrix, interpolation='nearest',
                           cmap='Blues')  # display the confusion matrix as a heatmap on the axis
            plt.colorbar(im, ax=ax)  # adds a colorbar to the plot, linked to the heatmap image

            for i in range(len(classes)):  # iterate over classes to write in each cell of the matrix
                for j in range(len(classes)):
                    ax.text(j, i, confusion_matrix[i, j], ha="center", va="center",
                            color="black")  # writes the value to the correct cell of the axys

            # Set axis labels and ticks
            ax.set(xticks=np.arange(len(classes)),
                   yticks=np.arange(len(classes)),
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   xlabel='Predicted Label',  # set x axis label
                   ylabel='True Label',  # set y axis label
                   title='Confusion Matrix')  # set matrix tile

            plt.xticks(rotation=45)  # rotate x-axis labels for better readability

            plt.tight_layout()  # this auto adjusts spacing
            plt.savefig(output_path)  # save the file
            plt.close()
            print(f"Confusion matrix successfully saved to {output_path}")
        except Exception as e:
            print(f"Error while plotting confusion matrix: {e}")

    """
        This method saves the evaluation metrics to a CSV file. 
        The metrics are provided as a dictionary, where keys represent metric names 
        and values represent their corresponding calculated values as described in the calculate_metrics method.
    """

    def _save_metrics(self, metrics: dict):
        import csv
        with open('output/result.csv', 'w') as fp:  # open the file in write mode
            csv.writer(fp).writerows(metrics.items())  # write the dictionary items as rows in the CSV file

    """
        This method saves averaged evaluation metrics to a CSV file. 
        It takes as an input is a list of metric dictionaries (one for each fold or evaluation run) and
        then it calculates the mean of each metric across all runs and writes the resulting averages to a CSV file.
    """

    def _save_metrics_from_metrics_list(self, metrics_list: list[dict]):
        metric_sum = {}  # dictionary to store the cumulative sum of each metric
        metric_count = {}  # dictionary to store the count of values for each metric

        for metrics in metrics_list:  # iterate over the list of metrics
            for key, value in metrics.items():  # iterate over each metric dictionary
                metric_sum[key] = metric_sum.get(key, 0) + value  # update the cumulative sum for each metric
                metric_count[key] = metric_count.get(key, 0) + 1  # update the count for each metric

        metrics_mean_list = {key: metric_sum[key] / metric_count[key] for key in
                             metric_count}  # calculate the mean for each metric
        import csv
        with open('output/result.csv', 'w') as fp:  # open the file in write mode
            csv.writer(fp).writerows(metrics_mean_list.items())  # write the dictionary items as rows in the CSV file

