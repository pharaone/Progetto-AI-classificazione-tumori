import unittest
import pandas as pd

from Evaluation.Evaluator import Evaluator
from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm


class TestEvaluator(unittest.TestCase):

    def setUp(self):

        self.features = pd.DataFrame({                                          # Create a sample dataset with two features and binary targets
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [7, 8, 9, 10, 11, 12]
        })
        self.targets = pd.Series([2, 4, 2, 4, 2, 4])                            # Series of associated targets
        self.metrics = ["1", "2", "3", "4", "5", "6"]                           # Selection of all metrics
        self.distance_strategy = 1                                              # Distance strategy to use for knn that chose the distance strategy

        self.evaluator = Evaluator(self.features, self.targets, self.metrics)   # Create an Evaluator instance

    """
    Test the functionality of holdout validation, including:
    Splitting data into training and test sets.
    Ensuring the split respects the specified training percentage.
    Training a KNN model and verifying its predictions have the correct length.
    """
    def test_holdout_validation(self):
        training_percentage = 0.67                                                          # Percentage of the dataset used for training
        k_neighbors = 3                                                                     # Number of neighbors to consider

        x_train, y_train, x_test, y_test = self.evaluator.split_data(training_percentage)   # Split the data into training and test sets

        self.assertEqual(len(x_train), int(len(self.features) * training_percentage))       # Check that the number of elements in training and test sets is correct
        self.assertEqual(len(x_test), len(self.features) - len(x_train))
        self.assertEqual(len(y_train), len(x_train))
        self.assertEqual(len(y_test), len(x_test))

        self.assertGreater(len(x_train), 0)                                              # Ensure that training and test data are not empty
        self.assertGreater(len(x_test), 0)

        knn = KnnAlgorithm(k_neighbors, x_train, y_train, self.distance_strategy)           # Train and predict using KNN
        y_pred = knn.predict(x_test)
        self.assertEqual(len(y_pred), len(y_test))                                          # Check prediction length

    """
    Test the calculation of evaluation metrics, including:
    Verifying metrics for a perfectly correct and incorrect prediction.
    Ensuring all necessary metrics are calculated.
    """
    def test_calculate_metrics(self):
        y_test = pd.Series([2, 4, 2, 4])                                                    # Actual target values
        y_pred_correct = pd.Series([2, 4, 2, 4])                                            # Completely correct predictions
        y_pred_incorrect = pd.Series([4, 2, 4, 2])                                          # Completely incorrect predictions


        metrics_correct = self.evaluator.calculate_metrics(y_test, y_pred_correct)          # Calculate metrics for both correct and incorrect predictions
        metrics_incorrect = self.evaluator.calculate_metrics(y_test, y_pred_incorrect)

        self.assertAlmostEqual(metrics_correct['Accuracy'], 1.0)                     # Check accuracy metric for correct predictions (should be 100%)

        self.assertAlmostEqual(metrics_incorrect['Accuracy'], 0.0)                   # Check accuracy metric for incorrect predictions (should be 0%)

        self.assertIn('Error Rate', metrics_correct)                                 # Ensure that all required metrics are calculated correctly
        self.assertIn('Sensitivity', metrics_correct)
        self.assertIn('Specificity', metrics_correct)
        self.assertIn('Geometric Mean', metrics_correct)
        self.assertIn('Area Under The Curve Rate', metrics_correct)


if __name__ == '__main__':
    unittest.main()
