import unittest

import numpy as np
import pandas as pd
import pytest

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

"""
This test class is a unit test  to test the functionality 
and correctness of the KnnAlgorithm class. It uses the unittest module to define 
various test cases to ensure the KNN algorithm operates as expected.
"""
class TestKnnAlgorithm(unittest.TestCase):

    def setUp(self):
        self.x_train = pd.DataFrame({                                               # Creating a training dataset with two features ('feature1' and 'feature2')
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8]
        })
        self.y_train = pd.Series(['A', 'B', 'A', 'B'])                              # Creating labels corresponding to the training data
        self.knn = KnnAlgorithm(k=2, x_train=self.x_train, y_train=self.y_train)    # Initializing an instance of the KnnAlgorithm class with k=2

    """
    This method verifies that the KnnAlgorithm object is correctly 
    initialized with the right attributes. It checks that the 'k' parameter is set correctly 
    and that the training data (x_train) and labels (y_train) are of the correct types 
    (pandas DataFrame and Series, respectively).
    """
    def test_initialization(self):
        self.assertEqual(self.knn.k, 2)                                      # Verify that the value of k is set correctly to 2
        self.assertTrue(isinstance(self.knn.x_train, pd.DataFrame))                 # Check that the x_train attribute is a pandas DataFrame
        self.assertTrue(isinstance(self.knn.y_train, pd.Series))                    # Check that the y_train attribute is a pandas Series

    """
    This method tests the get_neighbors function, which is responsible 
    for finding the k nearest neighbors to a given test point. It asserts that the method 
    returns the correct number of neighbors and that those neighbors' labels are contained 
    in the training labels.
    """
    def test_get_neighbors(self):
        test_point = pd.Series({'feature1': 2.5, 'feature2': 6.5})                  # Define a test point with two features ('feature1' and 'feature2')
        neighbors = self.knn.get_neighbors(test_point)                              # Call the get_neighbors function to obtain the k nearest neighbors to the test point

        self.assertEqual(len(neighbors), 2)                                  # Verify that the number of neighbors returned is equal to 2 (value of k)

        self.assertIn(neighbors.iloc[0], self.y_train.values)                        # Check that the first neighbor belongs to the original y_train labels
        self.assertIn(neighbors.iloc[1], self.y_train.values)                        # Check that the second neighbor belongs to the original y_train labels

    """
    This method checks that the predict function works as expected. It ensures 
    that the model returns predictions for a set of test points, verifies that the output 
    is in the correct format (numpy array), and confirms that the predicted labels match
    in the training labels.
    """
    def test_predict(self):
        test_df = pd.DataFrame({                                                        # Creating a DataFrame with two test points (two rows with 'feature1' and 'feature2')
            'feature1': [2.5, 3.5],
            'feature2': [6.5, 7.5]
        })
        predictions = self.knn.predict(test_df)                                         # Call the predict function to obtain predictions on the test data

        self.assertTrue(isinstance(predictions, np.ndarray))                            # Verify that the result returned is a numpy array

        self.assertEqual(len(predictions), len(test_df))                                # Check that the number of predictions matches the number of test points

        self.assertIn(predictions[0], self.y_train.values)                              # Ensure that the first prediction belongs to the original y_train labels
        self.assertIn(predictions[1], self.y_train.values)                              # Ensure that the second prediction belongs to the original y_train labels

    """
    This method tests that an exception is raised when an invalid value of k (negative) is provided.
    The expected behavior is that the function should raise a SystemExit error.
    """
    def test_invalid_k_value(self):
        knn = KnnAlgorithm(k=-1, x_train=self.x_train, y_train=self.y_train)            # Creating an instance of KnnAlgorithm with an invalid k value (-1)
        self.assertRaises(SystemExit, knn.get_neighbors, pd.Series([1, 2]))             # Verify that calling get_neighbors with a test point raises a SystemExit

if __name__ == '__main__':
    unittest.main()
