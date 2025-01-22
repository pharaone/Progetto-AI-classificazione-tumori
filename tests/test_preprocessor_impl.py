import unittest
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from Preprocessing.impl.PreprocessorImplementation import PreprocessorImplementation

"""
This test class is used to test that the Preprocessor class works as expected,
it mocks a sample data dataset and runs the methods of the Preprocessor on it
to check functionality
"""
class TestPreprocessorImplementation(unittest.TestCase):

    """
    This setUp method is used to create an instance of the Preprocessor class
    and the sample data set the other tests will use
    """
    def setUp(self):
        self.preprocessor = PreprocessorImplementation()    # creates preprocessor instance
        self.sample_data = {                                # creates sample data
            'Mitoses': [1, 2, np.nan, 4],                   # added null and not numeric values to test correctly the preprocessor
            'Normal Nucleoli': [3, 4, 5, 6],
            'Single Epithelial Cell Size': [7, 8, 9, 10],
            'uniformity_cellsize_xx': [11, 12, 13, 14],
            'clump_thickness_ty': [15, 16, 17, 18],
            'Marginal Adhesion': [19, 20, 21, 22],
            'Bland Chromatin': [23, 24, 25, 26],
            'classtype_v1': [2, 4, 2, 4],
            'Uniformity of Cell Shape': [27, 28, 29, 30],
            'bareNucleix_wrong': [31, 32, 33, 34],
            'Extra Column': [100, 101, 102, "hello"]
        }
        self.sample_df = pd.DataFrame(self.sample_data)     # converts sample data to a dataframe

    """
        This method tests if the load_dataset method is working as expected,
        by checking no exceptions are raised when loading the sample dataset
    """
    def test_load_dataset(self):
        mock_path = 'mock.csv'
        self.preprocessor._PreprocessorImplementation__data_path = mock_path
        pd.DataFrame.to_csv(self.sample_df, mock_path, index=False)                     # mock a csv containing the sample dataset
        loaded_df = self.preprocessor.load_dataset()
        self.sample_df['Extra Column'] = self.sample_df['Extra Column'].astype(str)     # ensure the Extra Column is treated as the same time for both dfs
        loaded_df['Extra Column'] = loaded_df['Extra Column'].astype(str)               # ensure the Extra Column is treated as the same time for both dfs
        self.assertTrue(loaded_df.equals(self.sample_df))                               # checks if the loaded dataset equals the mocked one


    """
    This method tests if the data_cleanup method is working as expected,
    by checking if unused columns are removed and that all the values are
    numeric and not null or broken values
    """
    def test_data_cleanup(self):
        cleaned_df = self.preprocessor.data_cleanup(self.sample_df)         # run data_cleanup method on test df
        expected_columns = [
            'Mitoses', 'Normal Nucleoli', 'Single Epithelial Cell Size', 'uniformity_cellsize_xx',
            'clump_thickness_ty', 'Marginal Adhesion', 'Bland Chromatin', 'classtype_v1',
            'Uniformity of Cell Shape', 'bareNucleix_wrong'
        ]
        self.assertListEqual(list(cleaned_df.columns), expected_columns)    # assert that the columns are the expected ones
        self.assertTrue(cleaned_df.equals(cleaned_df.apply(pd.to_numeric, errors='coerce')))        # assert all values are numeric
        self.assertFalse(cleaned_df.isnull().values.any())                                          # assert there are no null values

    """
    This method tests if the data_standardization method is working as expected,
    by checking if values are standardized after the method is applied to the sample df
    """
    def test_data_standardization(self):
        cleaned_df = self.preprocessor.data_cleanup(self.sample_df)                     # run cleanup on test df first
        standardized_df = self.preprocessor.data_standardization(cleaned_df)            # standardize the df
        self.assertAlmostEqual(standardized_df['Mitoses'].mean(), 0, places=1)  # checks that the mean of the column Mitoses is almost equal to zero
        self.assertAlmostEqual(standardized_df['Mitoses'].std(), 1, places=1)  # checks that the standard deviation of the column Mitoses is almost equal to one
        self.assertTrue(standardized_df['classtype_v1'].equals(cleaned_df['classtype_v1']) )                 # checks if the target column is still in the data frame unchanged

    """
        This method tests if the get_targets_and_features method is working as expected,
        by checking if values returned are a series (target) and a dataframe (features)
    """
    def test_get_targets_and_features(self):
        cleaned_df = self.preprocessor.data_cleanup(self.sample_df)                     # only clean up data, no need to standardize it for this test
        targets, features = self.preprocessor.get_targets_and_features(cleaned_df)      # runs the get_targets_and_features method on the cleaned df
        self.assertIsInstance(targets, Series)                                          # assert targets is a Series
        self.assertIsInstance(features, DataFrame)                                      # assert features is a DataFrame

    """
    This method tests if the preprocess method is working as expected,
    it does so by mocking a sample data dataset and running the preprocess method of the Preprocessor class
    and checking the targets and features have the correct type and length
    """
    def test_preprocess(self):
        mock_path = 'mock.csv'
        self.preprocessor._PreprocessorImplementation__data_path = mock_path
        pd.DataFrame.to_csv(self.sample_df, mock_path, index=False)             # mock a csv containing the sample dataset

        try:
            targets, features = self.preprocessor.preprocess(mock_path)         # runs the preprocess method on the mocked csv
            self.assertIsInstance(targets, Series)                              # assert targets is a Series
            self.assertIsInstance(features, DataFrame)                          # assert features is a DataFrame
            self.assertTrue(len(targets) == len(features))                      # assert targets and features have the same length
        finally:
            import os
            if os.path.exists(mock_path):
                os.remove(mock_path)                                            # cleans the mock csv from memory

if __name__ == '__main__':
    unittest.main()
