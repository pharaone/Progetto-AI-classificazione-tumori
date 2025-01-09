import pandas as pd
import sys

from pandas import DataFrame
from Preprocessing.Preprocessor import Preprocessor

"""
This class extends the Preprocessor class and implements it's preprocess method.
It does so by first loading the dataset from the input data path, cleaning up the data,
standardizing it and splitting it into features and target variable.
It has two variables: @__class_column_name which is the name of the column that will be used as 
the target variable, and a default data path in @__data_path.
"""
class PreprocessorImplementation(Preprocessor):
    __class_column_name = 'classtype_v1'        # name of the column that will be used as  the target variable
    __data_path = 'input.csv'                   # default (placeholder) data path

    """
    Empty class constructor that creates a new class instance
    """
    def __init__(self):
        pass

    """
    This method loads the dataset from the input data path and returns it as a Pandas DataFrame.
    It also checks and handles error while reading the file, such as FileNotFoundError or a 
    generic exception by returning an error message and quitting the execution.
    """
    def load_dataset(self) -> pd.DataFrame:
        print('Loading dataset...')                         # logging
        print('Reading data from ' + self.__data_path)      # logging
        try:
            df = pd.read_csv(self.__data_path)              # dataset is read from file
        except FileNotFoundError:                           # handles a FileNotFoundError
            sys.exit('Dataset not found.')                  # logs the error and quits
        except Exception as e:                              # handles every other exception
            print(e)                                        # prints the exception
            sys.exit('Error loading dataset.')              # logs the error and quits
        return df                                           # if no errors have occurred returns the dataframe

    """
    This method cleans up the dataframe given to it as a parameter:
    it first drops the sample code number column that is contained in the dataset per specification,
    then tries to convert all values to numeric, if it encounters errors during this process ( for example 
    if it finds a string and cannot convert it to numeric) it removes the row from the dataframe.
    It also removes all na rows and finally returns the dataframe.
    """
    def data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Sample code number'])            # drops not useful 'Sample code number' column
        df = df.apply(pd.to_numeric, errors='coerce')           # drops all non-numeric rows
        df = df.dropna()                                        # drops the rows with NA values
        return df                                               # returns the dataframe

    """
    This method standardizes the dataframe given to it as a parameter:
    It first removes the target column (__class_column_name) saving it in a temporary variable. 
    This is done to not modify this information which will be re-added after the dataframe is standardized.
    Then proceeds to standardize all the data by subtracting the mean of that column and dividing by the standard 
    deviation of said column.
    The output will be the dataframe standardized with the class column added back in.
    """
    def data_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        class_column = df[self.__class_column_name]
        df.drop(columns=[self.__class_column_name])
        standardized_df = (df - df.mean()) / df.std()
        standardized_df[self.__class_column_name] = class_column
        return standardized_df

    def get_targets_and_features(self, df: pd.DataFrame) -> tuple[DataFrame, DataFrame]:
        return df[self.__class_column_name], df.drop(columns=[self.__class_column_name])

    def preprocess(self, data_path: str) -> tuple[DataFrame, DataFrame]:
        self.__data_path = data_path
        df = self.load_dataset()
        df = self.data_cleanup(df)
        df = self.data_standardization(df)
        return self.get_targets_and_features(df)


