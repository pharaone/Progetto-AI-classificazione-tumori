import pandas as pd
import sys

from pandas import DataFrame
from Preprocessing.Preprocessor import Preprocessor

"""
This class implements the Preprocessor class and implements it's preprocess method.
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
    The load_dataset method loads the dataset from the input data path and returns it as a Pandas DataFrame.
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
    The data_cleanup method cleans up the dataframe given to it as a parameter:
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
    The data_standardization method standardizes the dataframe given to it as a parameter:
    It first removes the target column (__class_column_name) saving it in a temporary variable. 
    This is done to not modify this information which will be re-added after the dataframe is standardized.
    Then proceeds to standardize all the data by subtracting the mean of that column and dividing by the standard 
    deviation of said column.
    The output will be the dataframe standardized with the class column added back in.
    """
    def data_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        class_column = df[self.__class_column_name]                     # stores class column in a temporary variable
        df.drop(columns=[self.__class_column_name])                     # drops class column temporarily

        standardized_df = (df - df.mean()) / df.std()                   # standardizes all data
        standardized_df[self.__class_column_name] = class_column        # adds class column back in the dataframe

        return standardized_df                                          # returns the standardized dataframe

    """
    The get_targets_and_features method splits the dataframe given to it as a parameter into two dataframes:
    one for target variable and one for features.
    It does so by returning a tuple with one element containing the class column and the other one
    all the other columns.
    """
    def get_targets_and_features(self, df: pd.DataFrame) -> tuple[DataFrame, DataFrame]:
        return df[self.__class_column_name], df.drop(columns=[self.__class_column_name])        # returns the tuple with one element containing the class
                                                                                                # column and the other one all the other columns
    """
    The preprocess method overrides the abstract method of the Preprocessor class.
    Takes as a parameter the data path of the dataset and puts it in the local variable __data_path
    and runs different methods:
    1.  load_dataset() to load the dataset from the input data path
    2.  data_cleanup() to cleand up the dataset from unwanted and broken data
    3.  data_standardization() to standardize the dataframe
    4.  get_targets_and_features() which returns the target and features dataframes to called this method 
    """
    def preprocess(self, data_path: str) -> tuple[DataFrame, DataFrame]:
        self.__data_path = data_path                                        # give the datapath information to the local variable to be used by the other methods

        df = self.load_dataset()                                            # loads the dataset
        df = self.data_cleanup(df)                                          # cleans up the dataset
        df = self.data_standardization(df)                                  # standardizes the dataset
        return self.get_targets_and_features(df)                            # returns target and features dataframes


