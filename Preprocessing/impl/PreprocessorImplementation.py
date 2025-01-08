from typing import Any

import pandas as pd
import sys

from pandas import DataFrame

from Preprocessing.Preprocessor import Preprocessor


class PreprocessorImplementation(Preprocessor):
    __class_column_name = 'classtype_v1'
    __data_path = 'input.csv'
    def __init__(self, data_path: str):
        self.__data_path = data_path

    def load_dataset(self) -> pd.DataFrame:
        print('Loading dataset...')
        print('Reading data from ' + self.__data_path)
        try:
            df = pd.read_csv(self.__data_path)
        except FileNotFoundError:
            sys.exit('Dataset not found.')
        except Exception as e:
            print(e)
            sys.exit('Error loading dataset.')
        return df

    def data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Sample code number'])
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        return df

    def data_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        class_column = df[self.__class_column_name]
        df.drop(columns=[self.__class_column_name])
        standardized_df = (df - df.mean()) / df.std()
        standardized_df[self.__class_column_name] = class_column
        return standardized_df

    def get_targets_and_features(self, df: pd.DataFrame) -> tuple[DataFrame, DataFrame]:
        return df.columns[self.__class_column_name], df.drop(columns=[self.__class_column_name])

    def preprocess(self) -> tuple[DataFrame, DataFrame]:
        df = self.load_dataset()
        df = self.data_cleanup(df)
        df = self.data_standardization(df)
        return self.get_targets_and_features(df)


