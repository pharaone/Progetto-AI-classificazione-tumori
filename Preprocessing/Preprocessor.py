import pandas as pd
import sys

class Preprocessor:
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
        df = df.dropna()
        return df

    def data_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        mean_value : float = 0.0
        std_value : float = 0.0
        for column in df.columns:
            if df[column].name != 'Sample code number':
                mean_value = df[column].mean()
                std_value = df[column].std()
                for row in column:
                    row = (row - mean_value) / std_value

    def preprocess(self) -> pd.DataFrame:
        df = self.load_dataset()
        df = self.data_cleanup(df)
        df = self.data_standardization(df)
        print(df)


