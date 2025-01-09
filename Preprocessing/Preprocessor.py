from abc import ABC, abstractmethod
import pandas as pd

"""
This abstract class can be implemented by a class capable of preprocessing a dataset.
It was created to give flexibility on how to preprocess data, the "implementing" class 
just needs to expose a preprocess method as described here and to return the processed dataset,
the internal workings of the implementation can be different without any problems to the code.
"""
class Preprocessor(ABC):
    """
    This abstract method can be implemented by a class capable of preprocessing a dataset.
    It takes the data path as input and returns the processed dataset divided in feature and target.
    """
    @abstractmethod
    def preprocess(self, data_path: str) -> [pd.DataFrame, pd.DataFrame]:
        pass