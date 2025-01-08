from abc import ABC, abstractmethod
import pandas as pd

class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        pass