import pandas as pd

class Evaluator:
    __features : pd.DataFrame = None
    __targets : pd.Series = None

    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        self.__features = features
        self.__targets = targets

    def holdout_validation(self):
        pass

    def k_fold_cross_validation(self, k_times: int):
        pass

    def stratified_cross_validation(self, k_times: int):
        pass