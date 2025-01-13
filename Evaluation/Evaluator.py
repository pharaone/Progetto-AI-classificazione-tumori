import pandas as pd
import numpy as np

class Evaluator:
    __features : pd.DataFrame = None
    __targets : pd.Series = None

    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        self.__features = features
        self.__targets = targets

    def holdout_validation(self, training_percentage: float):
        pass

    def k_fold_cross_validation(self, k_times: int):
        features_subsets : [pd.DataFrame] = np.array_split(self.__features, k_times)
        targets_subsets : [pd.Series] = np.array_split(self.__targets, k_times)

        accuracy_rate_list = []
        error_rate_list = []
        sensitivity_rate_list = []
        specificity_rate_list = []
        geometric_mean_rate_list = []
        area_under_the_curve_rate_list = []

    def stratified_cross_validation(self, k_times: int):
        
        pass

    def calculate_metrics(self, y_test: pd.Series, y_prediction: pd.Series):
        pass