import pandas as pd
from Evaluation.evaluators.HoldoutEvaluator import HoldoutEvaluator
from Evaluation.evaluators.KFoldEvaluator import KFoldEvaluator
from Evaluation.evaluators.StratifiedEvaluator import StratifiedEvaluator

"""
This file contains the static methods needed for the Evaluator Factory.
By calling each method with all the required inputs, an instance of the chosen evaluator is returned.
"""

"""
This method creates and returns and HoldoutEvaluator instance with the requested parameters (training_percentage
is specific to this method)
"""
def get_holdout_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                          k_neighbors:int , selected_distance: int, training_percentage: float):
    return HoldoutEvaluator(features, targets, metrics, k_neighbors, selected_distance, training_percentage)

"""
This method creates and returns and KfoldEvaluator instance with the requested parameters (k_times is the number of folds)
"""
def get_K_fold_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                         k_neighbors:int , selected_distance: int, k_times: int):
    return KFoldEvaluator(features, targets, metrics, k_neighbors, selected_distance, k_times)

"""
This method creates and returns and StratifiedEvaluator instance with the requested parameters (k_times is the number of folds)
"""
def get_stratified_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                             k_neighbors:int , selected_distance: int, k_times: int):
    return StratifiedEvaluator(features, targets, metrics, k_neighbors, selected_distance, k_times)