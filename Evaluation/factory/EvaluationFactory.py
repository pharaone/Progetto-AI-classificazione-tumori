import pandas as pd
from Evaluation.evaluators.HoldoutEvaluator import HoldoutEvaluator
from Evaluation.evaluators.KFoldEvaluator import KFoldEvaluator
from Evaluation.evaluators.StratifiedEvaluator import StratifiedEvaluator


def get_holdout_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                          k_neighbors:int , selected_distance: int, training_percentage: float):
    return HoldoutEvaluator(features, targets, metrics, k_neighbors, selected_distance, training_percentage)

def get_K_fold_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                         k_neighbors:int , selected_distance: int, k_times: int):
    return KFoldEvaluator(features, targets, metrics, k_neighbors, selected_distance, k_times)

def get_stratified_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str],
                             k_neighbors:int , selected_distance: int, k_times: int):
    return StratifiedEvaluator(features, targets, metrics, k_neighbors, selected_distance, k_times)