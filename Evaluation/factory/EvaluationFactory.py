import pandas as pd
from Evaluation.evaluators.HoldoutValidation import HoldoutValidation
from Evaluation.evaluators.KFoldValidation import KFoldValidation
from Evaluation.evaluators.StratifiedValidation import StratifiedValidation


def get_holdout_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str]):
    return HoldoutValidation(features, targets, metrics)

def get_K_fold_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str]):
    return KFoldValidation(features, targets, metrics)

def get_stratified_evaluator(features: pd.DataFrame, targets: pd.Series, metrics: list[str]):
    return StratifiedValidation(features, targets, metrics)