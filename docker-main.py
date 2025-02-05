import os
from Preprocessing.impl.PreprocessorImplementation import PreprocessorImplementation
import Evaluation.factory.EvaluationFactory as EvaluationFactory


def main():
    global evaluator
    # Carica le variabili d'ambiente
    data_path = os.getenv("DATA_PATH", "input/version_1.csv")
    evaluation_method = os.getenv("EVALUATION_METHOD", "1")
    k_neighbors = int(os.getenv("K_NEIGHBORS", "5"))
    metrics_selected = os.getenv("METRICS", "1,2,3,4,5,6,7").split(",")
    distance_strategy = int(os.getenv("DISTANCE_STRATEGY", "1"))
    training_percentage = float(os.getenv("TRAINING_PERCENTAGE", "0.8"))
    k_folds = int(os.getenv("K_FOLDS", "5"))

    print(f"Dataset: {data_path}")
    print(f"Metodo di valutazione scelto: {evaluation_method}")
    print(f"Numero di vicini (k): {k_neighbors}")
    print(f"Metriche selezionate: {metrics_selected}")
    print(f"Strategia di distanza: {distance_strategy}")

    preprocessor = PreprocessorImplementation()
    targets, features = preprocessor.preprocess(data_path)

    if evaluation_method == "1":
        print(f"Utilizzo Holdout Validation con training {training_percentage * 100}%")
        evaluator = EvaluationFactory.get_holdout_evaluator(features, targets, metrics_selected,
                                                                    k_neighbors, distance_strategy, training_percentage)

    elif evaluation_method == "2":
        print(f"Utilizzo K-Fold Cross Validation con {k_folds} folds")
        evaluator = EvaluationFactory.get_K_fold_evaluator(features, targets, metrics_selected,
                                                                   k_neighbors, distance_strategy, k_folds)

    elif evaluation_method == "3":
        print(f"Utilizzo Stratified Cross Validation con {k_folds} folds")
        evaluator = EvaluationFactory.get_stratified_evaluator(features, targets, metrics_selected,
                                                                          k_neighbors, distance_strategy, k_folds)
    else:
        print("Metodo di valutazione non valido.")

    evaluator.evaluate()


if __name__ == "__main__":
    main()
