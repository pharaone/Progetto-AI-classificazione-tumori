from Preprocessing.impl.PreprocessorImplementation import PreprocessorImplementation
import Evaluation.factory.EvaluationFactory as EvaluationFactory


def main():
    data_path = 'input/version_1.csv'

    preprocessor = PreprocessorImplementation()
    targets, features = preprocessor.preprocess(data_path)

    while True:
        print("\nSeleziona la tecnica di valutazione:")
        print("1 - Holdout Validation")
        print("2 - K-Fold Cross Validation")
        print("3 - Stratified Cross Validation")
        print("0 - Esci")

        scelta = input("Inserisci il numero della tecnica desiderata: ")

        if scelta == "0":
            print("Uscita dal programma.")
            break

        k_neighbors = int(input("Inserisci il numero di vicini (k): "))

        # Richiesta della selezione delle metriche
        print("\nSeleziona le metriche da calcolare:")
        print("1 - Accuracy Rate")
        print("2 - Error Rate")
        print("3 - Sensitivity")
        print("4 - Specificity")
        print("5 - Geometric Mean")
        print("6 - Area Under the Curve")
        print("7 - Tutte le metriche")

        metriche_input = input("Inserisci i numeri delle metriche (separati da virgola): ")
        metriche_selezionate = metriche_input.split(",")

        print("\nSeleziona la strategia di distanza:")
        print("1 - Squared Euclidean Distance")
        distanza_scelta = int(input("Inserisci il numero della strategia desiderata: "))

        if scelta == "1":
            training_percentage = float(input("Inserisci la percentuale di training (es. 0.8 per 80%): "))
            holdout_evaluator = EvaluationFactory.get_holdout_evaluator(features, targets, metriche_selezionate)
            holdout_evaluator.holdout_validation(training_percentage, k_neighbors, distanza_scelta)

        elif scelta == "2":
            k_folds = int(input("Inserisci il numero di fold (es. 5): "))
            k_fold_evaluator = EvaluationFactory.get_K_fold_evaluator(features, targets, metriche_selezionate)
            k_fold_evaluator.k_fold_cross_validation(k_folds, k_neighbors, distanza_scelta)

        elif scelta == "3":
            k_folds = int(input("Inserisci il numero di fold per la validazione stratificata (es. 5): "))
            stratified_evaluator = EvaluationFactory.get_stratified_evaluator(features, targets, metriche_selezionate)
            stratified_evaluator.stratified_cross_validation(k_folds, k_neighbors, distanza_scelta)

        else:
            print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()
