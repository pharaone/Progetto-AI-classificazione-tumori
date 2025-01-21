from Evaluation.Evaluator import Evaluator
from Preprocessing.impl.PreprocessorImplementation import PreprocessorImplementation


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

        evaluator = Evaluator(features, targets, metriche_selezionate)

        if scelta == "1":
            training_percentage = float(input("Inserisci la percentuale di training (es. 0.8 per 80%): "))
            risultati = evaluator.holdout_validation(training_percentage, k_neighbors)
            print("Risultati Holdout Validation:", risultati)

        elif scelta == "2":
            k_folds = int(input("Inserisci il numero di fold (es. 5): "))
            evaluator.k_fold_cross_validation(k_folds, k_neighbors)

        elif scelta == "3":
            k_folds = int(input("Inserisci il numero di fold per la validazione stratificata (es. 5): "))
            evaluator.stratified_cross_validation(k_folds, k_neighbors)

        else:
            print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()
