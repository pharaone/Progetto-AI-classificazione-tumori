# Progetto-AI-classificazione-tumori

## Indice
1. [Descrizione funzionamento progetto](#1-descrizione-funzionamento-progetto)

### 1. Descrizione funzionamento progetto
Il progetto è stato sviluppato da Emanuele Antonio Faraone (@pharaone) e Stefano Imbalzano per il corso di Fondamenti di Intelligenza Artificiale (2024-2025).
Questo programma addestra e valuta le prestazioni di un classificatore di machine learning adatto a classificare i tumori come benigni o maligni in base alle caratteristiche fornite. 
Riceve in input un file .csv contenente il dataset che "preprocessa" per dividerlo in features e target label e per pulirlo da eventuali errori, poi addestra il classificatore k-nn con le caratteristiche che vengono fornite in input dall'utilizzatore ed ne valuta le performance utilizzando 3 diverse tecniche di divisione del dataset in training e test: Holdout, K-fold Cross Validation e Stratified Cross Validation.

Il file .csv contenente il dataset è strutturato con queste colonne:
  - 'Sample code number' che contiene un ID randomico univoco per ogni riga
  - 9 colonne contenti le feature: Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses. Tutte le feature devono avere un valore compreso tra 1 e 10.
  - 'Class label' che può essere 2 se il tumore è classificato come benigno o 4 se maligno

L'utente durante l'esecuzione del programma deve fornire dei parametri per specificare come eseguire la valutazione:
  - k (numero di vicini da utilizzare per il classificatore)
  - come valutare il modello, se in Holdout, in K-fold Cross Validation e Stratified Cross Validation
    - se non sceglie l' Holdout deve specificare K (numero di esperimenti)
  - quali metriche vuole che siano validate   

**k-nearest neighbors Algorithm (KNN)**

L'algoritmo dei **k-nearest neighbors (KNN)** è un metodo di classificazione supervisionata che assegna una categoria a un oggetto basandosi sulla similarità con altri oggetti già etichettati nel dataset. Per effettuare la classificazione, KNN calcola la distanza tra il punto da classificare e tutti gli altri punti del dataset, identificando quelli più vicini.

Per determinare i vicini più prossimi, l'algoritmo utilizza diverse metriche di distanza, tra cui la distanza euclidea, che è frequentemente scelta per la sua semplicità ed efficacia.

Il parametro "k" rappresenta il numero di vicini da prendere in considerazione per il processo di classificazione. Se si imposta k=1, l'oggetto da classificare viene associato alla categoria del punto più vicino.
La scelta ottimale di "k" è cruciale per bilanciare correttamente l'accuratezza del modello e la sua capacità di generalizzazione. La determinazione di questo valore dipende dalla distribuzione e dalla variabilità dei dati, richiedendo spesso un'analisi empirica per individuare il compromesso ideale.