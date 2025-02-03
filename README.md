# Progetto-AI-classificazione-tumori

## Indice
1. [Introduzione](#1-Introduzione)
2. [Esecuzione del progetto](#2-Esecuzione-del-progetto)
3. [Preprocessing del Dataset](#3-preprocessing-del-dataset)
4. [K-nearest neighbors Algorithm (KNN)](#4-k-nearest-neighbors-algorithm-knn)
5. [Validazione del modello e metriche calcolate](#5-validazione-del-modello-e-metriche-calcolate)
6. [Risultati](#6-risultati)
7. [Conclusione](#7-conclusione)

### 1. Introduzione
Il progetto è stato sviluppato da Emanuele Antonio Faraone (@pharaone) e Stefano Imbalzano per il corso di Fondamenti di Intelligenza Artificiale (2024-2025).
Questo programma addestra e valuta le prestazioni di un classificatore di machine learning adatto a classificare i tumori come benigni o maligni in base alle caratteristiche fornite. 
Riceve in input un file .csv contenente il dataset che "preprocessa" per dividerlo in features e target label e per pulirlo da eventuali errori, poi addestra il classificatore k-nn con le caratteristiche che vengono fornite in input dall'utilizzatore ed ne valuta le performance utilizzando 3 diverse tecniche di divisione del dataset in training e test: Holdout, K-fold Cross Validation e Stratified Cross Validation.

### 2. Esecuzione del progetto
Durante l'esecuzione, l'utente è tenuto a specificare diversi parametri:
- Numero di vicini (k) da utilizzare per la classificazione.
- Metodo di validazione del modello (Holdout, K-Fold Cross Validation, Stratified Cross Validation).
- Numero di fold nel caso di K-Fold o Stratified Cross Validation.
- Metriche di valutazione da calcolare.

### 3. Preprocessing del dataset
Il preprocessing dei dati è un passaggio cruciale per garantire la qualità dei dati utilizzati per il training del modello. Le operazioni eseguite sono le seguenti:

- Caricamento del dataset: Il file CSV viene caricato e trasformato in un DataFrame di Pandas.
- Pulizia dei dati:
    - Rimozione della colonna "Sample Code Number" in quanto non utile ai fini della classificazione.
    - Conversione di tutti i valori in formato numerico, eliminando eventuali righe contenenti dati non numerici.
    - Eliminazione di valori mancanti.
- Standardizzazione: I valori delle feature vengono trasformati per avere media zero e deviazione standard unitaria.
- Suddivisione in target e feature: Il dataset viene diviso in un vettore delle etichette (target) e una matrice delle feature.

### 4. K-nearest neighbors Algorithm (KNN)
L'__algoritmo dei k-nearest neighbors (KNN)__ è un metodo di classificazione supervisionata che classifica un elemento sulla base della classe più frequente tra i suoi k vicini più prossimi. Il nostro programma implementa un classificatore KNN personalizzato, 
il quale permette di selezionare diverse strategie di distanza per il calcolo della similarità tra i punti del dataset.
L'implementazione prevede:
- __Ricerca dei vicini più prossimi__: Per ogni punto di test, viene calcolata la distanza rispetto a tutti i punti del dataset di training utilizzando la strategia di distanza selezionata. I k elementi più vicini vengono scelti in base alle distanze più piccole.
- __Scelta della strategia di distanza__: Il sistema utilizza un'architettura basata sul design pattern Factory con Strategy, che consente di selezionare dinamicamente il metodo di calcolo della distanza. Questo approccio garantisce una maggiore flessibilità, facilitando l'integrazione di nuove strategie di distanza senza modificare il codice esistente. Attualmente è implementata la distanza euclidea. 
Tuttavia, grazie alla struttura del Factory, è possibile aggiungere facilmente nuove metriche di distanza, come la distanza di Manhattan o la distanza di Minkowski, semplicemente implementando nuove strategie senza alterare il flusso principale dell'algoritmo.
- __Classificazione__: Una volta individuati i k vicini più prossimi, la classe viene assegnata in base alla maggioranza tra le etichette dei vicini. In caso di parità tra classi, il programma sceglie casualmente una delle classi con il numero massimo di occorrenze.

Il parametro "k" rappresenta il numero di vicini da prendere in considerazione per il processo di classificazione. Se si imposta k=1, l'oggetto da classificare viene associato alla categoria del punto più vicino. La scelta ottimale di "k" è cruciale per bilanciare correttamente l'accuratezza del modello e la sua capacità di generalizzazione. 
L'implementazione include inoltre il controllo degli errori per garantire che i parametri utilizzati (ad esempio, il numero di vicini k e la strategia di distanza) siano validi. In caso di errore, il programma gestisce le eccezioni e termina con un messaggio appropriato.

### 5. Validazione del modello e metriche calcolate
Per valutare le prestazioni del classificatore, il programma implementa tre tecniche di validazione:
- __Holdout Validation__: L'Holdout Validation è una tecnica di validazione utilizzata per valutare le prestazioni di un modello di machine learning. In questa metodologia, il dataset viene suddiviso in due insiemi separati: un insieme di addestramento e un insieme di test. La percentuale di dati destinata all'addestramento e al test è determinata dall'utente, con una tipica divisione di 70-30 o 80-20.

    Il processo inizia con la suddivisione casuale dei dati in due gruppi, utilizzando la percentuale specificata per l'addestramento. Poi si usa, in questo caso un K-Nearest Neighbors (KNN), viene poi addestrato utilizzando l'insieme di addestramento. 
    
    Successivamente, il modello addestrato viene utilizzato per fare previsioni sull'insieme di test, che non è stato visto durante la fase di addestramento, permettendo di valutare le sue capacità di generalizzazione.
    Una volta ottenute le previsioni, si procede con il calcolo di metriche di valutazione fondamentali per comprendere le prestazioni del modello.
- __K-Fold Cross Validation__: Il dataset viene suddiviso in k sottoinsiemi. Il modello viene addestrato k volte, utilizzando ogni volta k-1 sottoinsiemi per il training e il restante per il test.
- __Stratified Cross Validation__: Variante della K-Fold in cui ogni fold mantiene la stessa proporzione di classi del dataset originale, migliorando la rappresentatività dei dati di test.

Le metriche di valutazione disponibili sono:
- __Accuracy Rate__: L'accuratezza misura la percentuale di previsioni corrette rispetto al totale delle previsioni effettuate.


- __Error Rate__: Il tasso di errore misura la percentuale di previsioni errate rispetto al totale delle previsioni effettuate.


- __Sensitivity__:  La sensibilità misura la capacità del modello di identificare correttamente i positivi, ossia quanti veri positivi sono stati correttamente identificati.


- __Specificity__:  La specificità misura la capacità del modello di identificare correttamente i negativi, ossia quanti veri negativi sono stati correttamente identificati.


- __Geometric Mean__:  La media geometrica è una metrica che combina la sensibilità e la specificità in un unico valore, utilizzato per avere un indicatore equilibrato delle prestazioni del modello.


- __All the above__: Opzione per calcolare e visualizzare tutte le metriche sopra elencate in una sola analisi.

### 6. Risultati

### 7. Conclusione
In conclusione, questo progetto offre un ambiente potente e interattivo per la classificazione dei dati medici, con un focus particolare sull'uso dell'algoritmo KNN. La struttura del progetto è progettata per garantire un'ampia flessibilità, permettendo agli utenti di personalizzare e ottimizzare il modello a seconda delle esigenze specifiche del loro dataset. Ogni fase del processo, dal preprocessing dei dati all'addestramento del modello, fino alla validazione, è pensata per offrire una solida base di lavoro che consenta di ottenere risultati accurati e significativi.
Nel complesso, il progetto fornisce un workflow completo e strutturato, che aiuta non solo a sviluppare modelli di classificazione efficaci, ma anche a comprenderne a fondo il comportamento e le prestazioni. Questo è particolarmente importante in ambito medico, dove la precisione e l'affidabilità delle previsioni possono avere un impatto diretto sulla diagnosi e sul trattamento dei pazienti.