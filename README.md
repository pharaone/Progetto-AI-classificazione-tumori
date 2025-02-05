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
Il preprocessing dei dati è una fase cruciale nel ciclo di vita di un modello di machine learning. Questa fase permette di trasformare i dati grezzi in un formato più pulito, strutturato e adatto all'analisi. Il processo si sviluppa in diverse fasi fondamentali, che vanno dal caricamento del dataset fino alla separazione delle variabili di input (feature) e della variabile target.
- __Caricamento del Dataset__:
La prima operazione consiste nel leggere i dati da un file di input. È fondamentale gestire eventuali errori di caricamento, come l'assenza del file o la presenza di formati non compatibili. Questa fase garantisce che il dataset sia disponibile in memoria sotto forma di una tabella strutturata, solitamente rappresentata come un DataFrame.


- __Pulizia dei Dati__:
Una volta caricato il dataset, è necessario effettuare un processo di pulizia. In questa fase si eseguono operazioni come:
  - __Eliminazione di colonne non necessarie__: Alcune colonne possono essere ridondanti o irrilevanti per l'analisi.
  - __Conversione dei dati__: I valori devono essere trasformati in un formato numerico per essere elaborati correttamente dal modello. Eventuali errori di conversione possono portare alla rimozione delle righe problematiche.
  - __Gestione dei valori mancanti__: Le righe contenenti valori nulli o mancanti vengono eliminate per garantire la qualità del dataset.


- __Standardizzazione dei Dati__:
Per garantire che i dati siano comparabili tra loro, è importante applicare una tecnica di standardizzazione. In questa fase, ogni colonna numerica viene trasformata in modo che abbia media pari a zero e deviazione standard pari a uno. Questo processo è particolarmente utile quando le variabili hanno scale diverse e aiuta i modelli a convergere più rapidamente durante l'addestramento. La colonna che rappresenta la variabile target viene esclusa dalla standardizzazione, poiché il suo valore deve rimanere inalterato.


- __Separazione di Feature e Target__:
  L'ultimo passaggio consiste nella suddivisione del dataset in due insiemi distinti:
  - __Feature__: Contengono le informazioni necessarie per la predizione.
  - __Target__: Rappresenta la variabile di output che il modello dovrà apprendere a predire.
  Questa separazione è essenziale per addestrare e valutare un modello di machine learning in modo corretto.

Il preprocessing è una fase determinante, perché permette di ottenere dati coerenti, privi di errori e ben strutturati, migliorando così l'efficacia del modello predittivo.

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
- __K-Fold Cross Validation__:  E' una tecnica di validazione per valutare la performance di un modello di machine learning. Il dataset viene suddiviso in k sottoinsiemi (folds) di dimensioni simili. 
Il modello viene addestrato su k-1 di questi sottoinsiemi e testato sull’ultimo rimanente. Questo processo si ripete k volte, cambiando il fold di test ogni volta. Alla fine, le metriche di valutazione vengono mediate su tutte le iterazioni per ottenere una stima più affidabile delle prestazioni. 
Questo metodo è utile per ridurre il rischio di overfitting rispetto alla semplice suddivisione tra training e test set, poiché il modello viene testato su diverse parti del dataset.
- __Stratified Cross Validation__: E' una variante del k-fold che mantiene la distribuzione originale delle classi in ogni fold. Questo è particolarmente utile per dataset sbilanciati, in cui alcune classi sono molto più frequenti di altre.
 La procedura è la stessa del k-fold, ma la suddivisione in fold avviene in modo che la proporzione delle classi sia simile a quella dell’intero dataset.

Le metriche di valutazione disponibili sono:
- __Accuracy Rate__: L'accuratezza misura la percentuale di previsioni corrette rispetto al totale delle previsioni effettuate.


- __Error Rate__: Il tasso di errore misura la percentuale di previsioni errate rispetto al totale delle previsioni effettuate.


- __Sensitivity__:  La sensibilità misura la capacità del modello di identificare correttamente i positivi, ossia quanti veri positivi sono stati correttamente identificati.


- __Specificity__:  La specificità misura la capacità del modello di identificare correttamente i negativi, ossia quanti veri negativi sono stati correttamente identificati.


- __Geometric Mean__:  La media geometrica è una metrica che combina la sensibilità e la specificità in un unico valore, utilizzato per avere un indicatore equilibrato delle prestazioni del modello.


- __All the above__: Opzione per calcolare e visualizzare tutte le metriche sopra elencate in una sola analisi.

### 6. Risultati
Dopo l'esecuzione della valutazione, il programma produce due output principali. Il primo è un file CSV che contiene i valori delle metriche selezionate, calcolati in base alle predizioni effettuate dal modello. Questo file permette di analizzare le prestazioni del modello in modo dettagliato e quantitativo. 
Il secondo output è un plot della matrice di confusione, salvato come immagine, che fornisce una rappresentazione visiva degli errori e delle corrette classificazioni effettuate. Questo grafico aiuta a comprendere meglio il comportamento del modello, specialmente in presenza di classi sbilanciate.
### 7. Conclusione
In conclusione, questo progetto offre un ambiente potente e interattivo per la classificazione dei dati medici, con un focus particolare sull'uso dell'algoritmo KNN. La struttura del progetto è progettata per garantire un'ampia flessibilità, permettendo agli utenti di personalizzare e ottimizzare il modello a seconda delle esigenze specifiche del loro dataset. 
Ogni fase del processo, dal preprocessing dei dati all'addestramento del modello, fino alla validazione, è pensata per offrire una solida base di lavoro che consenta di ottenere risultati accurati e significativi.
Nel complesso, il progetto fornisce un workflow completo e strutturato, che aiuta non solo a sviluppare modelli di classificazione efficaci, ma anche a comprenderne a fondo il comportamento e le prestazioni. Questo è particolarmente importante in ambito medico, dove la precisione e l'affidabilità delle previsioni possono avere un impatto diretto sulla diagnosi e sul trattamento dei pazienti.