# Benchmark

In questa cartella sono già presenti il modello migliore ottenuto durante la mia prima campagna sperimentale (best_model.h5) ed un dataset composto da 5000 profili con geometrie ed angoli di incidenza variabili.

A seguire la procedura per compiere uno studio analogo.

# Creazione train/validation set e test set

Necessari: profilo alare "NACA0012_40.dat" & "siga_compute_d_benchmark.py"

> Inserire il corretto path per il profilo alare nel codice python.

>Inserire il corretto path per wg2aer. 

> Inserire numero di profili nel dataset creato (num_profiles)

> Inserire il range di variabilità per le funzioni di forma (wg2aer_range = 1 per questo studio)

> Inserire il numero di funzioni di forma da utilizzare per variare la geometria (n_par= 10 per questo studio)

> Inserire il range di variabilità dell'angolo di incidenza (alpha_range = 10 per questo studio)

Il dataset creato avrà dimensioni [num_profiles X 51], dove le prime 10 colonne sono i parametri di wg2aer, seguite dall'angolo d'incidenza ed in fine i 40 coefficienti di pressione per ogni profilo.

# Analisi NN

La cartella grid_search contiene i tre codici python che permettono di valutare le diverse strutture neurali. L'ordine dell'analisi è 1) "batchVSmse_nn_train.py" 2) "wdVSlr_nn_train.py" 3) "lVSn_nn_train.py".

Per compiere qualsiasi tipo di analisi è necessario avere due set di dati nella cartella di lavoro; "filtered_dataset.txt" il quale viene utilizzato per training e validation della rete neurale e "test_data.txt" contenente i dati con i quali ogni rete creata valuta l'MSE finale e quindi le prestazioni della rete stessa.

> "batchVSmse_nn_train.py": Valuta il variare dell'MSE al variare del batch_size. Viene effettuato con valori di Learning Rate, Weight Decay, numero di neuroni e layer fissati.
>
> "wdVSlr_nn_train.py": Fissato il batch_size ottimale trovato prima, valuta il variare dell'MSE per ogni combinazione di 10 diversi valori di Weight Decay e 10 di Learning Rate. Inoltre è possibile scegliere di impostare una metodologia di riduzione del Learning Rate a sceltra tra: Valore Fisso, Step Decay, Cosine Decay e Cosine Decay with Warm Restart.
>
> "lVSn_nn_train.py": Fissato il batch_size, WD e LR migliori individuati precedentemente, permette di valutare l'MSE di 100 diverse reti neurali definite da un numero di Layer e Neuroni per layer variabile.

# PLOT

Nel caso in cui non fossero stati creati i plot alla fine dei rispettivi allenamenti, nella cartella "plots" è possibile trovare gli script per creare i plot a partire dal "file mse_results.csv" in output dagli allenamenti

# Verifica e ricostruzione CP-CL

All'interno della cartella cp_reconstruction ci sono due codici. Per poterli utilizzare è necessario avere il file "best_model.h5" relativo alla migliore rete identificata e, volendo, un dataset da utilizzare per verificare le capacità predittive.

> "cp_plot.py": utilizzando le stesse accortezze per i path, range e numero parametri utilizzati durante la creazione del dataset, permette di generare una nuova geometria e controllare la distribuzione della pressione reale confrontata con quella predetta dalla rete neurale. Calcola inoltre il Cl reale e quello predetto.

> "prediction_testset.py": Svolge predizioni del Cp per un grande dataset.

