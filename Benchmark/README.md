# Benchmark

In questa cartella sono già presenti il modello migliore ottenuto durante la mia prima campagna sperimentale (best_model.h5) ed un dataset composto da 5000 profili con geometrie ed angoli di incidenza variabili.

A seguire la procedura per compiere uno studio analogo.

# Creazione train/validation set e test set

Necessari: profilo alare "NACA0012_40.dat" & "siga_compute_d_benchmark.py"

> Inserire il corretto path per il profilo alare nel codice python.
> Inserire numero di profili nel dataset creato (num_profiles)
> Inserire il range di variabilità per le funzioni di forma (wg2aer_range = 1 per questo studio)
> Inserire il numero di funzioni di forma da utilizzare per variare la geometria (n_par= 10 per questo studio)
> Inserire il range di variabilità dell'angolo di incidenza (alpha_range = 10 per questo studio)

Il dataset creato avrà dimensioni [num_profiles X 51], dove le prime 10 colonne sono i parametri di wg2aer, seguite dall'angolo d'incidenza ed in fine i 40 coefficienti di pressione per ogni profilo.

# Analisi NN

La cartella grid_search contiene i tre codici python che permettono di valutare le diverse strutture neurali.
