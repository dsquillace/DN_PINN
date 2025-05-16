# INFORMAZIONI VARIE
All'interno della cartella PINN è possibile trovare i codici utilizzati per la realizzazione di una rete neurale informata della fisica.
Nella cartella Benchmark si trovano i codici utilizzati per identificare la migliore struttura neurale (Priva di informazioni fisiche) per prevedere il CP (ed il CL di conseguenza)

# PINN
Utilizzare "siga_compute..." per costruire i dataset con i quali allenare/testare la rete.
Assicurarsi che il percorso per la geometria sia corretto e che nel file di input di WG2AER sia richiamato correttamente.

Far girare il codice PINN_39pt_1.py nella cartella contenente il file di dati chiamato "filtered_dataset.txt".
Per modificare il contributo dell'informazione fisica è necessario modificare manualmente lo scalare che premoltiplica il parametro "continuity" nella funzione custom_loss

Una volta allenata la rete, inserire nella cartella un nuovo set di dati sotto il nome di "dataTEST.txt" per verificare le capacità della rete.

# Benchmark
