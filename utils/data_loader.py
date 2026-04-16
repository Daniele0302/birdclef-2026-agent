"""
utils/data_loader.py — Caricamento e preparazione dati per BirdCLEF 2026

Questo modulo si occupa di:
1. Leggere train.csv e taxonomy.csv
2. Creare le label multi-label (vettori binari di 234 posizioni)
3. Preparare il dataset pronto per il training
4. Fare train/validation split

Uso:
    from utils.data_loader import prepare_dataset
    X_train, X_val, y_train, y_val, label_names = prepare_dataset(max_samples=2000)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_CSV, TAXONOMY_CSV, TRAIN_AUDIO_DIR,
    N_CLASSES, VALIDATION_SPLIT
)
from utils.audio_pipeline import load_and_process_audio


def load_metadata():
    """
    Carica e prepara i metadati del training set.
    
    Returns:
        train_df: DataFrame con i dati di training
        taxonomy_df: DataFrame con tutte le 234 specie
        label_names: lista ordinata dei 234 nomi di specie (primary_label)
    
    Spiegazione passo per passo:
        1. Leggiamo train.csv — contiene info su ogni registrazione
        2. Leggiamo taxonomy.csv — contiene le 234 specie target
        3. Estraiamo la lista ordinata di nomi di specie
           Questa lista definisce l'ORDINE delle 234 colonne nell'output
    """
    # pd.read_csv() legge un file CSV e lo mette in un DataFrame
    # Un DataFrame è come una tabella Excel in Python
    train_df = pd.read_csv(TRAIN_CSV)
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    
    # sorted() ordina la lista in ordine alfabetico
    # Questo è importante: l'ordine deve essere lo stesso
    # nel training e nella submission!
    label_names = sorted(taxonomy_df['primary_label'].unique().tolist())
    
    print(f"Training set: {len(train_df)} registrazioni")
    print(f"Specie target: {len(label_names)}")
    
    return train_df, taxonomy_df, label_names


def create_label_vector(primary_label, secondary_labels, label_names):
    """
    Crea un vettore binario multi-label per una registrazione.
    
    Args:
        primary_label: la specie principale (es: "banana")
        secondary_labels: lista di specie secondarie (es: "['rubthr1', 'houspa']")
        label_names: lista ordinata delle 234 specie
    
    Returns:
        label_vec: array numpy di 234 zeri e uni
                   1 = la specie è presente, 0 = non presente
    
    Esempio:
        Se label_names = ["banana", "houspa", "osprey", ...]
        e primary_label = "banana", secondary_labels = ["osprey"]
        → label_vec = [1, 0, 1, 0, 0, ...]
                       ^banana  ^osprey
    
    Spiegazione:
        Creiamo un vettore di 234 zeri. Poi mettiamo 1 nelle posizioni
        corrispondenti alle specie presenti nella registrazione.
        Usiamo un dizionario (label_to_idx) per trovare velocemente
        la posizione di ogni specie nella lista.
    """
    # Creiamo un dizionario che mappa ogni nome → la sua posizione
    # Es: {"banana": 0, "houspa": 1, "osprey": 2, ...}
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    
    # Inizializziamo il vettore con tutti zeri
    # np.zeros crea un array di zeri della lunghezza specificata
    # dtype=np.float32 usa numeri decimali a 32 bit (standard per CNN)
    label_vec = np.zeros(len(label_names), dtype=np.float32)
    
    # Segniamo la specie primaria
    # str() converte in stringa (per sicurezza, nel caso sia un numero)
    primary = str(primary_label)
    if primary in label_to_idx:
        label_vec[label_to_idx[primary]] = 1.0
    
    # Segniamo le specie secondarie (se ci sono)
    # secondary_labels è una stringa tipo "['rubthr1', 'houspa']"
    # Dobbiamo parsarla per estrarre i nomi
    if isinstance(secondary_labels, str) and secondary_labels != '[]':
        try:
            # ast.literal_eval converte la stringa in una vera lista Python
            import ast
            sec_list = ast.literal_eval(secondary_labels)
            for sec in sec_list:
                sec = str(sec)
                if sec in label_to_idx:
                    label_vec[label_to_idx[sec]] = 1.0
        except (ValueError, SyntaxError):
            # Se il parsing fallisce, ignoriamo le secondarie
            pass
    
    return label_vec


def prepare_dataset(max_samples=None, random_state=42):
    """
    Prepara il dataset completo: carica audio, crea mel-spectrogrammi e label.
    
    Args:
        max_samples: se specificato, usa solo N campioni (per test rapidi)
                     Se None, usa tutto il dataset
        random_state: seed per la riproducibilità (stesso numero = stesso split)
    
    Returns:
        X_train: array (N_train, 128, 313, 1) — mel-spectrogrammi per training
        X_val: array (N_val, 128, 313, 1) — mel-spectrogrammi per validazione
        y_train: array (N_train, 234) — label multi-label per training
        y_val: array (N_val, 234) — label multi-label per validazione
        label_names: lista delle 234 specie
    
    Spiegazione del flusso:
        1. Carichiamo i metadati (CSV)
        2. Opzionalmente, prendiamo solo un subset (per velocità)
        3. Per ogni file audio:
           a. Creiamo il mel-spectrogram
           b. Creiamo il vettore di label
        4. Dividiamo in train e validation
    """
    # Step 1: Carichiamo i metadati
    train_df, taxonomy_df, label_names = load_metadata()
    
    # Step 2: Opzionalmente, prendiamo un subset casuale
    if max_samples is not None and max_samples < len(train_df):
        # .sample(n) prende n righe casuali dal DataFrame
        # random_state=42 garantisce che lo stesso subset venga scelto ogni volta
        train_df = train_df.sample(n=max_samples, random_state=random_state)
        print(f"Usando subset di {max_samples} campioni")
    
    # Step 3: Processiamo ogni file audio
    spectrograms = []   # qui metteremo tutti i mel-spectrogrammi
    labels = []         # qui metteremo tutte le label
    skipped = 0         # contatore file saltati (non trovati o corrotti)
    
    total = len(train_df)
    for idx, (_, row) in enumerate(train_df.iterrows()):
        # Stampiamo il progresso ogni 100 file
        if idx % 100 == 0:
            print(f"  Processamento: {idx}/{total} ({idx/total*100:.0f}%)")
        
        # Costruiamo il percorso completo del file audio
        # row['filename'] è qualcosa come "banana/XC12345.ogg"
        filepath = os.path.join(TRAIN_AUDIO_DIR, row['filename'])
        
        # Verifichiamo che il file esista
        if not os.path.exists(filepath):
            skipped += 1
            continue
        
        # Creiamo il mel-spectrogram
        mel = load_and_process_audio(filepath)
        if mel is None:
            skipped += 1
            continue
        
        # Creiamo il vettore di label
        label_vec = create_label_vector(
            row['primary_label'],
            row['secondary_labels'],
            label_names
        )
        
        spectrograms.append(mel)
        labels.append(label_vec)
    
    print(f"Processati: {len(spectrograms)}, Saltati: {skipped}")
    
    # Step 4: Convertiamo le liste in array numpy
    # np.array converte una lista di matrici in un unico array multidimensionale
    X = np.array(spectrograms)   # shape: (N, 128, 313)
    y = np.array(labels)         # shape: (N, 234)
    
    # Aggiungiamo la dimensione del canale per la CNN
    # La CNN vuole (N, altezza, larghezza, canali)
    # np.expand_dims aggiunge un asse in posizione -1 (ultima)
    # (N, 128, 313) → (N, 128, 313, 1)
    X = np.expand_dims(X, axis=-1)
    
    # Step 5: Dividiamo in train e validation
    # train_test_split mischia i dati e li divide in due parti
    # test_size=0.2 significa 80% training, 20% validazione
    # stratify non è possibile con multi-label, usiamo split casuale
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} campioni")
    print(f"Validation set: {X_val.shape[0]} campioni")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Output shape: {y_train.shape[1:]}")
    
    return X_train, X_val, y_train, y_val, label_names
