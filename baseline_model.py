"""
baseline_model.py — Modello CNN baseline per BirdCLEF 2026

Questo script:
1. Carica un subset dei dati di training
2. Crea mel-spectrogrammi
3. Addestra una CNN semplice
4. Stampa le metriche in formato JSON (per l'agente)

È il "punto di partenza" — il primo esperimento che l'agente farebbe.
Puoi eseguirlo manualmente per verificare che la pipeline funzioni:

    python baseline_model.py

IMPORTANTE: devi avere i dati nella cartella data/
"""

import os
import json
import numpy as np

# Controlliamo che TensorFlow sia disponibile
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    print(f"TensorFlow versione: {tf.__version__}")
except ImportError:
    print("ERRORE: TensorFlow non installato!")
    print("Installa con: pip install tensorflow")
    exit(1)

# Controlliamo che librosa sia disponibile
try:
    import librosa
    print(f"librosa disponibile")
except ImportError:
    print("ERRORE: librosa non installato!")
    print("Installa con: pip install librosa")
    exit(1)

from config import N_CLASSES, BATCH_SIZE, QUICK_TRAIN_EPOCHS, QUICK_TRAIN_SAMPLES
from utils.data_loader import prepare_dataset


def build_baseline_cnn(input_shape=(128, 313, 1), n_classes=N_CLASSES):
    """
    Costruisce una CNN baseline semplice.
    
    Architettura:
        3 blocchi convoluzionali (Conv2D + BatchNorm + MaxPool)
        → GlobalAveragePooling → Dense → Sigmoid
    
    Args:
        input_shape: dimensione dell'input (128, 313, 1)
        n_classes: numero di classi da predire (234)
    
    Returns:
        model: modello Keras compilato e pronto per il training
    """
    model = keras.Sequential([
        # --- Definizione dell'input ---
        # Dice al modello la forma dei dati in ingresso
        keras.Input(shape=input_shape),
        
        # --- Blocco 1: 32 filtri ---
        # Conv2D: 32 filtri 3x3, activation='relu' (se valore < 0, diventa 0)
        # padding='same' aggiunge zeri ai bordi per mantenere le dimensioni
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # BatchNormalization: normalizza i valori tra un layer e l'altro
        # Rende il training più stabile (evita che i numeri diventino troppo grandi/piccoli)
        layers.BatchNormalization(),
        # MaxPooling2D: dimezza altezza e larghezza
        # Prende ogni blocco 2x2 e tiene solo il valore massimo
        # (128, 313) → (64, 156)
        layers.MaxPooling2D((2, 2)),
        
        # --- Blocco 2: 64 filtri ---
        # Filtri più numerosi = pattern più complessi
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # (64, 156) → (32, 78)
        
        # --- Blocco 3: 128 filtri ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # GlobalAveragePooling2D: per ogni filtro, calcola la media
        # di TUTTI i valori spaziali. (128, 32, 78) → (128,)
        # Vantaggi rispetto a Flatten:
        #   - Meno parametri (meno rischio di overfitting)
        #   - Funziona con input di dimensioni diverse
        layers.GlobalAveragePooling2D(),
        
        # --- Classificazione ---
        layers.Dense(256, activation='relu'),
        # Dropout: durante il training, "spegne" casualmente il 30% dei neuroni
        # Questo previene l'overfitting (il modello non "memorizza" i dati)
        layers.Dropout(0.3),
        # Layer finale: 234 neuroni con sigmoid
        # sigmoid: ogni neurone produce un valore tra 0 e 1 INDIPENDENTEMENTE
        # (diverso da softmax dove tutti sommano a 1)
        # Questo è essenziale per il multi-label: più specie possono essere presenti
        layers.Dense(n_classes, activation='sigmoid')
    ])
    
    # --- Compilazione ---
    # Definiamo come il modello impara
    model.compile(
        # binary_crossentropy: loss per classificazione multi-label
        # Tratta ogni specie come una decisione binaria indipendente (sì/no)
        loss='binary_crossentropy',
        # Adam: ottimizzatore adattivo, il più usato
        # learning_rate=0.001: velocità di apprendimento (default di Adam)
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # Metriche: AUC è la metrica della competizione
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    return model


def main():
    """
    Esegue il training del modello baseline.
    Stampa le metriche in formato JSON alla fine (per l'agente).
    """
    print("=" * 60)
    print("BASELINE MODEL — BirdCLEF 2026")
    print("=" * 60)
    
    # --- Step 1: Preparare i dati ---
    print("\n[1/3] Preparazione dati...")
    X_train, X_val, y_train, y_val, label_names = prepare_dataset(
        max_samples=QUICK_TRAIN_SAMPLES
    )
    
    # --- Step 2: Costruire il modello ---
    print("\n[2/3] Costruzione modello...")
    model = build_baseline_cnn()
    model.summary()  # stampa un riassunto dell'architettura
    
    # --- Step 3: Training ---
    print("\n[3/3] Training...")
    
    # Callback: azioni automatiche durante il training
    callbacks = [
        # EarlyStopping: ferma il training se la metrica non migliora
        # patience=3: aspetta 3 epoche senza miglioramento prima di fermarsi
        # restore_best_weights=True: alla fine, usa i pesi migliori (non gli ultimi)
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            restore_best_weights=True,
            mode='max'  # 'max' perché vogliamo AUC il più alto possibile
        ),
        # ReduceLROnPlateau: riduce il learning rate quando smette di migliorare
        # factor=0.5: dimezza il learning rate
        # patience=2: aspetta 2 epoche prima di ridurre
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=2,
            mode='max'
        )
    ]
    
    # model.fit() lancia il training
    # Restituisce un oggetto history con le metriche di ogni epoca
    history = model.fit(
        X_train, y_train,                     # dati di training
        validation_data=(X_val, y_val),        # dati di validazione
        epochs=QUICK_TRAIN_EPOCHS,             # numero massimo di epoche
        batch_size=BATCH_SIZE,                 # campioni per batch
        callbacks=callbacks,                   # azioni automatiche
        verbose=1                              # mostra la barra di progresso
    )
    
    # --- Step 4: Risultati ---
    # Estraiamo le metriche finali dalla history
    # history.history è un dizionario con le metriche per ogni epoca
    # Prendiamo l'ultimo valore (l'ultima epoca completata)
    val_auc = float(max(history.history.get('val_auc', [0])))
    val_loss = float(min(history.history.get('val_loss', [999])))
    train_auc = float(max(history.history.get('auc', [0])))
    epochs_trained = len(history.history['loss'])
    
    # Salviamo il modello
    model.save('best_model.keras')
    print("\nModello salvato come 'best_model.keras'")
    
    # Stampiamo le metriche in formato JSON
    # L'agente le leggerà da stdout per capire come è andato l'esperimento
    metrics = {
        "val_auc": round(val_auc, 4),
        "val_loss": round(val_loss, 4),
        "train_auc": round(train_auc, 4),
        "epochs_trained": epochs_trained
    }
    print("\n" + json.dumps(metrics))


if __name__ == "__main__":
    main()
