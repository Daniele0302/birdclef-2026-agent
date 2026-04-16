"""
baseline_model.py — Baseline CNN model for BirdCLEF 2026

This script:
1. Loads a subset of the training data
2. Creates mel-spectrograms
3. Trains a simple CNN
4. Prints metrics in JSON format (for the agent)

It's the "starting point" — the first experiment the agent would run.
You can run it manually to verify the pipeline works:

    python baseline_model.py

IMPORTANT: you must have the data in the data/ folder
"""

import os
import json
import numpy as np

# Check that TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed!")
    print("Install with: pip install tensorflow")
    exit(1)

# Check that librosa is available
try:
    import librosa
    print(f"librosa available")
except ImportError:
    print("ERROR: librosa not installed!")
    print("Install with: pip install librosa")
    exit(1)

from config import N_CLASSES, BATCH_SIZE, QUICK_TRAIN_EPOCHS, QUICK_TRAIN_SAMPLES
from utils.data_loader import prepare_dataset


def build_baseline_cnn(input_shape=(128, 313, 1), n_classes=N_CLASSES):
    """
    Builds a simple baseline CNN.

    Architecture:
        3 convolutional blocks (Conv2D + BatchNorm + MaxPool)
        → GlobalAveragePooling → Dense → Sigmoid

    Args:
        input_shape: input size (128, 313, 1)
        n_classes: number of classes to predict (234)

    Returns:
        model: compiled Keras model ready for training
    """
    model = keras.Sequential([
        # --- Input definition ---
        # Tell the model the shape of incoming data
        keras.Input(shape=input_shape),
        
        # --- Block 1: 32 filters ---
        # Conv2D: 32 filters 3x3, activation='relu' (negative values -> 0)
        # padding='same' pads edges with zeros to keep spatial dimensions
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # BatchNormalization: normalize values between layers
        # Makes training more stable (prevents values from becoming too large/small)
        layers.BatchNormalization(),
        # MaxPooling2D: halves height and width
        # Takes each 2x2 block and keeps the maximum value
        # (128, 313) → (64, 156)
        layers.MaxPooling2D((2, 2)),
        
        # --- Block 2: 64 filters ---
        # More filters = more complex patterns
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # (64, 156) → (32, 78)
        
        # --- Block 3: 128 filters ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # GlobalAveragePooling2D: for each filter, computes the mean
        # across ALL spatial values. (128, 32, 78) → (128,)
        # Advantages vs Flatten:
        #   - Fewer parameters (less risk of overfitting)
        #   - Works with variable input sizes
        layers.GlobalAveragePooling2D(),
        
        # --- Classification ---
        layers.Dense(256, activation='relu'),
        # Dropout: during training, randomly "turns off" 30% of neurons
        # This helps prevent overfitting (model doesn't "memorize" data)
        layers.Dropout(0.3),
        # Final layer: 234 neurons with sigmoid
        # sigmoid: each neuron outputs a value between 0 and 1 INDEPENDENTLY
        # (different from softmax where all sum to 1)
        # This is essential for multi-label: multiple species can be present
        layers.Dense(n_classes, activation='sigmoid')
    ])
    
    # --- Compilation ---
    # Define how the model learns
    model.compile(
        # binary_crossentropy: loss for multi-label classification
        # Treats each species as an independent binary decision (yes/no)
        loss='binary_crossentropy',
        # Adam: adaptive optimizer, widely used
        # learning_rate=0.001: learning rate (default for Adam)
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # Metrics: AUC is the competition metric
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    return model


def main():
    """
    Runs training for the baseline model.
    Prints metrics in JSON format at the end (for the agent).
    """
    print("=" * 60)
    print("BASELINE MODEL — BirdCLEF 2026")
    print("=" * 60)
    
    # --- Step 1: Prepare data ---
    print("\n[1/3] Preparing data...")
    X_train, X_val, y_train, y_val, label_names = prepare_dataset(
        max_samples=QUICK_TRAIN_SAMPLES
    )
    
    # --- Step 2: Build model ---
    print("\n[2/3] Building model...")
    model = build_baseline_cnn()
    model.summary()  # Print model architecture and number of parameters
    
    # --- Step 3: Training ---
    print("\n[3/3] Training...")
    
    # Callbacks: EarlyStopping and ReduceLROnPlateau
    callbacks = [
        # EarlyStopping: stop training if the metric doesn't improve
        # patience=3: wait 3 epochs without improvement before stopping
        # restore_best_weights=True: at the end, use the best weights (not the last)
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            restore_best_weights=True,
            mode='max'  # 'max' because we want to maximize AUC
        ),
        # ReduceLROnPlateau: reduce the learning rate when it stops improving
        # factor=0.5: halve the learning rate
        # patience=2: wait 2 epochs before reducing
        # ReduceLROnPlateau: reduce the learning rate when it stops improving
        # factor=0.5: halve the learning rate
        # patience=2: wait 2 epochs before reducing
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=2,
            mode='max'
        )
    ]
    
    # model.fit returns a History object with training metrics for each epoch
    # We will extract the final AUC and loss from this object to print at the end
    history = model.fit(
        X_train, y_train,                     # training data
        validation_data=(X_val, y_val),        # validation data
        epochs=QUICK_TRAIN_EPOCHS,             # maximum number of epochs
        batch_size=BATCH_SIZE,                 # samples per batch
        callbacks=callbacks,                   # automatic actions
        verbose=1                              # show progress bar
    )
    
    # --- Step 4: Results ---
    # Extract final metrics from history
    # history.history is a dict with metrics per epoch
    # We take the final value (last completed epoch)
    val_auc = float(max(history.history.get('val_auc', [0])))
    val_loss = float(min(history.history.get('val_loss', [999])))
    train_auc = float(max(history.history.get('auc', [0])))
    epochs_trained = len(history.history['loss'])
    
    # save the model (optional, but useful for the agent to analyze later)
    model.save('best_model.keras')
    print("\nModel saved as 'best_model.keras'")
    
    # Print the metrics in JSON format
    # The agent will read them from stdout to understand how the experiment went
    metrics = {
        "val_auc": round(val_auc, 4),
        "val_loss": round(val_loss, 4),
        "train_auc": round(train_auc, 4),
        "epochs_trained": epochs_trained
    }
    print("\n" + json.dumps(metrics))


if __name__ == "__main__":
    main()
