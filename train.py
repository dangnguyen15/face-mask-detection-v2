#!/usr/bin/env python3
"""
Face Mask Detection (FMD) Training Script
Author: Senior Machine Learning Engineer
Description: Production-ready pipeline to train a 6-class face mask detection 
             model using Transfer Learning with MobileNetV2 in Keras 3 / TF 2.20+.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configure TensorFlow logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

# --- CONSTANTS & CONFIGURATION ---
DATASET_DIR = Path("FMD_DATASET")
OUTPUT_DIR = Path("model")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
RANDOM_STATE = 42

# Explicit mapping of subdirectories to 6 distinct target classes
CLASS_MAPPING = {
    "incorrect_mask_mc": 0,
    "incorrect_mask_mmc": 1,
    "with_mask_complex": 2,
    "with_mask_simple": 3,
    "without_mask_complex": 4,
    "without_mask_simple": 5
}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_and_prepare_dirs():
    """Validates dataset existence and creates output directories."""
    if not DATASET_DIR.exists() or not DATASET_DIR.is_dir():
        print(f"[ERROR] Dataset directory '{DATASET_DIR}' does not exist.")
        print("Please ensure the dataset structure matches the requirements.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_paths():
    """
    Scans the structured directory and maps images to their respective 6 classes.
    Returns:
        paths (list of str): Absolute or relative paths to images.
        labels (list of int): Integer labels corresponding to CLASS_MAPPING.
    """
    paths = []
    labels = []
    
    # Expected structured sub-paths
    structure = {
        "incorrect_mask/mc": "incorrect_mask_mc",
        "incorrect_mask/mmc": "incorrect_mask_mmc",
        "with_mask/complex": "with_mask_complex",
        "with_mask/simple": "with_mask_simple",
        "without_mask/complex": "without_mask_complex",
        "without_mask/simple": "without_mask_simple"
    }
    
    print("\n--- [1/5] Scanning Dataset Structure ---")
    class_counts = {k: 0 for k in CLASS_MAPPING.keys()}
    
    for sub_dir, class_name in structure.items():
        target_dir = DATASET_DIR / sub_dir
        if not target_dir.exists():
            print(f"[WARNING] Expected directory missing: {target_dir}")
            continue
            
        for file_path in target_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(str(file_path))
                labels.append(CLASS_MAPPING[class_name])
                class_counts[class_name] += 1

    # Print Dataset Summary
    print("\nDataset Summary:")
    print(f"Total Images Found: {len(paths)}")
    print("\nClass Mapping & Image Counts:")
    for class_name, idx in CLASS_MAPPING.items():
        print(f" - {class_name} (ID: {idx}): {class_counts[class_name]} images")
        
    if len(paths) == 0:
        print("[ERROR] No valid images found. Training aborted.")
        sys.exit(1)
        
    return np.array(paths), np.array(labels)


def load_and_preprocess_image(path, label):
    """tf.data pipeline function to read, decode, and preprocess images."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    # MobileNetV2 preprocess input standardizes pixels to [-1, 1] range
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def build_data_pipelines(train_paths, train_labels, val_paths, val_labels):
    """Creates performance-optimized tf.data.Dataset pipelines with Data Augmentation."""
    
    # Define Data Augmentation Sequential Block (Keras 3 compliant)
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(factor=0.2),
        keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
    ], name="data_augmentation")

    # Training Dataset Pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=RANDOM_STATE)
    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    # Apply data augmentation on batched tensor items to leverage vectorized execution
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Validation Dataset Pipeline
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model():
    """Constructs the Transfer Learning network using MobileNetV2 base."""
    print("\n--- [2/5] Constructing Architecture (MobileNetV2 Transfer Learning) ---")
    
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    # Freeze the feature extractor backbone
    base_model.trainable = False

    # Define Top Classification Architecture
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="FMD_MobileNetV2_Classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


def plot_and_save_metrics(history):
    """Generates and stores separate performance metrics graphs."""
    print("\n--- [4/5] Generating Training Performance Plots ---")
    
    # Save training metrics to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"[INFO] History metrics saved to: {OUTPUT_DIR / 'training_history.csv'}")

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
    plt.title('Model Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_plot.png", dpi=150)
    plt.close()

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
    plt.title('Model Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_plot.png", dpi=150)
    plt.close()
    
    print(f"[INFO] Performance plots exported successfully to '{OUTPUT_DIR}/'.")


def evaluate_final_model(model, val_paths, val_labels):
    """Executes deterministic evaluation and generates explicit analytical metrics reports."""
    print("\n--- [5/5] Executing Final Model Evaluation ---")
    
    # Construct clean sequential non-shuffled dataset for deterministic predictions
    eval_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    eval_ds = eval_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Generate predictions
    predictions = model.predict(eval_ds)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate evaluation loss and accuracy metrics
    loss, accuracy = model.evaluate(eval_ds, verbose=0)
    print(f"\nFinal Validation Dataset Results:")
    print(f" - Validation Loss:     {loss:.4f}")
    print(f" - Validation Accuracy: {accuracy * 100:.2f}%")
    
    # Target Class Names ordered by index
    target_names = [REVERSE_CLASS_MAPPING[i] for i in range(NUM_CLASSES)]
    
    # Generate Classification Report
    print("\nClassification Report:")
    report = classification_report(val_labels, predicted_classes, target_names=target_names)
    print(report)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(val_labels, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=target_names, 
        yticklabels=target_names
    )
    plt.title("Confusion Matrix - Face Mask Detection")
    plt.ylabel("True Categorical Labels")
    plt.xlabel("Predicted Categorical Labels")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[INFO] Confusion Matrix visualization saved to: {OUTPUT_DIR / 'confusion_matrix.png'}\n")


def main():
    # 1. Pipeline preparation and check environment
    check_and_prepare_dirs()
    paths, labels = load_dataset_paths()
    
    # Perform a verified stratified split to keep class distribution balance uniform
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, 
        labels, 
        test_size=0.20, 
        stratify=labels, 
        random_state=RANDOM_STATE
    )
    
    print(f"Stratified Split Completed: {len(train_paths)} Train | {len(val_paths)} Validation")
    
    # 2. Build datasets
    train_ds, val_ds = build_data_pipelines(train_paths, train_labels, val_paths, val_labels)
    
    # 3. Model assembly
    model = build_model()
    
    # 4. Configure Production Callback Routines
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 5. Model execution
    print("\n--- [3/5] Starting Model Optimization / Training Phase ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save structural final checkpoint
    model.save(OUTPUT_DIR / "final_model.keras")
    print(f"[INFO] Final artifact checkpoint saved to: {OUTPUT_DIR / 'final_model.keras'}")
    
    # 6. Evaluation and reporting output visualization
    plot_and_save_metrics(history)
    evaluate_final_model(model, val_paths, val_labels)
    print("[SUCCESS] Production training execution pipeline finished perfectly.")


if __name__ == "__main__":
    main()