"""
train_cnn.py
Trains MobileNetV2 CNN on water images to classify contamination type.
Uses transfer learning — fast to train, high accuracy.
Run AFTER preprocess.py
Usage: python train_cnn.py
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                      BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed/images"
MODELS_DIR    = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 30      # EarlyStopping will stop sooner if needed
NUM_CLASSES = 5      # safe, turbid, chemical, sewage, heavy_metal
LEARNING_RATE = 1e-4

# Limit GPU memory growth (prevents OOM errors on small GPUs)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU available: {len(gpus)} device(s)")
else:
    print("No GPU found — training on CPU (slower but works)")

# ─── Load preprocessed image data ────────────────────────────────────────────
print("Loading image data...")
img_X_train = np.load(os.path.join(PROCESSED_DIR, "img_X_train.npy"))
img_X_test  = np.load(os.path.join(PROCESSED_DIR, "img_X_test.npy"))
img_y_train = np.load(os.path.join(PROCESSED_DIR, "img_y_train.npy"))
img_y_test  = np.load(os.path.join(PROCESSED_DIR, "img_y_test.npy"))
categories  = joblib.load(os.path.join(MODELS_DIR, "image_categories.pkl"))

print(f"Train images: {img_X_train.shape} | Test images: {img_X_test.shape}")
print(f"Categories: {categories}")

# Convert labels to one-hot encoding
img_y_train_oh = tf.keras.utils.to_categorical(img_y_train, NUM_CLASSES)
img_y_test_oh  = tf.keras.utils.to_categorical(img_y_test,  NUM_CLASSES)

# ─── Data Augmentation ────────────────────────────────────────────────────────
# Augmentation creates variety to prevent overfitting on small datasets
# Mimics real-world photo variation: lighting, angle, distance
datagen = ImageDataGenerator(
    rotation_range=20,       # Rotate up to 20 degrees
    width_shift_range=0.15,  # Shift horizontally
    height_shift_range=0.15, # Shift vertically
    horizontal_flip=True,    # Mirror image
    brightness_range=[0.7, 1.3],  # Simulate different lighting
    zoom_range=0.15,         # Slight zoom variation
    fill_mode="nearest"
)
datagen.fit(img_X_train)

# ─── Build Model (Transfer Learning with MobileNetV2) ─────────────────────────
# MobileNetV2 was pre-trained on ImageNet (1.4M images, 1000 classes)
# We freeze the base and add custom classification head for water quality
print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,         # Remove ImageNet classification head
    weights="imagenet"         # Load pre-trained weights
)

# Phase 1: Freeze base — only train our new head
base_model.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)          # Flatten spatial features
x = BatchNormalization()(x)              # Normalize activations
x = Dense(256, activation="relu")(x)    # Dense layer 1
x = Dropout(0.4)(x)                     # Prevent overfitting
x = Dense(128, activation="relu")(x)    # Dense layer 2
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)  # Final prediction

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Model parameters: {model.count_params():,}")
print(f"Trainable parameters: "
      f"{sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    # Stop training if val_loss doesn't improve for 5 epochs
    EarlyStopping(monitor="val_loss", patience=5,
                  restore_best_weights=True, verbose=1),

    # Save best model automatically
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "water_cnn_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate when plateau (helps fine-tuning)
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1)
]

# ─── Phase 1: Train head only ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Phase 1: Training classification head (base frozen)")
print("=" * 55)

history1 = model.fit(
    datagen.flow(img_X_train, img_y_train_oh, batch_size=BATCH_SIZE),
    steps_per_epoch=len(img_X_train) // BATCH_SIZE,
    epochs=15,
    validation_data=(img_X_test, img_y_test_oh),
    callbacks=callbacks,
    verbose=1
)

# ─── Phase 2: Fine-tune top layers of base model ──────────────────────────────
# Unfreeze the last 30 layers of MobileNetV2 for fine-tuning
print("\n" + "=" * 55)
print("Phase 2: Fine-tuning top layers of MobileNetV2")
print("=" * 55)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False   # Keep earlier layers frozen

# Use lower LR for fine-tuning to avoid destroying pre-trained weights
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    datagen.flow(img_X_train, img_y_train_oh, batch_size=BATCH_SIZE),
    steps_per_epoch=len(img_X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(img_X_test, img_y_test_oh),
    callbacks=callbacks,
    verbose=1
)

# ─── Evaluation ──────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(img_X_test, img_y_test_oh, verbose=0)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss:     {test_loss:.4f}")

y_pred_prob = model.predict(img_X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(img_y_test, y_pred, target_names=categories))

# ─── Training Curves Plot ─────────────────────────────────────────────────────
# Merge phase 1 + phase 2 histories
all_acc     = history1.history["accuracy"]     + history2.history["accuracy"]
all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
all_loss    = history1.history["loss"]         + history2.history["loss"]
all_val_loss = history1.history["val_loss"]    + history2.history["val_loss"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(all_acc,     label="Train Accuracy", color="#534AB7")
ax1.plot(all_val_acc, label="Val Accuracy",   color="#1D9E75", linestyle="--")
ax1.axvline(x=len(history1.history["accuracy"])-1,
            color="gray", linestyle=":", label="Fine-tune start")
ax1.set_title("CNN Training Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(all_loss,     label="Train Loss", color="#D85A30")
ax2.plot(all_val_loss, label="Val Loss",   color="#BA7517", linestyle="--")
ax2.axvline(x=len(history1.history["loss"])-1,
            color="gray", linestyle=":", label="Fine-tune start")
ax2.set_title("CNN Training Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "cnn_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: models/cnn_training_curves.png")

# ─── Save Final Model ────────────────────────────────────────────────────
model.save(os.path.join(MODELS_DIR, "water_cnn.h5"))
print(f"\nFinal model saved: models/water_cnn.h5")
print(f"Best model saved:  models/water_cnn_best.h5")
print("\nCNN training complete.")