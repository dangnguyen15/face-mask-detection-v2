import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# ==================================================

# CONFIG

# ==================================================

DATASET_PATH = "FMD_DATASET"
MODEL_DIR = "model"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

MODEL_PATH = os.path.join(
MODEL_DIR,
"mask_detection_model.keras"
)

CSV_PATH = os.path.join(
MODEL_DIR,
"training_history.csv"
)

ACC_PLOT_PATH = os.path.join(
MODEL_DIR,
"accuracy_plot.png"
)

LOSS_PLOT_PATH = os.path.join(
MODEL_DIR,
"loss_plot.png"
)

# ==================================================

# CREATE OUTPUT DIRECTORY

# ==================================================

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================

# LOAD DATASET

# ==================================================

print("Loading dataset...")

image_paths = []
labels = []

for root, dirs, files in os.walk(DATASET_PATH):

```
if len(files) == 0:
    continue

class_name = os.path.relpath(
    root,
    DATASET_PATH
).replace("\\", "/")

for file in files:

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_paths.append(
            os.path.join(root, file)
        )

        labels.append(class_name)
```

df = pd.DataFrame({
"filename": image_paths,
"class": labels
})

print("\nClass Distribution:")
print(df["class"].value_counts())

# ==================================================

# TRAIN / VALIDATION SPLIT

# ==================================================

train_df, val_df = train_test_split(
df,
test_size=0.2,
random_state=42,
stratify=df["class"]
)

print(f"\nTrain samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ==================================================

# IMAGE GENERATORS

# ==================================================

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=20,
zoom_range=0.2,
width_shift_range=0.2,
height_shift_range=0.2,
horizontal_flip=True
)

val_datagen = ImageDataGenerator(
rescale=1./255
)

train_generator = train_datagen.flow_from_dataframe(
dataframe=train_df,
x_col="filename",
y_col="class",
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical",
shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
dataframe=val_df,
x_col="filename",
y_col="class",
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical",
shuffle=False
)

print("\nClasses:")
print(train_generator.class_indices)

NUM_CLASSES = len(
train_generator.class_indices
)

# ==================================================

# BUILD MODEL

# ==================================================

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
weights="imagenet",
include_top=False,
input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
base_model,

```
GlobalAveragePooling2D(),

Dense(
    256,
    activation="relu"
),

Dropout(0.3),

Dense(
    NUM_CLASSES,
    activation="softmax"
)
```

])

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.summary()

# ==================================================

# TRAIN MODEL

# ==================================================

print("\nTraining model...")

history = model.fit(
train_generator,
validation_data=val_generator,
epochs=EPOCHS
)

# ==================================================

# SAVE MODEL

# ==================================================

model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

# ==================================================

# SAVE TRAINING HISTORY

# ==================================================

history_df = pd.DataFrame({
"Epoch": range(
1,
len(history.history["accuracy"]) + 1
),
"Train Accuracy":
history.history["accuracy"],
"Validation Accuracy":
history.history["val_accuracy"],
"Train Loss":
history.history["loss"],
"Validation Loss":
history.history["val_loss"]
})

history_df.to_csv(
CSV_PATH,
index=False
)

print("\nTraining History:")

print(history_df)

# ==================================================

# PLOT ACCURACY

# ==================================================

plt.figure(figsize=(10,5))

plt.plot(
history.history["accuracy"],
marker="o",
label="Train Accuracy"
)

plt.plot(
history.history["val_accuracy"],
marker="o",
label="Validation Accuracy"
)

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(ACC_PLOT_PATH)

# ==================================================

# PLOT LOSS

# ==================================================

plt.figure(figsize=(10,5))

plt.plot(
history.history["loss"],
marker="o",
label="Train Loss"
)

plt.plot(
history.history["val_loss"],
marker="o",
label="Validation Loss"
)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(LOSS_PLOT_PATH)

# ==================================================

# EVALUATION

# ==================================================

print("\nEvaluating model...")

predictions = model.predict(
val_generator
)

y_pred = np.argmax(
predictions,
axis=1
)

cm = confusion_matrix(
val_generator.classes,
y_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
classification_report(
val_generator.classes,
y_pred,
target_names=list(
val_generator.class_indices.keys()
)
)
)

print("\nDone!")

print(f"Model      : {MODEL_PATH}")
print(f"History    : {CSV_PATH}")
print(f"Accuracy   : {ACC_PLOT_PATH}")
print(f"Loss Plot  : {LOSS_PLOT_PATH}")
