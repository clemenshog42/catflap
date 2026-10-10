import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from collections import Counter

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
DATASET_PATH = "/kaggle/input/cleanpreycls/CleanPreyData"

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Quick class distribution check
class_names = train_ds.class_names
all_train_labels = np.concatenate([labels.numpy() for _, labels in train_ds.unbatch()])
label_counts = Counter(all_train_labels)
print("📊 Training class distribution:", {class_names[int(k)]: int(v) for k, v in label_counts.items()})

# Augmentation
def preprocess_and_augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_left_right(image)
    return image, label

def preprocess(image, label):
    return image, label

train_ds = train_ds.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Train
print("🔧 Training...")
model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=callbacks)

# Evaluate
true_labels = np.concatenate([y.numpy() for _, y in val_ds.unbatch()])
pred_probs = model.predict(val_ds)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

cm = confusion_matrix(true_labels.astype(int), pred_labels)
ConfusionMatrixDisplay(cm, display_labels=["No Prey", "Prey"]).plot()
plt.title("Validation Set Confusion Matrix")
plt.show()

# Save
model.save("cat_prey_model.keras")
print("✅ Saved model as 'cat_prey_model.keras'")

# Inspect predictions
for image, label in val_ds.take(1):
    preds = model.predict(image)
    for i in range(5):
        confidence = preds[i][0]
        print(f"🧠 Prediction: {'Prey' if confidence > 0.5 else 'No Prey'} (Confidence: {confidence:.2%})")
        print("Ground truth:", "Prey" if label[i].numpy() == 1 else "No Prey")

# Convert to TFLite (float16)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("cat_prey_model_fp16.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model (float16) saved as 'cat_prey_model_fp16.tflite'")
