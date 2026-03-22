"""
CNN Image Classification - Fashion-MNIST
=========================================
Requirements covered:
  [REQ-1] Train CNN on Fashion-MNIST
  [REQ-2] Conv + ReLU + Pooling blocks, BatchNorm & Dropout
  [REQ-3] Simple CNN vs Transfer Learning (MobileNetV2)
  [REQ-4] Report: accuracy, confusion matrix, training curves, model size & time

Install dependencies:
    pip install tensorflow numpy matplotlib seaborn scikit-learn
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-GUI backend (works on any machine)
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# ─────────────────────────────────────────────
# CONFIG  (change EPOCHS/SUBSET to go faster)
# ─────────────────────────────────────────────
EPOCHS        = 10          # small epoch count keeps it quick
BATCH_SIZE    = 64
SUBSET        = 10_000      # use only N training samples to stay fast
IMG_SIZE      = 32          # resize to 32×32 (MobileNet needs ≥32)
NUM_CLASSES   = 10
REPORT_DIR    = "cnn_report" # folder where all report images are saved

os.makedirs(REPORT_DIR, exist_ok=True)

CLASS_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ─────────────────────────────────────────────
# [REQ-1] LOAD & PREPARE Fashion-MNIST
# ─────────────────────────────────────────────
print("\n[REQ-1] Loading Fashion-MNIST …")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Restrict to SUBSET samples for speed
x_train, y_train = x_train[:SUBSET], y_train[:SUBSET]

# Grayscale (28×28,1) → resize to (32×32,3) so MobileNet can reuse weights
def preprocess(images):
    # Expand channel dim: (N,28,28) → (N,28,28,1)
    images = images[..., np.newaxis].astype("float32") / 255.0
    # Resize to IMG_SIZE × IMG_SIZE
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
    # Repeat single channel → 3 channels (needed by MobileNetV2)
    images = tf.repeat(images, 3, axis=-1)
    return images.numpy()

print("  Preprocessing images …")
x_train = preprocess(x_train)
x_test  = preprocess(x_test)

# One-hot encode labels
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"  Train: {x_train.shape}  Test: {x_test.shape}")

# ─────────────────────────────────────────────────────────────────
# [REQ-2] & [REQ-3a]  SIMPLE CNN
#   Architecture:
#     Conv → ReLU → BatchNorm → MaxPool   (block 1)
#     Conv → ReLU → BatchNorm → MaxPool   (block 2)
#     Flatten → Dense → Dropout → Output
# ─────────────────────────────────────────────────────────────────
def build_simple_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    [REQ-2] Convolution + ReLU activation + Pooling blocks
            BatchNorm after each conv layer
            Dropout before the final Dense layer
    """
    model = keras.Sequential([

        # ── Block 1 ──────────────────────────────────
        # [REQ-2] Convolution layer (32 filters, 3×3 kernel)
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        # [REQ-2] ReLU activation
        layers.Activation("relu"),
        # [REQ-2] Batch Normalisation (stabilises training)
        layers.BatchNormalization(),
        # [REQ-2] Max Pooling (down-samples spatial dims by 2)
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Block 2 ──────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.Activation("relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Block 3 ──────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.Activation("relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Classifier head ──────────────────────────
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        # [REQ-2] Dropout (regularisation – drops 40 % of neurons randomly)
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),

    ], name="SimpleCNN")
    return model

# ─────────────────────────────────────────────────────────────────
# [REQ-3b]  TRANSFER LEARNING with MobileNetV2
#   Base:    MobileNetV2 pretrained on ImageNet (frozen)
#   Head:    GlobalAvgPool → Dropout → Dense(10)
# ─────────────────────────────────────────────────────────────────
def build_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    [REQ-3] Transfer learning: load MobileNetV2 with ImageNet weights.
    The base layers are FROZEN – only the custom head is trained.
    """
    # Load pretrained base (no top classifier)
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,      # remove ImageNet classification head
        weights="imagenet"      # pretrained weights
    )
    # [REQ-3] Freeze base model – we do NOT retrain ImageNet weights
    base.trainable = False

    inputs  = keras.Input(shape=input_shape)
    # Pass through frozen MobileNetV2
    x = base(inputs, training=False)
    # [REQ-2] Pooling block (global average over spatial dims)
    x = layers.GlobalAveragePooling2D()(x)
    # [REQ-2] Dropout for regularisation
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_TL")
    return model

# ─────────────────────────────────────────────────────────────────
# TRAIN HELPER
# ─────────────────────────────────────────────────────────────────
def train_model(model, x_tr, y_tr, x_te, y_te):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    start = time.time()
    history = model.fit(
        x_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,   # 15 % of train data → validation
        verbose=1
    )
    elapsed = time.time() - start
    # Evaluate on held-out test set
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    return history, acc, elapsed

# ─────────────────────────────────────────────────────────────────
# BUILD & TRAIN
# ─────────────────────────────────────────────────────────────────
print("\n[REQ-3a] Training Simple CNN …")
simple_cnn = build_simple_cnn()
simple_cnn.summary()
hist_simple, acc_simple, time_simple = train_model(
    simple_cnn, x_train, y_train_oh, x_test, y_test_oh)

print("\n[REQ-3b] Training MobileNetV2 (Transfer Learning) …")
mobilenet = build_mobilenet_model()
mobilenet.summary()
hist_mobile, acc_mobile, time_mobile = train_model(
    mobilenet, x_train, y_train_oh, x_test, y_test_oh)

# ─────────────────────────────────────────────────────────────────
# PREDICTIONS (for confusion matrix)
# ─────────────────────────────────────────────────────────────────
pred_simple  = np.argmax(simple_cnn.predict(x_test, verbose=0), axis=1)
pred_mobile  = np.argmax(mobilenet.predict(x_test,  verbose=0), axis=1)

# ─────────────────────────────────────────────────────────────────
# MODEL SIZE (parameter count)
# ─────────────────────────────────────────────────────────────────
def param_count(model):
    return model.count_params()

params_simple = param_count(simple_cnn)
params_mobile = param_count(mobilenet)

# ─────────────────────────────────────────────────────────────────
# [REQ-4] REPORT PLOTS
# ─────────────────────────────────────────────────────────────────

# ── 1. Training / Validation Accuracy Curves ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training & Validation Curves  [REQ-4]", fontsize=14)

for ax, hist, title in [
    (axes[0], hist_simple, "Simple CNN"),
    (axes[1], hist_mobile, "MobileNetV2 (TL)")
]:
    ax.plot(hist.history["accuracy"],     label="Train Acc",  linewidth=2)
    ax.plot(hist.history["val_accuracy"], label="Val Acc",    linewidth=2, linestyle="--")
    ax.plot(hist.history["loss"],         label="Train Loss", linewidth=2)
    ax.plot(hist.history["val_loss"],     label="Val Loss",   linewidth=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(REPORT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150)
plt.close()
print(f"\n[REQ-4] Saved training curves → {curves_path}")

# ── 2. Confusion Matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Confusion Matrices  [REQ-4]", fontsize=14)

for ax, preds, title in [
    (axes[0], pred_simple, "Simple CNN"),
    (axes[1], pred_mobile, "MobileNetV2 (TL)")
]:
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
cm_path = os.path.join(REPORT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"[REQ-4] Saved confusion matrices → {cm_path}")

# ── 3. Comparison Bar Chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison  [REQ-4]", fontsize=14)

models     = ["Simple CNN", "MobileNetV2 (TL)"]
accuracies = [acc_simple * 100, acc_mobile * 100]
times      = [time_simple,      time_mobile]
params     = [params_simple / 1e6, params_mobile / 1e6]  # in millions

# Accuracy
axes[0].bar(models, accuracies, color=["steelblue", "coral"], width=0.5)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].set_title("Accuracy")
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

# Training time
axes[1].bar(models, times, color=["steelblue", "coral"], width=0.5)
axes[1].set_ylabel("Time (seconds)")
axes[1].set_title("Training Time")
for i, v in enumerate(times):
    axes[1].text(i, v + 1, f"{v:.0f}s", ha="center", fontweight="bold")

# Parameter count
axes[2].bar(models, params, color=["steelblue", "coral"], width=0.5)
axes[2].set_ylabel("Parameters (Millions)")
axes[2].set_title("Model Size (# Parameters)")
for i, v in enumerate(params):
    axes[2].text(i, v + 0.05, f"{v:.2f}M", ha="center", fontweight="bold")

plt.tight_layout()
cmp_path = os.path.join(REPORT_DIR, "model_comparison.png")
plt.savefig(cmp_path, dpi=150)
plt.close()
print(f"[REQ-4] Saved comparison chart → {cmp_path}")

# ── 4. Text Report ────────────────────────────────────────────────
report_txt = os.path.join(REPORT_DIR, "report.txt")
with open(report_txt, "w", encoding="utf-8") as f:
    sep = "=" * 60

    f.write(f"{sep}\n  CNN IMAGE CLASSIFICATION REPORT\n  Dataset: Fashion-MNIST\n{sep}\n\n")

    f.write("── SIMPLE CNN ──\n")
    f.write(f"  Test Accuracy   : {acc_simple*100:.2f}%\n")
    f.write(f"  Training Time   : {time_simple:.1f} s\n")
    f.write(f"  Parameters      : {params_simple:,}\n\n")
    f.write("  Classification Report:\n")
    f.write(classification_report(y_test, pred_simple, target_names=CLASS_NAMES))

    f.write(f"\n{sep}\n")

    f.write("── MobileNetV2 (Transfer Learning) ──\n")
    f.write(f"  Test Accuracy   : {acc_mobile*100:.2f}%\n")
    f.write(f"  Training Time   : {time_mobile:.1f} s\n")
    f.write(f"  Parameters      : {params_mobile:,}\n\n")
    f.write("  Classification Report:\n")
    f.write(classification_report(y_test, pred_mobile, target_names=CLASS_NAMES))

    f.write(f"\n{sep}\n")
    f.write("── COMPARISON SUMMARY ──\n")
    f.write(f"  {'Metric':<25} {'Simple CNN':>15} {'MobileNetV2 TL':>15}\n")
    f.write(f"  {'-'*55}\n")
    f.write(f"  {'Test Accuracy (%)':<25} {acc_simple*100:>15.2f} {acc_mobile*100:>15.2f}\n")
    f.write(f"  {'Training Time (s)':<25} {time_simple:>15.1f} {time_mobile:>15.1f}\n")
    f.write(f"  {'Parameters (M)':<25} {params_simple/1e6:>15.2f} {params_mobile/1e6:>15.2f}\n")
    f.write(f"\n{sep}\n")

print(f"[REQ-4] Saved text report → {report_txt}")

# ── Console summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Simple CNN     → Acc: {acc_simple*100:.2f}%  |  "
      f"Time: {time_simple:.0f}s  |  Params: {params_simple/1e6:.2f}M")
print(f"  MobileNetV2 TL → Acc: {acc_mobile*100:.2f}%  |  "
      f"Time: {time_mobile:.0f}s  |  Params: {params_mobile/1e6:.2f}M")
print("=" * 60)
print(f"\nAll report files saved in: ./{REPORT_DIR}/")
print("  training_curves.png")
print("  confusion_matrices.png")
print("  model_comparison.png")
print("  report.txt")