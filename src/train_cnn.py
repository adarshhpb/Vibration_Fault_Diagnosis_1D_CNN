

#%% CWRU Optimized Preprocessing Pipeline (ML + DL)

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LayerNormalization, GlobalAveragePooling1D, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input
from collections import Counter
# ---------------- CONFIGURATION ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "selected_mat")


SAMPLE_RATE = 12000
WINDOW_SIZE = 1024

FILE_LABELS = {
    "118.mat": "BallFault",
    "185.mat": "BallFault",
    "222.mat": "BallFault",

    "105.mat": "InnerRace",
    "169.mat": "InnerRace",
    "209.mat": "InnerRace",

    "130.mat": "OuterRace",
    "197.mat": "OuterRace",
    "234.mat": "OuterRace",
    
    
    "97.mat":  "Normal",
}


# --------------- HELPER FUNCTIONS ---------------

def load_de_fe_signals(mat_path):
    """Load first available DE and FE time signals from a .mat file."""
    data = sio.loadmat(mat_path)
    keys = list(data.keys())

    de_keys = [k for k in keys if k.endswith('_DE_time')]
    fe_keys = [k for k in keys if k.endswith('_FE_time')]

    if not de_keys or not fe_keys:
        raise ValueError(f"No DE/FE time keys found in {mat_path}")

    de = data[de_keys[0]].flatten()
    fe = data[fe_keys[0]].flatten()
    return de, fe


def max_abs_normalize(signal):
    """Normalize signal to [-1, 1] using max-abs scaling."""
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def create_windows(signal, window_size):
    """Slice 1D signal into non-overlapping windows."""
    num_windows = len(signal) // window_size
    if num_windows == 0:
        return np.empty((0, window_size))
    return signal[:num_windows * window_size].reshape(num_windows, window_size)


# --------------- MAIN PREPROCESSING LOOP ---------------

all_windows_raw = []   # for Deep Learning (raw windows)
all_labels_raw  = []

for fname, label in FILE_LABELS.items():
    mat_path = os.path.join(DATA_DIR, fname)
    print(f"Processing file: {fname}  -> label: {label}")

    # 1) Load DE and FE signals
    de_signal, fe_signal = load_de_fe_signals(mat_path)

    # 2) Normalize (DE only)
    de_norm = max_abs_normalize(de_signal)

    # 3) Windowing (DE only)
    de_windows = create_windows(de_norm, WINDOW_SIZE)

    # 4) Labels for these windows
    file_labels = np.array([label] * de_windows.shape[0])

    # 5) Store for raw pipeline (DL)
    all_windows_raw.append(de_windows)
    all_labels_raw.append(file_labels)

# --------------- FINAL DATASETS ---------------

X_windows_raw = np.vstack(all_windows_raw)       # (N_samples, 1024)
y_raw         = np.concatenate(all_labels_raw)   # (N_samples,)

print("X_windows_raw shape:", X_windows_raw.shape)
print("y_raw shape:        ", y_raw.shape)
print("Unique labels:", np.unique(y_raw))
print("\nClass distribution (window-level after segmentation):")
print(Counter(y_raw))

# 1) Encode labels as integers
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# 2) Build X and reshape for CNN
X = X_windows_raw.astype(np.float32).reshape(-1, WINDOW_SIZE, 1)
print("X shape before shuffle:", X.shape)

# 3) Shuffle X and y_int **together**
from sklearn.utils import shuffle
X, y_int = shuffle(X, y_int, random_state=42)

# 4) FIRST SPLIT: Train (70%) and Temp (30%) using **integer labels only** for stratify
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train_int, y_temp_int = train_test_split(
    X, y_int, test_size=0.3, random_state=42, stratify=y_int
)

# 5) SECOND SPLIT: Temp into Validation (15%) and Test (15%) using **integer labels only** for stratify
X_val, X_test, y_val_int, y_test_int = train_test_split(
    X_temp, y_temp_int, test_size=0.5, random_state=42, stratify=y_temp_int
)

print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)
print("X_test shape: ", X_test.shape)

# 6) ONE-HOT ENCODE **AFTER** all splits are complete, using integer labels only
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_int, num_classes=num_classes)
y_val   = to_categorical(y_val_int,   num_classes=num_classes)
y_test  = to_categorical(y_test_int,  num_classes=num_classes)
print("y_train shape:", y_train.shape)
print("y_val shape:  ", y_val.shape)
print("y_test shape: ", y_test.shape)


# --------------- BUILD SIMPLE CNN-1D MODEL ---------------

model = Sequential([
    # Input
    Input(shape=(WINDOW_SIZE, 1)),

    # large receptive field – capture global patterns
    Conv1D(64, kernel_size=17, activation='relu', padding='same'),
    LayerNormalization(),
    MaxPooling1D(pool_size=2),

    # still wide, more filters
    Conv1D(128, kernel_size=11, activation='relu', padding='same'),
    LayerNormalization(),
    MaxPooling1D(pool_size=2),

    # medium kernels
    Conv1D(256, kernel_size=7, activation='relu', padding='same'),
    LayerNormalization(),
    MaxPooling1D(pool_size=2),

    # dilated conv to catch subtle IR differences
        Conv1D(256, kernel_size=5, activation='relu',
            padding='same', dilation_rate=2),
        LayerNormalization(),
    MaxPooling1D(pool_size=2),

    # small kernels for fine details
        Conv1D(128, kernel_size=3, activation='relu',
            padding='same', dilation_rate=2),
        LayerNormalization(),

    # Global pooling instead of Flatten--- fewer params, less overfitting
    GlobalAveragePooling1D(),

    # Dense head
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early Call backs 

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_cnn_model.keras', 
    monitor='val_accuracy', 
    save_best_only=True
)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weight_dict = {i: class_weights[i] for i in range(num_classes)}
print(f"\nClass weights: {class_weight_dict}")

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_scheduler, checkpoint],
    verbose=1
)

# --------------- EVALUATE MODEL ---------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test_accuracy)

# --------------- CONFUSION MATRIX ---------------

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap='Blues'
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN Confusion Matrix')
plt.show()