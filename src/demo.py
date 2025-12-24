import numpy as np
import scipy.io
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# ---------------- CONFIG ----------------
WINDOW_SIZE = 1024
CLASSES = ['BallFault', 'InnerRace', 'Normal', 'OuterRace']

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)

# ---------------- UTILS ----------------
def load_de_signal(mat_file):
    mat = scipy.io.loadmat(mat_file)
    for key in mat:
        if 'DE_time' in key:
            return mat[key].ravel()
    raise ValueError("DE signal not found")

def create_windows(signal, window_size):
    n = len(signal) // window_size
    return np.array([
        signal[i*window_size:(i+1)*window_size]
        for i in range(n)
    ])

# ---------------- DEMO ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_cnn_model.keras")
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "selected_mat", "210.mat")

def run_demo():
    model = load_model(MODEL_PATH)

    signal = load_de_signal(DATA_PATH)
    windows = create_windows(signal, WINDOW_SIZE)
    windows = windows[..., np.newaxis]

    preds = model.predict(windows, verbose=0)
    avg_pred = np.mean(preds, axis=0)
    pred_class = np.argmax(avg_pred)

    label = label_encoder.inverse_transform([pred_class])[0]
    confidence = np.max(avg_pred) * 100

    print(f"Predicted Fault Type: {label}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    run_demo()
