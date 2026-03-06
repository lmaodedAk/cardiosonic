import librosa
import numpy as np
from scipy.signal import butter, filtfilt

TARGET_SR  = 2000
TARGET_LEN = 6000  # 3 seconds

def preprocess_audio(path):
    # 1. Load
    y, sr = librosa.load(path, sr=None, mono=True)

    # 2. Resample to 2000 Hz
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # 3. Normalize amplitude
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / (max_val + 1e-8)

    # 4. Bandpass filter 20–400 Hz (cardiac band)
    nyq  = TARGET_SR / 2.0
    b, a = butter(4, [20/nyq, 400/nyq], btype='band')
    y    = filtfilt(b, a, y)

    # 5. Trim silence
    y, _ = librosa.effects.trim(y, top_db=40)

    # 6. Pad or crop to fixed length
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode='constant')
    else:
        y = y[:TARGET_LEN]

    assert y.shape == (TARGET_LEN,), f"Shape error: {y.shape}"
    assert not np.isnan(y).any(),    "NaN detected after preprocessing"

    return y.astype(np.float32)
