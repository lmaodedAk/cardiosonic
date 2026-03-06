import librosa
import numpy as np
from scipy.signal import butter, filtfilt

TARGET_SR  = 2000
TARGET_LEN = 6000

def process_uploaded_file(file_path):
    """
    MUST be identical to preprocessing/preprocess.py
    Any difference = wrong predictions
    """
    # 1. Load
    y, sr = librosa.load(file_path, sr=None, mono=True)
    print(f"Loaded: sr={sr}, length={len(y)}, duration={len(y)/sr:.2f}s")

    # 2. Resample to 2000 Hz
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    print(f"After resample: length={len(y)}")

    # 3. Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / (max_val + 1e-8)

    # 4. Bandpass 20-400 Hz
    nyq  = TARGET_SR / 2.0
    b, a = butter(4, [20/nyq, 400/nyq], btype='band')
    y    = filtfilt(b, a, y)

    # 5. Trim silence
    y, _ = librosa.effects.trim(y, top_db=40)
    print(f"After trim: length={len(y)}")

    # 6. Fixed length
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode='constant')
    else:
        y = y[:TARGET_LEN]

    return y.astype(np.float32)


def extract_features_for_inference(y):
    """
    MUST use identical parameters to preprocessing/features.py
    """
    sr = TARGET_SR

    # Log Mel — IDENTICAL params to training
    mel     = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # MFCC + Delta + Delta-Delta — IDENTICAL to training
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512, hop_length=128)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_full = np.concatenate([mfcc, mfcc_d, mfcc_d2], axis=0)

    # Normalize — IDENTICAL to training
    def norm(x):
        return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)

    log_mel   = norm(log_mel)
    mfcc_full = norm(mfcc_full)

    print(f"log_mel shape: {log_mel.shape}  (expected: (128, 47))")
    print(f"mfcc shape:    {mfcc_full.shape} (expected: (120, 47))")

    return log_mel, mfcc_full


def predict(file_path, model, device):
    CLASSES = ["Normal", "Murmur", "Abnormal"]

    # Process
    y               = process_uploaded_file(file_path)
    log_mel, mfcc   = extract_features_for_inference(y)

    # To tensors
    import torch
    mel_t  = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)
    mfcc_t = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(mel_t, mfcc_t)
        probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred   = int(np.argmax(probs))

    result = {
        "predicted_class":      CLASSES[pred],
        "confidence":           round(float(probs[pred]), 4),
        "probabilities": {
            "Normal":   round(float(probs[0]), 4),
            "Murmur":   round(float(probs[1]), 4),
            "Abnormal": round(float(probs[2]), 4)
        },
        "low_confidence": bool(float(probs[pred]) < 0.60),
        "warning": "Low confidence — consider re-recording"
                   if float(probs[pred]) < 0.60 else None
    }

    print(f"\nPrediction: {result['predicted_class']} "
          f"({result['confidence']*100:.1f}% confidence)")
    print(f"All probs: {result['probabilities']}")

    return result

if __name__ == "__main__":
    import torch
    from training.models.cnn2d import CNN2D
    
    device = torch.device('cpu')
    model  = CNN2D(num_classes=3).to(device)
    model.load_state_dict(torch.load('saved_models/best_model.pt', map_location=device))
    
    # Find a normal file to test
    import glob
    normal_files = glob.glob('data/raw/Normal/*.wav')
    if normal_files:
        test_file = normal_files[0]
        print(f"Testing with known Normal file: {test_file}")
        result = predict(test_file, model, device)
        print(result)
    else:
        print("No Normal files found to test.")
