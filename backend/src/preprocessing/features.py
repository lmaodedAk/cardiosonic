import os
import csv
import pickle
import numpy as np
import librosa
from src.preprocessing.preprocess import preprocess_audio, TARGET_SR

def extract_features(y, sr=TARGET_SR):
    # Branch A: Log Mel Spectrogram → (128, 47)
    mel     = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Branch B: MFCC + Delta + Delta-Delta → (120, 47)
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512, hop_length=128)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_full = np.concatenate([mfcc, mfcc_d, mfcc_d2], axis=0)

    # Normalize each feature map
    def norm(x):
        return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)

    return norm(log_mel), norm(mfcc_full)


def precompute_all(manifest_path="data/manifest.csv",
                   cache_dir="data/processed"):
    os.makedirs(cache_dir, exist_ok=True)

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    total   = len(rows)
    done    = 0
    failed  = 0

    print(f"Caching features for {total} files...")
    print(f"{'='*50}")

    for i, row in enumerate(rows):
        fname      = os.path.basename(row["path"]).replace(".wav", ".pkl")
        cache_path = os.path.join(cache_dir, fname)

        if os.path.exists(cache_path):
            done += 1
            continue

        try:
            y               = preprocess_audio(row["path"])
            log_mel, mfcc   = extract_features(y)

            with open(cache_path, "wb") as f:
                pickle.dump({
                    "log_mel": log_mel,
                    "mfcc":    mfcc,
                    "label":   int(row["label"]),
                    "path":    row["path"]
                }, f)
            done += 1

            # Print progress every 25 files
            if (i + 1) % 25 == 0 or (i + 1) == total:
                pct = (i+1)/total*100
                bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
                print(f"  [{bar}] {i+1}/{total} ({pct:.0f}%) — "
                      f"{row['class_name']:10} | {os.path.basename(row['path'])}")

        except Exception as e:
            print(f"  FAILED [{i+1}] {row['path']}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"  Successful: {done}")
    print(f"  Failed:     {failed}")
    print(f"  Cached at:  {cache_dir}/")

    # Verify one sample
    sample = os.listdir(cache_dir)[0]
    with open(os.path.join(cache_dir, sample), "rb") as f:
        test = pickle.load(f)
    print(f"\nSample verification:")
    print(f"  log_mel shape: {test['log_mel'].shape}  (expected: (128, 47))")
    print(f"  mfcc shape:    {test['mfcc'].shape}     (expected: (120, 47))")
    print(f"  label:         {test['label']}")


if __name__ == "__main__":
    precompute_all()
