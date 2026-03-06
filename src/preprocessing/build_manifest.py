import os
import csv
import librosa
from collections import Counter

DATA_DIR      = "data/raw"
MANIFEST_PATH = "data/manifest.csv"
CLASSES       = ["Normal", "Murmur", "Abnormal"]

rows = []
skipped = 0

for label_idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    files = sorted([f for f in os.listdir(class_dir) if f.endswith(".wav")])
    print(f"Processing {class_name}: {len(files)} files...")

    for fname in files:
        fpath = os.path.join(class_dir, fname)
        try:
            y, sr = librosa.load(fpath, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            # Extract patient ID — adjust split logic if your filenames differ
            # Assumes format: patient_001_rec1.wav OR 12345_rec.wav etc
            base = os.path.splitext(fname)[0]
            # No real patient IDs in this dataset — use full filename as unique ID
            patient_id = base  # each recording treated as independent

            rows.append({
                "path":       fpath,
                "label":      label_idx,
                "class_name": class_name,
                "patient_id": patient_id,
                "duration":   round(duration, 2),
                "sr":         sr,
                "quality":    "ok" if duration >= 1.5 else "short"
            })
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            skipped += 1

# Write manifest
os.makedirs("data", exist_ok=True)
with open(MANIFEST_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'='*40}")
print(f"MANIFEST COMPLETE")
print(f"  Total recordings: {len(rows)}")
print(f"  Skipped:          {skipped}")
dist = Counter(r["class_name"] for r in rows)
for k, v in dist.items():
    print(f"  {k:10}: {v} ({v/len(rows)*100:.1f}%)")

# Check patient ID uniqueness
patients = Counter(r["patient_id"] for r in rows)
print(f"\n  Unique patients: {len(patients)}")
print(f"  Avg recordings per patient: {len(rows)/len(patients):.1f}")
print(f"  Sample patient IDs: {list(patients.keys())[:5]}")
