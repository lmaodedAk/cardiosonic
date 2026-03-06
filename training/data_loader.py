import csv
import os
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from training.dataset import CardioDataset

MANIFEST_PATH = "data/manifest.csv"
CACHE_DIR     = "data/processed"
BATCH_SIZE    = 32
NUM_WORKERS   = 0

def load_manifest():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    paths, labels, patient_ids = [], [], []
    missing = 0
    for row in rows:
        cache_path = os.path.join(
            CACHE_DIR,
            os.path.basename(row["path"]).replace(".wav", ".pkl")
        )
        if os.path.exists(cache_path):
            paths.append(cache_path)
            labels.append(int(row["label"]))
            patient_ids.append(row["patient_id"])
        else:
            missing += 1

    if missing > 0:
        print(f"WARNING: {missing} cache files missing — run features.py first")

    print(f"\nLoaded {len(paths)} samples")
    dist = Counter(labels)
    print(f"Class distribution: "
          f"Normal={dist[0]}, Murmur={dist[1]}, Abnormal={dist[2]}")

    return (np.array(paths),
            np.array(labels),
            np.array(patient_ids))


def make_sampler(labels):
    class_weights = {
        0: 1.0,   # Normal
        1: 2.2,   # Murmur
        2: 2.8    # Abnormal — 3.1 overcorrected, 2.8 was stable
    }
    weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights     = torch.tensor(weights, dtype=torch.float),
        num_samples = len(weights),
        replacement = True
    )


def get_loaders():
    paths, labels, patient_ids = load_manifest()

    # Patient-level split: 70 / 15 / 15
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, test_idx = next(gss1.split(paths, labels, groups=patient_ids))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
    tr_idx, val_idx = next(
        gss2.split(paths[tv_idx], labels[tv_idx],
                   groups=patient_ids[tv_idx])
    )
    tr_idx  = tv_idx[tr_idx]
    val_idx = tv_idx[val_idx]

    # Print splits
    print(f"\nSplit summary:")
    print(f"  Train:  {len(tr_idx):3d} samples | "
          f"{Counter(labels[tr_idx])}")
    print(f"  Val:    {len(val_idx):3d} samples | "
          f"{Counter(labels[val_idx])}")
    print(f"  Test:   {len(test_idx):3d} samples | "
          f"{Counter(labels[test_idx])}")

    # Leakage check
    tp = set(patient_ids[tr_idx])
    vp = set(patient_ids[val_idx])
    ep = set(patient_ids[test_idx])
    assert not tp & ep, f"LEAKAGE: {len(tp & ep)} patients in train+test"
    assert not tp & vp, f"LEAKAGE: {len(tp & vp)} patients in train+val"
    print(f"  Leakage check: PASSED ✓")

    # Build datasets
    train_ds = CardioDataset(paths[tr_idx],   labels[tr_idx],   augment=True)
    val_ds   = CardioDataset(paths[val_idx],  labels[val_idx],  augment=False)
    test_ds  = CardioDataset(paths[test_idx], labels[test_idx], augment=False)

    sampler = make_sampler(labels[tr_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler,   num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False,     num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False,     num_workers=NUM_WORKERS)

    # Verify first batch
    _, _, y_b = next(iter(train_loader))
    counts = torch.bincount(y_b, minlength=3)
    print(f"\nFirst batch: Normal={counts[0].item()} "
          f"Murmur={counts[1].item()} "
          f"Abnormal={counts[2].item()} "
          f"(should be roughly balanced)")

    return train_loader, val_loader, test_loader, labels[tr_idx]
