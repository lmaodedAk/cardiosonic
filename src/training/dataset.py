import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class CardioDataset(Dataset):
    def __init__(self, cache_paths, labels, augment=False):
        self.cache_paths = cache_paths
        self.labels      = labels
        self.augment     = augment

    def __len__(self):
        return len(self.cache_paths)

    def __getitem__(self, idx):
        with open(self.cache_paths[idx], "rb") as f:
            data = pickle.load(f)

        log_mel = data["log_mel"].copy()
        mfcc    = data["mfcc"].copy()
        label   = self.labels[idx]

        # Back to 1x augmentation — 3x caused instability
        if self.augment:
            log_mel, mfcc = self._augment(log_mel, mfcc)
            
            # Additional domain-shift augmentation for Abnormal robustness
            if random.random() < 0.40:
                log_mel = log_mel + np.random.uniform(-0.005, 0.005, log_mel.shape).astype(np.float32)

        mel_t   = torch.tensor(log_mel).unsqueeze(0)
        mfcc_t  = torch.tensor(mfcc).unsqueeze(0)
        label_t = torch.tensor(label, dtype=torch.long)

        return mel_t, mfcc_t, label_t

    def _augment(self, log_mel, mfcc):
        # Time mask
        if random.random() < 0.5:
            t  = random.randint(1, 10)
            t0 = random.randint(0, log_mel.shape[1] - t)
            log_mel[:, t0:t0+t] = log_mel.mean()
            mfcc[:,    t0:t0+t] = mfcc.mean()

        # Frequency mask
        if random.random() < 0.4:
            f  = random.randint(1, 15)
            f0 = random.randint(0, log_mel.shape[0] - f)
            log_mel[f0:f0+f, :] = log_mel.mean()

        # Gaussian noise — mild
        if random.random() < 0.35:
            log_mel += np.random.normal(0, 0.05,
                            log_mel.shape).astype(np.float32)
            mfcc    += np.random.normal(0, 0.03,
                            mfcc.shape).astype(np.float32)

        # Amplitude scale
        if random.random() < 0.35:
            scale   = random.uniform(0.8, 1.2)
            log_mel = log_mel * scale
            mfcc    = mfcc * scale

        return log_mel, mfcc

def simulate_phone_mic(y, sample_rate=2000):
    """Simulates cheap phone microphone domain shift"""
    from scipy.signal import butter, filtfilt
    
    # 1. Phone mic low-end roll-off (Highpass @ 100Hz)
    nyq = sample_rate / 2.0
    b, a = butter(2, 100/nyq, btype='high')
    y_shifted = filtfilt(b, a, y)
    
    # 2. Uniform Quantization / Mic Hiss
    noise = np.random.uniform(-0.005, 0.005, len(y_shifted))
    return y_shifted + noise
