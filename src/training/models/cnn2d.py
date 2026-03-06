import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Mel spectrogram branch — input: (B, 1, 128, 47)
        self.mel_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # MFCC branch — input: (B, 1, 120, 47)
        self.mfcc_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Classifier — fused features: 128*16 + 64*16 = 3072
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4 + 64*4*4, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel, mfcc):
        m = self.mel_branch(mel).view(mel.size(0), -1)
        c = self.mfcc_branch(mfcc).view(mfcc.size(0), -1)
        return self.classifier(torch.cat([m, c], dim=1))
