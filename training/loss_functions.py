import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        # Approx inverse proportions: 1/248, 1/95, 1/63 -> scaled locally
        if alpha is None:
            self.alpha = torch.tensor([0.15, 0.35, 0.50]) 
        else:
            self.alpha = torch.tensor(alpha)
            
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def get_class_weights(train_labels, device):
    labels  = (train_labels.tolist()
               if hasattr(train_labels, 'tolist')
               else list(train_labels))
    counts  = Counter(labels)
    total   = sum(counts.values())
    n       = len(counts)
    weights = torch.tensor(
        [total / (n * counts[i]) for i in range(n)],
        dtype=torch.float32
    ).to(device)
    print(f"Class weights → "
          f"Normal={weights[0]:.3f} "
          f"Murmur={weights[1]:.3f} "
          f"Abnormal={weights[2]:.3f}")
    return weights
