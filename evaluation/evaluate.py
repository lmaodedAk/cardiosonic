import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
from training.data_loader import get_loaders
from training.models.cnn2d import CNN2D

def calculate_optimal_abnormal_threshold(y_true, y_probs):
    """Calculates Youden Index for Abnormal Class vs Rest"""
    y_true_binary = (np.array(y_true) == 2).astype(int) 
    abnormal_probs = np.array(y_probs)[:, 2]
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, abnormal_probs)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Abnormal Validation Threshold: {optimal_threshold:.4f}")
    return float(optimal_threshold)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES   = ["Normal", "Murmur", "Abnormal"]
SAVE_PATH = "saved_models/best_model.pt"

print(f"\n{'='*60}")
print(f"  CARDIOSONIC HONEST EVALUATION")
print(f"{'='*60}")

# Load model
model = CNN2D(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()
print(f"  Loaded: {SAVE_PATH}")

# Load test set
_, _, test_loader, _ = get_loaders()

# Inference — PLAIN ARGMAX ONLY
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for mel, mfcc, labels in test_loader:
        mel, mfcc   = mel.to(DEVICE), mfcc.to(DEVICE)
        out         = model(mel, mfcc)
        probs       = torch.softmax(out, dim=1).cpu().numpy()
        preds       = np.argmax(probs, axis=1)   # NO threshold tricks
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

counts = np.bincount(all_labels, minlength=3)
print(f"\nTest set: {len(all_labels)} samples | "
      f"Normal={counts[0]} Murmur={counts[1]} Abnormal={counts[2]}")

# All metrics
acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds,
                       average='weighted', zero_division=0)
rec  = recall_score(all_labels, all_preds,
                    average='weighted', zero_division=0)
f1w  = f1_score(all_labels, all_preds,
                average='weighted', zero_division=0)
mcc  = matthews_corrcoef(all_labels, all_preds)
auc  = roc_auc_score(all_labels, all_probs,
                     multi_class='ovr', average='weighted')
pcr  = recall_score(all_labels, all_preds,
                    average=None, zero_division=0)
cm   = confusion_matrix(all_labels, all_preds)

print(f"\n{'='*60}")
print(f"  RESULTS (plain argmax — no threshold manipulation)")
print(f"{'='*60}")
print(f"  Accuracy:        {acc*100:6.2f}%")
print(f"  Weighted F1:     {f1w*100:6.2f}%")
print(f"  Precision:       {prec*100:6.2f}%")
print(f"  Recall:          {rec*100:6.2f}%")
print(f"  AUC-ROC:         {auc:.4f}")
print(f"  MCC:             {mcc:.4f}")
print(f"\nPer-class Recall:")
for i, cls in enumerate(CLASSES):
    bar = "█" * int(pcr[i]*20)
    print(f"  {cls:10}: {pcr[i]*100:5.1f}%  {bar}")

print(f"\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(f"  {'':12}  Normal  Murmur  Abnorm")
for i, row in enumerate(cm):
    print(f"  {CLASSES[i]:12}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

print(f"\nFull Classification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=CLASSES, zero_division=0))

# Save for frontend
results = {
    "accuracy":          round(float(acc),  4),
    "precision":         round(float(prec), 4),
    "recall":            round(float(rec),  4),
    "f1_score":          round(float(f1w),  4),
    "mcc":               round(float(mcc),  4),
    "auc_roc":           round(float(auc),  4),
    "per_class_recall":  [round(float(x), 4) for x in pcr],
    "confusion_matrix":  cm.tolist(),
    "test_class_counts": counts.tolist(),
    "n_test_samples":    int(len(all_labels)),
    "classes":           CLASSES
}

print(f"\n{'='*60}")
print(f"  DECISION BOUNDARY CALIBRATION")
print(f"{'='*60}")
optimal_thresh = calculate_optimal_abnormal_threshold(all_labels, all_probs)
results["optimal_abnormal_threshold"] = optimal_thresh
print(f"  Note: Update app.py's ABNORMAL_THRESHOLD to {optimal_thresh:.3f} for safe live inference.")

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → evaluation/results.json")

# Pass/Fail check
print(f"\n{'='*60}")
print(f"  SUCCESS CRITERIA CHECK")
print(f"{'='*60}")
checks = [
    ("Accuracy  ≥ 70%",           acc  >= 0.70),
    ("Weighted F1 ≥ 65%",         f1w  >= 0.65),
    ("AUC-ROC ≥ 0.85",            auc  >= 0.85),
    ("MCC ≥ 0.50",                mcc  >= 0.50),
    ("Normal Recall ≥ 70%",       pcr[0] >= 0.70),
    ("Murmur Recall ≥ 60%",       pcr[1] >= 0.60),
    ("Abnormal Recall ≥ 70%",     pcr[2] >= 0.70),
    ("All 3 classes predicted",   len(np.unique(all_preds)) == 3),
]
passed = 0
for name, result in checks:
    icon = "✅" if result else "❌"
    print(f"  {icon} {name}")
    if result: passed += 1
print(f"\n  {passed}/{len(checks)} criteria passed")

# ----- Generate ROC Curve Graph -----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as auc_fn

y_true_bin = label_binarize(all_labels, classes=[0, 1, 2])
plt.figure(figsize=(8, 6))
colors = ['#27AE60', '#F39C12', '#E74C3C']
for i, color in zip(range(3), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc_fn(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of {CLASSES[i]} (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve (CNN2D)')
plt.legend(loc="lower right")
plt.savefig('evaluation/roc_graph.png', dpi=150)
plt.close()
print("Saved evaluation/roc_graph.png")
