import os
import torch
import numpy as np
from src.training.data_loader import get_loaders
from src.training.models.cnn2d import CNN2D
from src.training.loss_functions import WeightedFocalLoss, get_class_weights

# ── Config ──────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS  = 150
PATIENCE    = 20
SAVE_PATH   = "models/best_model.pt"
os.makedirs("saved_models", exist_ok=True)

print(f"\n{'='*60}")
print(f"  CARDIOSONIC TRAINING")
print(f"{'='*60}")
print(f"  Device: {DEVICE}")
print(f"  Max epochs: {MAX_EPOCHS} | Early stop patience: {PATIENCE}")
print(f"{'='*60}\n")

# ── Data ────────────────────────────────────────────────────
train_loader, val_loader, test_loader, train_labels = get_loaders()

# ── Model ───────────────────────────────────────────────────
model  = CNN2D(num_classes=3).to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: CNN2D | Trainable parameters: {params:,}")

# ── Loss / Optimizer / Scheduler ────────────────────────────
criterion = WeightedFocalLoss(gamma=2.0, alpha=[1.0, 1.0, 1.0])
optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=70, eta_min=1e-6
)

# ── Training Loop ────────────────────────────────────────────
best_val_loss    = float('inf')
patience_counter = 0
history          = []

print(f"\n{'─'*80}")
print(f"{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} "
      f"{'ValLoss':>10} {'ValAcc':>10} {'LR':>10} {'Status':>12}")
print(f"{'─'*80}")

for epoch in range(1, MAX_EPOCHS + 1):

    # ── Train ──────────────────────────────
    model.train()
    t_loss = t_correct = t_total = 0
    for mel, mfcc, labels in train_loader:
        mel, mfcc, labels = (mel.to(DEVICE),
                             mfcc.to(DEVICE),
                             labels.to(DEVICE))
        optimizer.zero_grad()
        out  = model(mel, mfcc)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (out.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)

    # ── Validate ────────────────────────────
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for mel, mfcc, labels in val_loader:
            mel, mfcc, labels = (mel.to(DEVICE),
                                 mfcc.to(DEVICE),
                                 labels.to(DEVICE))
            out    = model(mel, mfcc)
            loss   = criterion(out, labels)
            v_loss    += loss.item()
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total   += labels.size(0)

    t_loss /= len(train_loader)
    v_loss /= len(val_loader)
    t_acc   = t_correct / t_total
    v_acc   = v_correct / v_total
    lr_now  = scheduler.get_last_lr()[0]
    scheduler.step()

    history.append({"epoch": epoch,
                    "train_loss": t_loss, "val_loss": v_loss,
                    "train_acc":  t_acc,  "val_acc":  v_acc})

    # ── Checkpoint ──────────────────────────
    if v_loss < best_val_loss:
        best_val_loss    = v_loss
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        status = "✓ SAVED"
    else:
        patience_counter += 1
        status = f"patience {patience_counter}/{PATIENCE}"

    # ── Print every epoch ───────────────────
    print(f"{epoch:>6d} {t_loss:>10.4f} {t_acc:>10.4f} "
          f"{v_loss:>10.4f} {v_acc:>10.4f} {lr_now:>10.6f} {status:>12}")

    if patience_counter >= PATIENCE:
        print(f"\n{'─'*80}")
        print(f"Early stopping triggered at epoch {epoch}")
        print(f"No val_loss improvement for {PATIENCE} consecutive epochs")
        break

print(f"{'─'*80}")
print(f"\nTRAINING COMPLETE")
print(f"  Best val_loss: {best_val_loss:.4f}")
print(f"  Model saved:   {SAVE_PATH}")
print(f"  Total epochs:  {epoch}")
print(f"\nNow run: python evaluation/evaluate.py")
