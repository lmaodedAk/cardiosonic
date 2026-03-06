import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

train_loss = [0.4900, 0.4420, 0.4284, 0.3987, 0.3950, 0.3787, 0.3827, 0.3856, 0.4058, 0.3638, 0.3446, 0.3354, 0.3513, 0.3469, 0.3456, 0.3230, 0.3035, 0.2853, 0.3247, 0.3167, 0.2973, 0.2992, 0.2892, 0.2722, 0.2872, 0.2683, 0.2333, 0.2729, 0.2776, 0.2608, 0.2525, 0.2303, 0.2302, 0.2378, 0.2374, 0.2123, 0.2167, 0.2044, 0.2017, 0.1675, 0.1810, 0.2080, 0.1605, 0.1701, 0.2154, 0.2025, 0.2068, 0.1908]
val_loss = [0.4686, 0.4362, 0.4506, 0.4618, 0.4831, 0.5077, 0.4542, 0.4733, 0.4388, 0.4943, 0.4720, 0.4431, 0.4271, 0.4481, 0.4357, 0.3999, 0.4055, 0.3722, 0.3798, 0.3563, 0.3512, 0.4284, 0.4284, 0.3923, 0.3464, 0.4298, 0.4616, 0.3343, 0.3465, 0.4406, 0.3896, 0.3860, 0.4304, 0.4122, 0.3572, 0.3935, 0.3771, 0.4331, 0.4137, 0.4250, 0.4272, 0.4026, 0.4225, 0.4103, 0.4018, 0.4163, 0.5038, 0.4532]

os.makedirs('evaluation', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss (Focal + Imbalance)', color='#3498DB', linewidth=2)
plt.plot(val_loss, label='Val Loss', color='#E74C3C', linewidth=2, linestyle='--')
plt.scatter(27, 0.3343, color='red', zorder=5, label='Best Model (Epoch 28)') # 0-indexed is 27
plt.title('Training Loss Curve (1.1M CNN2D)')
plt.xlabel('Epochs')
plt.ylabel('Weighted Focal Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('evaluation/loss_graph.png', dpi=150)
print('Saved evaluation/loss_graph.png')
