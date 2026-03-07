import torch
checkpoint = torch.load('models/best_model.pt', map_location='cpu')
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print("Keys:", checkpoint.keys())
else:
    print("Full model, class:", checkpoint.__class__.__name__)
