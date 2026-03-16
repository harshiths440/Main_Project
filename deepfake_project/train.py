from models.model import DeepfakeDetector
import torch

model = DeepfakeDetector()
dummy = torch.randn(1, 3, 224, 224)
output = model(dummy)

print(output.shape)