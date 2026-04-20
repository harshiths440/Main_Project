import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.model import DeepfakeDetector

# -----------------------------
# Device (CPU or M2 GPU)
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset
# -----------------------------
dataset = datasets.ImageFolder(root="data", transform=transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# -----------------------------
# Model
# -----------------------------
model = DeepfakeDetector().to(device)

# Loss + Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "models/deepfake_model.pth")
print("Model saved!")