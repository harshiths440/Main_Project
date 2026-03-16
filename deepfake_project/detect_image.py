import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Face Detector
# -----------------------------
mtcnn = MTCNN(keep_all=True, device=device)

# -----------------------------
# Deepfake Model (EfficientNet)
# -----------------------------
model = EfficientNet.from_pretrained('efficientnet-b0')

# Replace final layer for binary classification
model._fc = nn.Linear(model._fc.in_features, 1)

model = model.to(device)
model.eval()

# -----------------------------
# Image Transform
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
# Load Image
# -----------------------------
image_path = "test_images/test.png"
image = Image.open(image_path).convert("RGB")

# -----------------------------
# Detect Faces
# -----------------------------
boxes, _ = mtcnn.detect(image)

if boxes is not None:
    print(f"Detected {len(boxes)} face(s)")
    
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        face = image.crop((x1, y1, x2, y2))
        
        face_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probability = torch.sigmoid(output)
            confidence = probability.item()

        label = "FAKE" if confidence > 0.5 else "REAL"
        
        print(f"Prediction: {label} (confidence: {confidence:.4f})")

else:
    print("No face detected")