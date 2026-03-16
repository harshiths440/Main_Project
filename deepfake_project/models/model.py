import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()

        # Load EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Get number of features in classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with binary output
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)