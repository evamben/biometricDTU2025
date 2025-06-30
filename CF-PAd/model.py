import torch
import torch.nn as nn
from torchvision import models

class SimpleModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleModel, self).__init__()

        # Seleccionar pesos preentrenados de forma compatible con versiones modernas
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        # Cargar el modelo con o sin pesos
        self.backbone = models.resnet18(weights=weights)

        # Reemplazar la última capa totalmente conectada
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
