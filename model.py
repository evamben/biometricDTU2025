# import torch
# import torch.nn as nn
# from mixstyle import MixStyle  # Asegúrate de que mixstyle.py está disponible


# class ResNetBackbone(nn.Module):
#     def __init__(self, base_model, ms_layers=None, ms_p=0.5, ms_a=0.1):
#         super().__init__()
#         self.base_model = base_model
#         self.mixstyle = MixStyle(p=ms_p, alpha=ms_a) if ms_layers else None
#         self.ms_layers = ms_layers or []

#     def forward(self, x, labels=None):
#         x = self.base_model.conv1(x)
#         x = self.base_model.bn1(x)
#         x = self.base_model.relu(x)
#         x = self.base_model.maxpool(x)

#         x = self.base_model.layer1(x)
#         if "layer1" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)

#         x = self.base_model.layer2(x)
#         if "layer2" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)

#         x = self.base_model.layer3(x)
#         if "layer3" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)

#         x = self.base_model.layer4(x)
#         if "layer4" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)

#         return x


# class ComplexModel(nn.Module):
#     def __init__(self, base_model, num_classes=2, ms_layers=["layer1", "layer2"]):
#         super().__init__()
#         self.backbone = ResNetBackbone(base_model, ms_layers=ms_layers)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(p=0.5)
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x, labels=None, cf_ops=None):
#         features = self.backbone(x, labels)
#         features = self.avgpool(features)
#         features = torch.flatten(features, 1)

#         out = self.classifier(features)

#         if not self.training or cf_ops is None:
#             return out

#         # Aplica contrafactual con dropout si está activado
#         if "dropout" in cf_ops:
#             cf_features = self.dropout(features)
#             cf_out = self.classifier(cf_features)
#             return out, cf_out

#         # Otros contrafactuales podrían ir aquí (ej. replace, shuffle, etc.)
#         return out
# import torch
# import torch.nn as nn
# from mixstyle import MixStyle
# from torch.nn import functional as F

# class ChannelAttention(nn.Module):
#     """Módulo de atención por canal (adaptado de CBAM)"""
#     def __init__(self, in_channels, reduction_ratio=8):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         avg_out = self.fc(self.avg_pool(x).view(b, c))
#         max_out = self.fc(self.max_pool(x).view(b, c))
#         out = avg_out + max_out
#         return self.sigmoid(out).view(b, c, 1, 1) * x

# class ResNetBackbone(nn.Module):
#     def __init__(self, base_model, ms_layers=None, ms_p=0.5, ms_a=0.1, use_attention=False):
#         super().__init__()
#         self.base_model = base_model
#         self.mixstyle = MixStyle(p=ms_p, alpha=ms_a) if ms_layers else None
#         self.ms_layers = ms_layers or []
#         self.use_attention = use_attention
        
#         if use_attention:
#             # Añadir módulos de atención después de cada capa residual
#             self.ca1 = ChannelAttention(64)
#             self.ca2 = ChannelAttention(128)
#             self.ca3 = ChannelAttention(256)
#             self.ca4 = ChannelAttention(512)

#     def forward(self, x, labels=None):
#         x = self.base_model.conv1(x)
#         x = self.base_model.bn1(x)
#         x = self.base_model.relu(x)
#         x = self.base_model.maxpool(x)

#         x = self.base_model.layer1(x)
#         if "layer1" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)
#         if self.use_attention:
#             x = self.ca1(x)

#         x = self.base_model.layer2(x)
#         if "layer2" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)
#         if self.use_attention:
#             x = self.ca2(x)

#         x = self.base_model.layer3(x)
#         if "layer3" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)
#         if self.use_attention:
#             x = self.ca3(x)

#         x = self.base_model.layer4(x)
#         if "layer4" in self.ms_layers and self.mixstyle is not None:
#             x = self.mixstyle(x, labels)
#         if self.use_attention:
#             x = self.ca4(x)

#         return x

# class AuxiliaryBranch(nn.Module):
#     """Rama auxiliar para aprendizaje multi-task"""
#     def __init__(self, in_features, out_features=1):
#         super().__init__()
#         self.branch = nn.Sequential(
#             nn.Linear(in_features, in_features // 2),
#             nn.BatchNorm1d(in_features // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features // 2, out_features)
#         )
    
#     def forward(self, x):
#         return self.branch(x)

# class ComplexModel(nn.Module):
#     def __init__(self, base_model, num_classes=2, ms_layers=["layer1", "layer2"], 
#                  use_attention=True, use_auxiliary=True):
#         super().__init__()
#         self.backbone = ResNetBackbone(base_model, ms_layers=ms_layers, use_attention=use_attention)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)  # Pooling adicional
#         self.dropout = nn.Dropout(p=0.5)
        
#         # Capas de proyección para contrafactuales
#         self.projection = nn.Sequential(
#             nn.Linear(1024, 512),  # Concatenamos avg y max pool
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True)
#         )
        
#         self.classifier = nn.Linear(512, num_classes)
        
#         # Rama auxiliar para regularización
#         self.use_auxiliary = use_auxiliary
#         if use_auxiliary:
#             self.auxiliary = AuxiliaryBranch(512)
        
#         # Inicialización de pesos
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward_features(self, x, labels=None):
#         features = self.backbone(x, labels)
        
#         # Poolings múltiples para capturar diferentes características
#         avg_features = self.avgpool(features)
#         max_features = self.maxpool(features)
        
#         avg_features = torch.flatten(avg_features, 1)
#         max_features = torch.flatten(max_features, 1)
        
#         # Concatenamos ambos poolings
#         combined_features = torch.cat([avg_features, max_features], dim=1)
#         projected_features = self.projection(combined_features)
        
#         return projected_features

#     def forward(self, x, labels=None, cf_ops=None):
#         features = self.forward_features(x, labels)
        
#         # Clasificación principal
#         out = self.classifier(features)
        
#         if not self.training:
#             return out
        
#         outputs = {'main': out}
        
#         # Rama auxiliar si está activa
#         if self.use_auxiliary:
#             aux_out = self.auxiliary(features.detach())  # Detach para no afectar los gradientes principales
#             outputs['auxiliary'] = aux_out
        
#         # Operaciones contrafactuales durante el entrenamiento
#         if cf_ops is not None:
#             cf_outputs = {}
            
#             if "dropout" in cf_ops:
#                 cf_features = self.dropout(features)
#                 cf_outputs['dropout'] = self.classifier(cf_features)
            
#             if "shuffle" in cf_ops:
#                 # Shuffle de características entre muestras del batch
#                 shuffled_idx = torch.randperm(features.size(0))
#                 cf_shuffle = features[shuffled_idx]
#                 cf_outputs['shuffle'] = self.classifier(cf_shuffle)
            
#             if "noise" in cf_ops:
#                 # Añadir ruido gaussiano
#                 noise = torch.randn_like(features) * 0.1
#                 cf_noise = features + noise
#                 cf_outputs['noise'] = self.classifier(cf_noise)
            
#             outputs['cf'] = cf_outputs
        
#         return outputs if self.training else out
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 3, 224, 224] -> [B, 16, 224, 224]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 112, 112]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> [B, 32, 112, 112]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 56, 56]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> [B, 64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 64, 28, 28]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                      # -> [B, 64*28*28]
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x