import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HybridViTCNNMLP import HybridViTCNNMLP
from models.cnn import CustomCNN
from models.mobilenetv3 import MobileNetV3
from models.denseNet121 import DenseNet121Medical
from models.HybridSwinDenseNetMLP import HybridSwinDenseNetMLP
from models.efficientnet_medical import EfficientNetB3Medical, EfficientNetB4Medical
from models.ShuffleNetV2Student import ShuffleNetV2Student


def get_model(model_name: str, num_classes: int, pretrained: bool = False, dropout_rate: float = 0.5):
    if model_name == 'HVTCMLP':
        model = HybridViTCNNMLP(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbones=False
        )
    elif model_name == 'cnn':
        model = CustomCNN(num_classes=num_classes)
    elif model_name == 'shufflenetv2':
        model = ShuffleNetV2Student(
            num_classes=num_classes,
            in_channels=1,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_name == 'mobilenetv3':
        model = MobileNetV3(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_name == 'densenet121':
        model = DenseNet121Medical(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_name == 'HSwinDNMLP':
        model = HybridSwinDenseNetMLP(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout_rate,
            freeze_backbones=False
        )
    elif model_name == 'effnetb3':
        model = EfficientNetB3Medical(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_name == 'effnetb4':
        model = EfficientNetB4Medical(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )


    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    return model