import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2Student(nn.Module):
    """
    ShuffleNetV2-1.0x student model for Federated Learning + KD
    on brain MRI (glioma, meningioma, pituitary, no_tumor).

    - Adapts first conv to 1-channel input
    - Keeps pretrained backbone (optional)
    - Adds an MLP classifier head similar to your MobileNetV3 design
    """
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        super(ShuffleNetV2Student, self).__init__()

        # 1) Load backbone (optionally with ImageNet weights)
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = shufflenet_v2_x1_0(weights=weights)

        # 2) Adapt the first conv layer for grayscale or custom channels
        if in_channels != 3:
            # In torchvision, conv1 is a Sequential [Conv2d, BN, ReLU]
            old_conv = self.backbone.conv1[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            if pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        # Average RGB weights -> single channel
                        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                    else:
                        # Repeat/trim if in_channels > 3 (fallback)
                        repeat = int((in_channels + 2) // 3)
                        w = old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
                        new_conv.weight[:] = w

            self.backbone.conv1[0] = new_conv

        # 3) Replace the built-in classifier (fc) with Identity to get feature vector
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 4) Custom MLP classifier head (mirroring your MobileNetV3 style)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes),
        )

        # Only initialize the classifier, keep pretrained backbone intact
        self._initialize_classifier_weights()

    def _initialize_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Features from backbone (global pooled)
        features = self.backbone(x)   # shape: [B, num_features]
        # Pass through MLP head
        output = self.classifier(features)
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return latent features before the MLP head.
        Useful if you ever want feature-level KD in addition to logits KD.
        """
        return self.backbone(x)
