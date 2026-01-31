import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18Student(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.5):
        super(R3D18Student, self).__init__()

        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)