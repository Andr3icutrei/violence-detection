import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18Violence(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_layers=None, dropout_p=0.5):
        super(R3D18Violence, self).__init__()

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

        if freeze_layers:
            self._freeze_layers(freeze_layers)

        self.gradients = None
        self.activations = None

    def _freeze_layers(self, layer_names):
        for name, param in self.backbone.named_parameters():
            for layer_name in layer_names:
                if name.startswith(layer_name):
                    param.requires_grad = False
                    break

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, return_cam=False):
        x = self.backbone.stem(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if return_cam:
            self.activations = x
            if x.requires_grad:
                x.register_hook(self.save_gradient)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x

    def get_cam(self, target_class):
        if self.gradients is None or self.activations is None:
            return None

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze(1)

        batch_size = cam.size(0)
        cams = []
        for i in range(batch_size):
            single_cam = cam[i]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            cams.append(single_cam)

        return torch.stack(cams)

    def get_spatial_cam(self, target_class):
        cam_3d = self.get_cam(target_class)
        if cam_3d is None:
            return None

        cam_2d = torch.sum(cam_3d, dim=1)

        batch_size = cam_2d.size(0)
        cams = []
        for i in range(batch_size):
            single_cam = cam_2d[i]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            cams.append(single_cam)

        return torch.stack(cams)