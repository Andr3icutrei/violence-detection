from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18


class R3D18Violence(nn.Module):
    """Wrap an R3D-18 video backbone for binary violence classification and Grad-CAM."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_layers: Sequence[str] | None = None,
        dropout_p: float = 0.5,
    ) -> None:
        """Create the classifier head and optionally freeze selected backbone layers."""
        super().__init__()

        weights: R3D_18_Weights | None = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone: nn.Module = r3d_18(weights=weights)

        in_features: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        if freeze_layers:
            self._freeze_layers(freeze_layers)

        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

    def _freeze_layers(self, layer_names: Sequence[str]) -> None:
        """Disable gradient updates for backbone layers whose names match the provided prefixes."""
        for name, parameter in self.backbone.named_parameters():
            for layer_name in layer_names:
                if name.startswith(layer_name):
                    parameter.requires_grad = False
                    break

    def unfreeze_all(self) -> None:
        """Enable gradient updates for every backbone parameter."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def save_gradient(self, gradient: torch.Tensor) -> None:
        """Store the latest activation gradient for Grad-CAM generation."""
        self.gradients = gradient

    def forward(self, inputs: torch.Tensor, return_cam: bool = False) -> torch.Tensor:
        """Run a forward pass and optionally store activations for Grad-CAM."""
        features: torch.Tensor = self.backbone.stem(inputs)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        if return_cam:
            self.activations = features
            if features.requires_grad:
                features.register_hook(self.save_gradient)

        pooled_features: torch.Tensor = self.backbone.avgpool(features)
        flattened_features: torch.Tensor = torch.flatten(pooled_features, 1)
        logits: torch.Tensor = self.backbone.fc(flattened_features)

        return logits

    def get_cam(self, target_class: int) -> torch.Tensor | None:
        """Compute normalized 3D Grad-CAM maps from stored activations and gradients."""
        if self.gradients is None or self.activations is None:
            return None

        gradients: torch.Tensor = self.gradients.detach()
        activations: torch.Tensor = self.activations.detach()
        weights: torch.Tensor = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        cam: torch.Tensor = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze(1)

        batch_size: int = cam.size(0)
        normalized_cams: list[torch.Tensor] = []
        for index in range(batch_size):
            single_cam: torch.Tensor = cam[index]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            normalized_cams.append(single_cam)

        return torch.stack(normalized_cams)

    def get_spatial_cam(self, target_class: int) -> torch.Tensor | None:
        """Collapse a 3D Grad-CAM map over time to produce a normalized 2D spatial map."""
        cam_3d: torch.Tensor | None = self.get_cam(target_class)
        if cam_3d is None:
            return None

        cam_2d: torch.Tensor = torch.sum(cam_3d, dim=1)
        batch_size: int = cam_2d.size(0)
        normalized_cams: list[torch.Tensor] = []

        for index in range(batch_size):
            single_cam: torch.Tensor = cam_2d[index]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            normalized_cams.append(single_cam)

        return torch.stack(normalized_cams)
