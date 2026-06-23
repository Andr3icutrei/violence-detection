from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class X3DViolence(nn.Module):
    """Wraps a pretrained X3D backbone with a binary classification head and Grad-CAM support."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_p: float = 0.5,
        x3d_version: str = "m",
    ) -> None:
        """Creates the requested X3D variant and replaces its projection layer."""

        super().__init__()

        self.x3d_version: str = x3d_version.lower()
        self.backbone: nn.Module

        try:
            if self.x3d_version == "xs":
                from pytorchvideo.models.hub import x3d_xs

                self.backbone = x3d_xs(pretrained=pretrained)
            elif self.x3d_version == "s":
                from pytorchvideo.models.hub import x3d_s

                self.backbone = x3d_s(pretrained=pretrained)
            elif self.x3d_version == "m":
                from pytorchvideo.models.hub import x3d_m

                self.backbone = x3d_m(pretrained=pretrained)
            elif self.x3d_version == "l":
                from pytorchvideo.models.hub import x3d_l

                self.backbone = x3d_l(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown X3D version: {x3d_version}. Use 'xs', 's', 'm', or 'l'.")

            projection_module: nn.Module = self.backbone.blocks[-1].proj
            input_features: Optional[int] = None

            if isinstance(projection_module, nn.Linear):
                input_features = projection_module.in_features
            elif isinstance(projection_module, nn.Sequential):
                for module in projection_module:
                    if isinstance(module, nn.Linear):
                        input_features = module.in_features
                        break

            if input_features is None:
                input_features = 2048

            self.backbone.blocks[-1].proj = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(input_features, num_classes),
            )

        except ImportError as exc:
            raise ImportError("pytorchvideo is not installed. Install it with: pip install pytorchvideo.") from exc

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def save_gradient(self, grad: torch.Tensor) -> None:
        """Stores gradients from the hooked feature map for Grad-CAM."""

        self.gradients = grad

    def forward(self, x: torch.Tensor, return_cam: bool = False) -> torch.Tensor:
        """Runs a forward pass and optionally registers a hook for Grad-CAM features."""

        if return_cam:
            for name, module in self.backbone.named_modules():
                if "blocks.4" in name and "branch1_conv" in name:
                    def hook(
                        hook_module: nn.Module,
                        hook_input: tuple[torch.Tensor, ...],
                        hook_output: torch.Tensor,
                    ) -> None:
                        """Captures feature activations and their gradients."""

                        self.activations = hook_output
                        if hook_output.requires_grad:
                            hook_output.register_hook(self.save_gradient)

                    module.register_forward_hook(hook)

        output: torch.Tensor = self.backbone(x)
        return output

    def get_cam(self, target_class: int) -> Optional[torch.Tensor]:
        """Computes normalized spatiotemporal Grad-CAM maps for the last hooked feature map."""

        if self.gradients is None or self.activations is None:
            return None

        gradients: torch.Tensor = self.gradients.detach()
        activations: torch.Tensor = self.activations.detach()
        weights: torch.Tensor = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        cam: torch.Tensor = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(1)

        batch_size: int = cam.size(0)
        cams: list[torch.Tensor] = []

        for index in range(batch_size):
            single_cam: torch.Tensor = cam[index]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            cams.append(single_cam)

        stacked_cams: torch.Tensor = torch.stack(cams)
        return stacked_cams

    def get_spatial_cam(self, target_class: int) -> Optional[torch.Tensor]:
        """Reduces spatiotemporal Grad-CAM maps into normalized spatial heatmaps."""

        cam_3d: Optional[torch.Tensor] = self.get_cam(target_class)
        if cam_3d is None:
            return None

        cam_2d: torch.Tensor = torch.sum(cam_3d, dim=1)
        batch_size: int = cam_2d.size(0)
        cams: list[torch.Tensor] = []

        for index in range(batch_size):
            single_cam: torch.Tensor = cam_2d[index]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            cams.append(single_cam)

        stacked_cams: torch.Tensor = torch.stack(cams)
        return stacked_cams
