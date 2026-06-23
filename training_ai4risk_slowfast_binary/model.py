import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pytorchvideo.models.hub import slowfast_r50

class SlowFastViolence(nn.Module):
    """Wrap the SlowFast backbone for violence classification and Grad-CAM extraction."""

    def __init__(self, num_classes: int=2, pretrained: bool=True, dropout_p: float=0.5, slowfast_alpha: int=4, slowfast_beta: float=0.125) -> None:
        """Initialize the object and its runtime state."""
        super(SlowFastViolence, self).__init__()
        self.slowfast_alpha: int = slowfast_alpha
        self.slowfast_beta: float = slowfast_beta
        self.backbone: nn.Module = slowfast_r50(pretrained=pretrained)
        proj_module: nn.Module = self.backbone.blocks[-1].proj
        in_features: Optional[int] = None
        if isinstance(proj_module, nn.Linear):
            in_features = proj_module.in_features
        elif isinstance(proj_module, nn.Sequential):
            for module in proj_module:
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    break
        if in_features is None:
            in_features = 2304
        self.backbone.blocks[-1].proj = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))

        self.slow_gradients: Optional[torch.Tensor] = None
        self.slow_activations: Optional[torch.Tensor] = None
        self.fast_gradients: Optional[torch.Tensor] = None
        self.fast_activations: Optional[torch.Tensor] = None

    def save_slow_gradient(self, grad: torch.Tensor) -> None:
        """Store gradients from the slow pathway for Grad-CAM."""
        self.slow_gradients = grad

    def save_fast_gradient(self, grad: torch.Tensor) -> None:
        """Store gradients from the fast pathway for Grad-CAM."""
        self.fast_gradients = grad

    def forward(self, x: List[torch.Tensor], return_cam: bool=False) -> torch.Tensor:
        """Run a forward pass through the wrapped SlowFast model."""
        if return_cam:
            if not hasattr(self, '_hooks_registered'):
                self._hooks_registered: bool = True
                name: str
                module: nn.Module
                for name, module in self.backbone.named_modules():
                    if 'blocks.4.multipathway_blocks.0.res_blocks.2.branch2.conv_c' in name:

                        def slow_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                            """Execute the slow_hook operation."""
                            self.slow_activations = output
                            if output.requires_grad:
                                output.register_hook(self.save_slow_gradient)
                        module.register_forward_hook(slow_hook)
                    if 'blocks.4.multipathway_blocks.1.res_blocks.2.branch2.conv_c' in name:

                        def fast_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                            """Execute the fast_hook operation."""
                            self.fast_activations = output
                            if output.requires_grad:
                                output.register_hook(self.save_fast_gradient)
                        module.register_forward_hook(fast_hook)
        output: torch.Tensor = self.backbone(x)
        return output

    def get_cam(self, target_class: int, pathway: str='slow') -> Optional[torch.Tensor]:
        """Compute a temporal-spatial Grad-CAM map for one pathway."""
        gradients: Optional[torch.Tensor]
        activations: Optional[torch.Tensor]
        if pathway == 'slow':
            gradients = self.slow_gradients
            activations = self.slow_activations
        else:
            gradients = self.fast_gradients
            activations = self.fast_activations
        if gradients is None or activations is None:
            return None
        gradients = gradients.detach()
        activations = activations.detach()
        weights: torch.Tensor = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        cam: torch.Tensor = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(1)
        batch_size: int = cam.size(0)
        cams: List[torch.Tensor] = []
        for i in range(batch_size):
            single_cam: torch.Tensor = cam[i]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-08)
            cams.append(single_cam)
        return torch.stack(cams)

    def get_spatial_cam(self, target_class: int, pathway: str='slow') -> Optional[torch.Tensor]:
        """Collapse the pathway Grad-CAM map into a spatial heatmap."""
        cam_3d: Optional[torch.Tensor] = self.get_cam(target_class, pathway)
        if cam_3d is None:
            return None
        cam_2d: torch.Tensor = torch.sum(cam_3d, dim=1)
        batch_size: int = cam_2d.size(0)
        cams: List[torch.Tensor] = []
        for i in range(batch_size):
            single_cam: torch.Tensor = cam_2d[i]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-08)
            cams.append(single_cam)
        return torch.stack(cams)

    def get_fused_spatial_cam(self, target_class: int, slow_weight: float=0.6, fast_weight: float=0.4) -> Optional[torch.Tensor]:
        """Fuse slow and fast pathway spatial Grad-CAM heatmaps."""
        slow_cam: Optional[torch.Tensor] = self.get_spatial_cam(target_class, pathway='slow')
        fast_cam: Optional[torch.Tensor] = self.get_spatial_cam(target_class, pathway='fast')
        if slow_cam is None and fast_cam is None:
            return None
        elif slow_cam is None:
            return fast_cam
        elif fast_cam is None:
            return slow_cam
        if slow_cam.shape != fast_cam.shape:
            fast_cam = F.interpolate(fast_cam.unsqueeze(1), size=slow_cam.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        fused_cam: torch.Tensor = slow_weight * slow_cam + fast_weight * fast_cam
        batch_size: int = fused_cam.size(0)
        cams: List[torch.Tensor] = []
        for i in range(batch_size):
            single_cam: torch.Tensor = fused_cam[i]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-08)
            cams.append(single_cam)
        return torch.stack(cams)