import torch
import torch.nn as nn
import torch.nn.functional as F


class X3DViolence(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout_p=0.5, x3d_version='m'):
        super(X3DViolence, self).__init__()

        self.x3d_version = x3d_version.lower()

        try:
            if self.x3d_version == 'xs':
                from pytorchvideo.models.hub import x3d_xs
                self.backbone = x3d_xs(pretrained=pretrained)
            elif self.x3d_version == 's':
                from pytorchvideo.models.hub import x3d_s
                self.backbone = x3d_s(pretrained=pretrained)
            elif self.x3d_version == 'm':
                from pytorchvideo.models.hub import x3d_m
                self.backbone = x3d_m(pretrained=pretrained)
            elif self.x3d_version == 'l':
                from pytorchvideo.models.hub import x3d_l
                self.backbone = x3d_l(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown X3D version: {x3d_version}. Use 'xs', 's', 'm', or 'l'")

            proj_module = self.backbone.blocks[-1].proj
            in_features = None

            if isinstance(proj_module, nn.Linear):
                in_features = proj_module.in_features
            elif isinstance(proj_module, nn.Sequential):
                for module in proj_module:
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        break

            if in_features is None:
                in_features = 2048

            self.backbone.blocks[-1].proj = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )

        except ImportError:
            raise ImportError(
                "pytorchvideo not installed. Install with: pip install pytorchvideo"
            )

        self.gradients = None
        self.activations = None
        self._hook_handles = []

    def save_gradient(self, grad):
        self.gradients = grad

    def _register_cam_hooks(self):
        for name, module in self.backbone.named_modules():
            if 'blocks.4' in name and 'branch1_conv' in name:
                def forward_hook(module, input, output):
                    self.activations = output
                    if output.requires_grad:
                        output.register_hook(self.save_gradient)

                handle = module.register_forward_hook(forward_hook)
                self._hook_handles.append(handle)

    def _remove_cam_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def forward(self, x, return_cam=False):
        if return_cam:
            self.gradients = None
            self.activations = None
            self._remove_cam_hooks()
            self._register_cam_hooks()

        output = self.backbone(x)

        if return_cam:
            self._remove_cam_hooks()

        return output

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