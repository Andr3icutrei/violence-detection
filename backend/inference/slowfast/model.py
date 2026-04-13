import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowFastViolence(nn.Module):
    def __init__(self, num_classes=5, pretrained=False, dropout_p=0.5,
                 slowfast_alpha=4, slowfast_beta=0.125):
        super(SlowFastViolence, self).__init__()
        self.slowfast_alpha = slowfast_alpha
        self.slowfast_beta = slowfast_beta

        from pytorchvideo.models.hub import slowfast_r50
        self.backbone = slowfast_r50(pretrained=pretrained)

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
            in_features = 2304

        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

        self.slow_gradients = None
        self.slow_activations = None
        self.fast_gradients = None
        self.fast_activations = None

    def save_slow_gradient(self, grad):
        self.slow_gradients = grad

    def save_fast_gradient(self, grad):
        self.fast_gradients = grad

    def forward(self, x, return_cam=False):
        if return_cam:
            if not hasattr(self, '_hooks_registered'):
                self._hooks_registered = True
                for name, module in self.backbone.named_modules():
                    if 'blocks.4.multipathway_blocks.0.res_blocks.2.branch2.conv_c' in name:
                        def slow_hook(module, input, output):
                            self.slow_activations = output
                            if output.requires_grad:
                                output.register_hook(self.save_slow_gradient)
                        module.register_forward_hook(slow_hook)

                    if 'blocks.4.multipathway_blocks.1.res_blocks.2.branch2.conv_c' in name:
                        def fast_hook(module, input, output):
                            self.fast_activations = output
                            if output.requires_grad:
                                output.register_hook(self.save_fast_gradient)
                        module.register_forward_hook(fast_hook)

        return self.backbone(x)

    def get_cam(self, target_class, pathway='slow'):
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

    def get_spatial_cam(self, target_class, pathway='slow'):
        cam_3d = self.get_cam(target_class, pathway)
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

    def get_fused_spatial_cam(self, target_class, slow_weight=0.6, fast_weight=0.4):
        slow_cam = self.get_spatial_cam(target_class, pathway='slow')
        fast_cam = self.get_spatial_cam(target_class, pathway='fast')

        if slow_cam is None and fast_cam is None: return None
        elif slow_cam is None: return fast_cam
        elif fast_cam is None: return slow_cam

        if slow_cam.shape != fast_cam.shape:
            fast_cam = F.interpolate(
                fast_cam.unsqueeze(1),
                size=slow_cam.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        fused_cam = slow_weight * slow_cam + fast_weight * fast_cam

        batch_size = fused_cam.size(0)
        cams = []
        for i in range(batch_size):
            single_cam = fused_cam[i]
            single_cam = single_cam - single_cam.min()
            single_cam = single_cam / (single_cam.max() + 1e-8)
            cams.append(single_cam)

        return torch.stack(cams)