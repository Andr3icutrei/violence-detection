import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowFastViolence(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.5, alpha=4):
        super(SlowFastViolence, self).__init__()

        self.alpha = alpha

        try:
            from pytorchvideo.models.hub import slowfast_r50

            if pretrained:
                self.backbone = slowfast_r50(pretrained=True)
            else:
                self.backbone = slowfast_r50(pretrained=False)

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

        except ImportError:
            raise ImportError(
                "pytorchvideo not installed. Install with: pip install pytorchvideo"
            )

        self.gradients_slow = None
        self.gradients_fast = None
        self.activations_slow = None
        self.activations_fast = None

    def save_gradient_slow(self, grad):
        self.gradients_slow = grad

    def save_gradient_fast(self, grad):
        self.gradients_fast = grad

    def forward(self, x, return_cam=False):
        if not isinstance(x, list):
            raise ValueError("Input must be a list [slow_pathway, fast_pathway]")

        slow_pathway, fast_pathway = x

        if return_cam:
            for name, module in self.backbone.named_modules():
                if 'blocks.5.res_blocks.2.branch1_conv' in name:
                    def hook_slow(module, input, output):
                        self.activations_slow = output
                        if output.requires_grad:
                            output.register_hook(self.save_gradient_slow)

                    module.register_forward_hook(hook_slow)

                if 'blocks.4.res_blocks.2.branch1_conv' in name:
                    def hook_fast(module, input, output):
                        self.activations_fast = output
                        if output.requires_grad:
                            output.register_hook(self.save_gradient_fast)

                    module.register_forward_hook(hook_fast)

        output = self.backbone([slow_pathway, fast_pathway])

        return output

    def get_cam(self, target_class, pathway='slow'):
        if pathway == 'slow':
            gradients = self.gradients_slow
            activations = self.activations_slow
        else:
            gradients = self.gradients_fast
            activations = self.activations_fast

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