import torch
import torch.nn as nn
import torch.nn.functional as F

class X3DViolence(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.5, x3d_version='m', input_channels=2):
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
                raise ValueError(f"Unknown X3D version: {x3d_version}")

            if input_channels != 3:
                old_conv = self.backbone.blocks[0].conv.conv_t
                new_conv = nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=(old_conv.bias is not None)
                )
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)
                self.backbone.blocks[0].conv.conv_t = new_conv

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
            raise ImportError("pytorchvideo not installed")

    def forward(self, x):
        return self.backbone(x)