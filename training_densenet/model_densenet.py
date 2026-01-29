import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], 1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer3D(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet3D(nn.Module):
    def __init__(self, num_classes=2, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, dropout_p=0.5):
        super(DenseNet3D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, num_init_features, kernel_size=(7, 7, 7),
                      stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionBlock3D(num_features, num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(num_features, num_classes)

        self.gradients = None
        self.activations = None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, return_cam=False):
        x = self.features(x)

        if return_cam:
            self.activations = x
            if x.requires_grad:
                x.register_hook(self.save_gradient)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

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