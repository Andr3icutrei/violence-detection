import torch
import torch.nn as nn
from typing import Optional
from pytorchvideo.models.hub import mvit_base_16x4


class MViTViolence(nn.Module):
    """Wrap the pytorchvideo MViT-B (16x4) backbone for violence classification.

    MViT is a single-pathway multiscale video transformer: the forward pass
    takes ONE tensor of shape [B, C, T, H, W] (T = 16 frames), not a
    [slow, fast] pair. The whole network is trainable (no frozen backbone).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_p: float = 0.5) -> None:
        """Initialize the object and its runtime state."""
        super(MViTViolence, self).__init__()
        # Base ("B") MViT from pytorchvideo, pretrained on Kinetics-400.
        self.backbone: nn.Module = mvit_base_16x4(pretrained=pretrained)

        # Locate the final Linear in the classification head to size the new head.
        head: nn.Module = self.backbone.head
        in_features: Optional[int] = None
        if hasattr(head, 'proj') and isinstance(head.proj, nn.Linear):
            in_features = head.proj.in_features
        else:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
        if in_features is None:
            in_features = 768

        # Replace the K400 head (400 classes) with a dropout + linear head.
        head.proj = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass. ``x`` is a single tensor of shape [B, C, T, H, W]."""
        return self.backbone(x)


# Backward-compatible alias so modules importing the old name keep working.
SlowFastViolence = MViTViolence