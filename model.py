import torch.nn as nn
from timm import create_model


class MultiLabelNet(nn.Module):
    """Any timm backbone + one linear layer producing 14 logits."""

    def __init__(self, backbone, num_labels=14, dropout=0.0):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=True, num_classes=0)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.backbone.num_features, num_labels)

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, tuple):  # distilled DeiT can return (cls, dist) tokens
            f = (f[0] + f[1]) / 2
        return self.head(self.drop(f))
