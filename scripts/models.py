import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadNet(nn.Module):
    def __init__(self, input_dim, trunk_hidden=128, trunk_layers=2,
                 reg_hidden=64, cls_hidden=64, num_classes=2):
        super().__init__()
        # 1) Build the shared trunk
        trunk_layers_list = []
        prev_dim = input_dim
        for _ in range(trunk_layers):
            trunk_layers_list += [
                nn.Linear(prev_dim, trunk_hidden),
                nn.BatchNorm1d(trunk_hidden),
                nn.ReLU(inplace=True),
            ]
            prev_dim = trunk_hidden
        self.trunk = nn.Sequential(*trunk_layers_list)

        # 2) Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(trunk_hidden, reg_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(reg_hidden, 1)
        )

        # 3) Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(trunk_hidden, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cls_hidden, num_classes)
        )

    def forward(self, x):
        features = self.trunk(x)
        reg_out = self.reg_head(features).squeeze(-1)    # shape (N,)
        cls_logits = self.cls_head(features)            # shape (N, num_classes)
        return reg_out, cls_logits
