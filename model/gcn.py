import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNLayer


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, num_classes)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: two-layer GCN
        """
        h1 = self.gcn1(H, A_norm)
        logits = self.gcn2(h1, A_norm)
        return logits