import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        H: [N, in_features], A_norm: [N, N]
        return ReLU( A_norm @ H @ W )
        """
        support = torch.matmul(A_norm, H)
        out = self.linear(support)
        return F.relu(out)
