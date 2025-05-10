import torch
import torch.nn.functional as F


def weighted_cross_entropy(outputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted cross-entropy
    """
    return F.cross_entropy(outputs, labels, weight=weights)


def lateralization_loss(outputs: torch.Tensor, labels: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    Placeholder lateralization penalty: returns zero tensor.
    """
    return torch.tensor(0.0, device=outputs.device)


def compute_loss(outputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    Total loss = weighted CE + lambda * lateralization
    """
    ce = weighted_cross_entropy(outputs, labels, weights)
    lat = lateralization_loss(outputs, labels, lambda_)
    return ce + lambda_ * lat