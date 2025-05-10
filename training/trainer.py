
import torch
from torch.utils.data import DataLoader


def train(model: torch.nn.Module,
          data_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion,
          device: torch.device) -> float:
    """
    Train for one epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        features, A_norm, labels = batch
        features = features.to(device)
        A_norm = A_norm.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features, A_norm)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def validate(model: torch.nn.Module,
             data_loader: DataLoader,
             criterion,
             device: torch.device) -> float:
    """
    Validate for one epoch. Returns average loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            features, A_norm, labels = batch
            features = features.to(device)
            A_norm = A_norm.to(device)
            labels = labels.to(device)

            outputs = model(features, A_norm)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model