import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def train_local_model(model: nn.Module, X_train: np.ndarray, Y_train: np.ndarray,
                      epochs: int = 3, batch_size: int = 64, lr: float = 0.001,
                      loss_fn: str = 'MSE') -> tuple:
    """
    Performs local gradient training:  W_i^(t+1) = W_i^t - η ∇L(W_i^t)
    Returns updated model and final epoch loss.
    """
    model.train()
    criterion = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss(),
                 'Huber': nn.HuberLoss()}.get(loss_fn, nn.MSELoss())
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(float(np.mean(batch_losses)))

    return model, epoch_losses[-1] if epoch_losses else 0.0


def evaluate_model(model: nn.Module, X: np.ndarray, Y: np.ndarray) -> tuple:
    """Evaluate model — returns (mse_loss, predictions_np)."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X))
    preds_np = preds.numpy()
    mse = float(((Y - preds_np) ** 2).mean())
    return mse, preds_np
