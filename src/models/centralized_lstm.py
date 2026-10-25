import torch
import torch.nn as nn


class CentralizedLSTM(nn.Module):
    """Centralized LSTM baseline — trained on pooled data from all patients."""

    def __init__(self, input_features: int, hidden_units: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {'total': total, 'trainable': total}
