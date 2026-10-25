import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class LightweightLGLM(nn.Module):
    """
    Lightweight Sparse Quantized Neural Network for patient-specific glucose prediction.

    Architecture:
        Input (batch, window_size, features)
        -> Conv1D temporal feature extraction
        -> Flatten
        -> Dense + ReLU  (sparse/quantized representation)
        -> Output (predicted glucose, scaled)
    """

    def __init__(self, input_features: int, hidden_units: int = 32,
                 window_size: int = 12, sparsity_ratio: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.input_features = input_features
        self.sparsity_ratio = sparsity_ratio

        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=16,
                               kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.flatten_size = 16 * window_size
        self.fc1 = nn.Linear(self.flatten_size, hidden_units)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_units, 1)

        # Quantization stubs (for QAT / PTQ)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, features) -> Conv1d wants (batch, features, window_size)
        x = x.transpose(1, 2)
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu2(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x.squeeze(-1)

    def apply_sparsity(self):
        """L1-unstructured pruning to simulate sparse weights."""
        if self.sparsity_ratio > 0.0:
            prune.l1_unstructured(self.fc1, name='weight', amount=self.sparsity_ratio)
            prune.remove(self.fc1, 'weight')
            prune.l1_unstructured(self.fc2, name='weight', amount=self.sparsity_ratio)
            prune.remove(self.fc2, 'weight')

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        nonzero = sum(
            torch.count_nonzero(p).item() for p in self.parameters()
        )
        return {
            'total': total,
            'trainable': trainable,
            'nonzero': nonzero,
            'sparsity_percent': (1.0 - nonzero / max(1, total)) * 100
        }
