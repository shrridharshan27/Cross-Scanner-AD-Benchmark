"""PyTorch MLP Classifier for Alzheimer's Disease Classification.

Architecture:
  Linear(4, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1) -> Sigmoid
"""

import numpy as np
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """3-layer Multilayer Perceptron for tabular neuroimaging features."""

    def __init__(self, input_dim: int = 4, hidden_dims: tuple = (32, 16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probabilities in [0, 1]."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x) -> np.ndarray:
        """Convenience inference function returning 1D probability array."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            probs = self.forward(x).cpu().numpy()
        return probs
