from __future__ import annotations

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, output_dim: int = 4) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)