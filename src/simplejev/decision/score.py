"""Turn each option's last hidden state into one probability."""

import torch
from torch import nn


def option_probs(score: nn.Linear, hiddens: torch.Tensor) -> torch.Tensor:
    """Score a [num_options, hidden] tensor. The returned probabilities sum to 1."""
    logits = [score(row.float()).squeeze() for row in hiddens]
    return torch.softmax(torch.stack(logits), dim=0)
