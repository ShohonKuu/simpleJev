"""Qwen3.5 language backbone plus one shared score head.

Loaded Qwen weights are not re-initialized. The forward calls three steps
in the decision package: rows, branch, then score.
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration
from transformers.utils import ModelOutput

from .configuration import SimpleJevConfig
from .decision import init_decision_rows, option_hiddens, option_probs


@dataclass
class SimpleJevOutput(ModelOutput):
    probs: torch.Tensor | None = None


class SimpleJevForDecision(nn.Module):
    def __init__(self, config: SimpleJevConfig, qwen: Qwen3_5ForConditionalGeneration) -> None:
        super().__init__()
        self.config = config
        self.qwen = qwen
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

    def init_decision_rows(self) -> None:
        embed = self.qwen.model.language_model.embed_tokens
        init_decision_rows(embed, self.config)

    def forward(self, prefix_ids: list[int], suffix_ids: list[list[int]]) -> SimpleJevOutput:
        # Cache tensors must be leaves. deepcopy raises if autograd is tracking them.
        with torch.no_grad():
            hiddens = option_hiddens(
                self.qwen.model.language_model,
                prefix_ids,
                suffix_ids,
                self.score.weight.device,
            )
            probs = option_probs(self.score, hiddens)
        return SimpleJevOutput(probs=probs)
