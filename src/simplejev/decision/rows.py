"""Initialize the six decision-token rows. Every other vocabulary row stays as loaded."""

import torch
from torch import nn

from ..configuration import SimpleJevConfig


def init_decision_rows(embed: nn.Embedding, config: SimpleJevConfig) -> None:
    ids = config.decision_token_ids()
    start = min(ids)
    stop = max(ids) + 1
    if tuple(range(start, stop)) != ids:
        raise ValueError(
            "decision token ids must be the contiguous range from question_begin through candidate_end"
        )
    # The embedding is bfloat16. Sample the noise in float32, then copy it in.
    with torch.no_grad():
        rows = embed.weight[start:stop]
        noise = torch.empty(rows.shape, dtype=torch.float32, device=rows.device)
        noise.normal_(0.0, config.initializer_range)
        rows.copy_(noise)
