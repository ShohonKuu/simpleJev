"""Qwen3.5 文本骨干加一个共享评分头。不重新初始化已加载的权重。"""

import copy
from dataclasses import dataclass

import torch
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput

from configuration_simplejev import SimpleJevConfig


@dataclass
class SimpleJevOutput(ModelOutput):
    probs: torch.Tensor | None = None


class SimpleJevForDecision(nn.Module):
    def __init__(self, config: SimpleJevConfig, qwen: Qwen3_5ForConditionalGeneration) -> None:
        super().__init__()
        self.config: SimpleJevConfig = config
        self.qwen: Qwen3_5ForConditionalGeneration = qwen
        self.score: nn.Linear = nn.Linear(config.hidden_size, 1, bias=False)

    def init_decision_rows(self) -> None:
        # 权重是 bfloat16。只改决策标记那几行：噪声在 float32 里 normal_，再 copy_ 回去。
        ids: tuple[int, ...] = (
            self.config.question_begin_id,
            self.config.question_end_id,
            self.config.options_begin_id,
            self.config.options_end_id,
            self.config.candidate_begin_id,
            self.config.candidate_end_id,
        )
        start: int = min(ids)
        stop: int = max(ids) + 1
        embed: nn.Embedding = self.qwen.model.language_model.embed_tokens
        with torch.no_grad():
            rows: torch.Tensor = embed.weight[start:stop]
            noise: torch.Tensor = torch.empty(rows.shape, dtype=torch.float32, device=rows.device)
            noise.normal_(0.0, self.config.initializer_range)
            rows.copy_(noise)

    def forward(self, prefix_ids: list[int], suffix_ids: list[list[int]]) -> SimpleJevOutput:
        # 不关梯度时，cache 里的 K/V 不是叶子张量，deepcopy 会直接报错。
        with torch.no_grad():
            device: torch.device = self.score.weight.device
            prefix_tensor: torch.Tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            cache: Cache | None = self.qwen.model.language_model(
                input_ids=prefix_tensor, use_cache=True
            ).past_key_values
            if cache is None:
                raise RuntimeError("prefix cache 是 None")
            # 每条选项单独 deepcopy：后缀 forward 会改掉 cache 里的 DeltaNet 循环状态和卷积历史。
            scores: list[torch.Tensor] = []
            suffix: list[int]
            for suffix in suffix_ids:
                branch_cache: Cache = copy.deepcopy(cache)
                suffix_tensor: torch.Tensor = torch.tensor([suffix], dtype=torch.long, device=device)
                hidden: torch.Tensor = self.qwen.model.language_model(
                    input_ids=suffix_tensor,
                    past_key_values=branch_cache,
                    use_cache=True,
                ).last_hidden_state[0, -1]
                logit: torch.Tensor = self.score(hidden.float()).squeeze()
                scores.append(logit)
            probs: torch.Tensor = torch.softmax(torch.stack(scores), dim=0)
            return SimpleJevOutput(probs=probs)
