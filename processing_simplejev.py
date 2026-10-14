"""把题干和选项编成共享前缀和每条后缀。标记 id 不进分词器。"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from configuration_simplejev import SimpleJevConfig


@dataclass
class SimpleJevInputs:
    prefix_ids: list[int]
    suffix_ids: list[list[int]]


class SimpleJevProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: SimpleJevConfig) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.config: SimpleJevConfig = config

    def __call__(self, description: str, options: Sequence[str]) -> SimpleJevInputs:
        if len(options) < 2:
            raise ValueError("options 至少两项")
        # 原始分词器没有这六个字符串。只编码用户文字，再把整数 id 拼到两侧。
        prefix: list[int] = (
            [self.config.question_begin_id]
            + self.tokenizer.encode(description, add_special_tokens=False)
            + [self.config.question_end_id]
            + [self.config.options_begin_id]
            + self.tokenizer.encode(
                json.dumps(list(options), ensure_ascii=False), add_special_tokens=False
            )
            + [self.config.options_end_id]
        )
        suffixes: list[list[int]] = [
            [self.config.candidate_begin_id]
            + self.tokenizer.encode(option, add_special_tokens=False)
            + [self.config.candidate_end_id]
            for option in options
        ]
        return SimpleJevInputs(prefix_ids=prefix, suffix_ids=suffixes)
