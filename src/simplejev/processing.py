"""Turn a question and its options into one prefix and one suffix each."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .configuration import SimpleJevConfig


@dataclass
class SimpleJevInputs:
    prefix_ids: list[int]
    suffix_ids: list[list[int]]


class SimpleJevProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: SimpleJevConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, description: str, options: Sequence[str]) -> SimpleJevInputs:
        if len(options) < 2:
            raise ValueError("options must contain at least two items")
        prefix = self.question_ids(description) + self.option_list_ids(options)
        suffixes = [self.candidate_ids(option) for option in options]
        return SimpleJevInputs(prefix_ids=prefix, suffix_ids=suffixes)

    def question_ids(self, description: str) -> list[int]:
        return self._wrap(description, self.config.question_begin_id, self.config.question_end_id)

    def option_list_ids(self, options: Sequence[str]) -> list[int]:
        text = json.dumps(list(options), ensure_ascii=False)
        return self._wrap(text, self.config.options_begin_id, self.config.options_end_id)

    def candidate_ids(self, option: str) -> list[int]:
        return self._wrap(option, self.config.candidate_begin_id, self.config.candidate_end_id)

    def _wrap(self, text: str, begin_id: int, end_id: int) -> list[int]:
        # Encode the text only. The marker ids are not in the tokenizer vocabulary.
        body = self.tokenizer.encode(text, add_special_tokens=False)
        return [begin_id, *body, end_id]
