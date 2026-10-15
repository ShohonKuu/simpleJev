"""Widths and the six reserved decision-token ids."""

from transformers import PreTrainedConfig


class SimpleJevConfig(PreTrainedConfig):
    model_type = "simplejev"

    def __init__(self, hidden_size: int = 1024, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        # These six ids sit in unused vocabulary slots. The tokenizer has no
        # string for them, so the processor inserts the integers itself.
        self.question_begin_id = 248077
        self.question_end_id = 248078
        self.options_begin_id = 248079
        self.options_end_id = 248080
        self.candidate_begin_id = 248081
        self.candidate_end_id = 248082

    def decision_token_ids(self) -> tuple[int, ...]:
        """The six ids, contiguous from question_begin through candidate_end."""
        return (
            self.question_begin_id,
            self.question_end_id,
            self.options_begin_id,
            self.options_end_id,
            self.candidate_begin_id,
            self.candidate_end_id,
        )
