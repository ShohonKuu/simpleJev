"""决策模型的配置。六个决策标记的编号写在本类上。"""

from transformers import PreTrainedConfig


class SimpleJevConfig(PreTrainedConfig):
    model_type = "simplejev"

    def __init__(self, hidden_size: int = 1024, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size: int = hidden_size
        self.initializer_range: float = initializer_range
        self.question_begin_id: int = 248077
        self.question_end_id: int = 248078
        self.options_begin_id: int = 248079
        self.options_end_id: int = 248080
        self.candidate_begin_id: int = 248081
        self.candidate_end_id: int = 248082
