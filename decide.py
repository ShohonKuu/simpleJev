"""跑一道样例。模型本身在 modeling_simplejev.py。"""

import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.processing_utils import ProcessorMixin

from configuration_simplejev import SimpleJevConfig
from load import load
from modeling_simplejev import SimpleJevForDecision, SimpleJevOutput
from processing_simplejev import SimpleJevInputs, SimpleJevProcessor


def main() -> None:
    qwen: Qwen3_5ForConditionalGeneration
    processor: ProcessorMixin
    qwen, processor = load()
    config: SimpleJevConfig = SimpleJevConfig()
    model: SimpleJevForDecision = SimpleJevForDecision(config, qwen)
    model.init_decision_rows()
    inputs: SimpleJevInputs = SimpleJevProcessor(processor.tokenizer, config)(
        "一加一等于几？",
        ["二", "三","四"],
    )
    output: SimpleJevOutput = model(inputs.prefix_ids, inputs.suffix_ids)
    probs: torch.Tensor = output.probs
    print(probs, float(probs.sum()))


if __name__ == "__main__":
    main()
