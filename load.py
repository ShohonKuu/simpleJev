"""从已经下好的快照加载 Qwen3.5-0.8B。不联网。"""

from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.processing_utils import ProcessorMixin

# 本文件在 DATA/projects/jev/simpleJev。向上四级是用户主目录。
# Windows 的 Z: 和 Linux 的 /home/suheng 是同一份文件。
# 快照 2fc06364715b967f1860aea9cf38778875588b17，权重约 1.75GB。
MODEL = str(
    Path(__file__).resolve().parents[4]
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
)


def load() -> tuple[Qwen3_5ForConditionalGeneration, ProcessorMixin]:
    # local_files_only 只读上面这个目录，不会写到 Windows 用户自己的缓存。
    # Qwen3_5ForConditionalGeneration 自带视觉塔，OneJev 带图时加载的就是这个类。
    model: Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
    )
    processor: ProcessorMixin = AutoProcessor.from_pretrained(
        MODEL, trust_remote_code=False, local_files_only=True
    )
    # 权重文件里只有参数，没有 .grad。requires_grad 默认为 True，梯度要等以后 backward 才写上。
    model.eval()
    return model, processor
