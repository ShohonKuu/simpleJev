"""Load the local Qwen3.5-0.8B snapshot. This module does not download."""

from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.processing_utils import ProcessorMixin

# Snapshot 2fc06364715b967f1860aea9cf38778875588b17. Weights are about 1.75 GB.
SNAPSHOT = "2fc06364715b967f1860aea9cf38778875588b17"


def snapshot_dir() -> Path:
    # This file is <home>/DATA/projects/jev/simpleJev/src/simplejev/loading.py.
    # Windows Z: and Linux /home/suheng are the same tree, so parent 6 is home.
    home = Path(__file__).resolve().parents[6]
    return home / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots" / SNAPSHOT


def load_backbone() -> tuple[Qwen3_5ForConditionalGeneration, ProcessorMixin]:
    # local_files_only reads that directory and does not write another cache.
    # Qwen3_5ForConditionalGeneration includes the vision tower. OneJev loads
    # this same class when a question contains an image.
    path = str(snapshot_dir())
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    # The checkpoint stores parameters only. requires_grad stays True until a
    # later backward writes .grad.
    model.eval()
    return model, processor
