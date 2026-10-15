"""Run the shared prefix once. Continue each option from its own copy of that cache."""

import copy

import torch
from transformers.cache_utils import Cache


def _batch(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def prefill(language_model, prefix_ids: list[int], device: torch.device) -> Cache:
    cache = language_model(input_ids=_batch(prefix_ids, device), use_cache=True).past_key_values
    if cache is None:
        raise RuntimeError("prefix cache is None")
    return cache


def suffix_hidden(
    language_model, cache: Cache, suffix_ids: list[int], device: torch.device
) -> torch.Tensor:
    # A suffix forward writes the DeltaNet recurrence and the convolution history.
    # Each option needs a private copy so later options do not see earlier ones.
    branch_cache = copy.deepcopy(cache)
    hidden = language_model(
        input_ids=_batch(suffix_ids, device),
        past_key_values=branch_cache,
        use_cache=True,
    ).last_hidden_state[0, -1]
    return hidden


def option_hiddens(
    language_model,
    prefix_ids: list[int],
    suffix_ids: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    cache = prefill(language_model, prefix_ids, device)
    rows = [suffix_hidden(language_model, cache, suffix, device) for suffix in suffix_ids]
    return torch.stack(rows)
