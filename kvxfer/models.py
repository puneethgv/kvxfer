"""Loading models and picking a sane device/dtype for this machine."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def best_device() -> torch.device:
    """Pick the fastest available backend."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """Compute dtype to use when the caller does not specify one.

    float32 on CPU because bf16 matmuls there are slow and often emulated;
    bfloat16 on accelerators, which is what these models were trained in.
    """
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def load_model(
    model_id: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Load a causal LM in eval mode with gradients disabled.

    Args:
        model_id: HF model id.
        device: target device; defaults to :func:`best_device`.
        dtype: compute dtype; defaults to :func:`default_dtype`. Pass float32
            explicitly when running correctness gates, so that dtype noise does
            not get mistaken for mapper error.
    """
    device = torch.device(device) if device is not None else best_device()
    dtype = dtype if dtype is not None else default_dtype(device)

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_tokenizer(model_id: str):
    """Load a tokenizer, ensuring a pad token exists for batched work."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
