"""
Pretrained Model Loader

Loads any HuggingFace CausalLM as a plain nn.Module.
Both InferenceEngine and TrainingEngine receive the model from here.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from typing import Optional


def load_pretrained(
    model_name_or_path: str,
    dtype: str = "float16",
    device: str = "cuda",
    device_map: Optional[str] = None,
) -> nn.Module:
    """
    Load a pretrained HuggingFace CausalLM.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        dtype: One of "float16", "bfloat16", "float32".
        device: Target device (ignored if device_map is set).
        device_map: HF device_map string ("auto", "balanced", etc.).

    Returns:
        nn.Module — the loaded model in eval mode.
    """

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if device_map is None:
        model = model.to(device)

    # model.eval()
    return model


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """
    Load tokenizer for a pretrained model. Sets pad_token if missing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_vllm(
    model_name_or_path: str,
    dtype: str = "float16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> LLM:
    """
    Load a pretrained model via vLLM's LLM class.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        dtype: One of "float16", "bfloat16", "float32".
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Max sequence length (None = model default).

    Returns:
        vllm.LLM instance ready for generation.
    """
    return LLM(
        model=model_name_or_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
