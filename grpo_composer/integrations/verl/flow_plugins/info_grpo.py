"""Info-GRPO latent-seed injection as a :class:`FlowPlugin`.

Replaces the previous ``InfoGRPORolloutAugmentor.wrap_generate_sequences``
monkey-patch on ``actor_rollout_wg.generate_sequences`` /
``async_rollout_manager.generate_sequences``. The trainer now calls
:meth:`InfoGRPOFlowPlugin.before_generate` directly at every rollout site,
so we no longer rebind methods on live worker handles.
"""

from __future__ import annotations

import os
import random
from typing import Any

import torch

from ..trainer import FlowPlugin


def _inject_latent_seeds(batch: Any, tokenizer: Any) -> Any:
    """Append a tokenized latent seed to the second half of each prompt's rollouts.

    ``batch`` is a veRL ``DataProto`` carrying ``input_ids`` /
    ``attention_mask`` shaped ``(B_total, seq_len)`` where
    ``B_total = B_prompts * rollout_n``. The first ``rollout_n // 2``
    trajectories per prompt are left unchanged; the remaining half receive
    a per-prompt random latent seed appended to the prompt.
    """

    debug = os.environ.get("GRPO_COMPOSER_DEBUG") == "1"
    if debug:
        print(
            "[Info-GRPO] Intercepting generation to inject latent seeds into "
            "augmented prompts..."
        )

    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]

    B_total = input_ids.shape[0]
    G_total = batch.meta_info.get("rollout_n", None)
    if G_total is None or G_total <= 1:
        return batch

    G = G_total // 2
    B = B_total // G_total

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    seed_tokens_list: list[list[int]] = []
    max_add_len = 0
    for _ in range(B):
        z = random.randint(0, 1000)
        seed_str = f"\nLatent Seed: {z}"
        encoded = tokenizer.encode(seed_str, add_special_tokens=False)
        seed_tokens_list.append(encoded)
        max_add_len = max(max_add_len, len(encoded))

    if max_add_len == 0:
        return batch

    seq_len = input_ids.shape[1]
    ext_seq_len = seq_len + max_add_len

    new_input_ids = torch.full(
        (B_total, ext_seq_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    new_attention_mask = torch.zeros(
        (B_total, ext_seq_len), dtype=attention_mask.dtype, device=attention_mask.device
    )

    # Place active tokens at the right edge to keep safe left-padding for generation.
    for b in range(B):
        seed_tokens = seed_tokens_list[b]
        for g in range(G_total):
            idx = b * G_total + g
            row_ids = input_ids[idx]
            row_mask = attention_mask[idx]
            active_ids = row_ids[row_mask == 1]

            if g < G:
                final_ids = active_ids
            else:
                seed_tensor = torch.tensor(
                    seed_tokens, dtype=row_ids.dtype, device=row_ids.device
                )
                final_ids = torch.cat([active_ids, seed_tensor])

            L = len(final_ids)
            new_input_ids[idx, -L:] = final_ids
            new_attention_mask[idx, -L:] = 1

    batch.batch["input_ids"] = new_input_ids
    batch.batch["attention_mask"] = new_attention_mask
    return batch


class InfoGRPOFlowPlugin(FlowPlugin):
    """Inject Info-GRPO latent seeds before each rollout call."""

    def configure(self, trainer) -> None:
        self._tokenizer = trainer.tokenizer

    def before_generate(self, trainer, batch):
        return _inject_latent_seeds(batch, self._tokenizer)
