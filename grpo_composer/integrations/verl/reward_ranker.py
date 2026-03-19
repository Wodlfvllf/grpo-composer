"""Reward ranker adapters and DataProto integration for RankGRPO-like flows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

_DEBUG_DUMP_DONE = False
_MASK_WARN_DONE = False


class BaseRanker:
    """Interface: rank each response within a prompt group."""

    def rank(self, prompts: list[str], responses: list[list[str]]) -> list[list[int]]:
        """Return per-group ranks aligned to input order (1=best, G=worst)."""
        raise NotImplementedError


class HeuristicRanker(BaseRanker):
    """Simple baseline ranker: shorter responses rank higher."""

    def rank(self, prompts: list[str], responses: list[list[str]]) -> list[list[int]]:
        all_ranks: list[list[int]] = []
        for group in responses:
            order = sorted(range(len(group)), key=lambda i: len(group[i]))
            ranks = [0] * len(group)
            for pos, idx in enumerate(order):
                ranks[idx] = pos + 1  # 1-indexed
            all_ranks.append(ranks)
        return all_ranks


class RRMRanker(BaseRanker):
    def __init__(self, ray_actor: Any):
        self.actor = ray_actor

    def rank(self, prompts: list[str], responses: list[list[str]]) -> list[list[int]]:
        import ray

        return ray.get(self.actor.rank.remote(prompts, responses))


def _batch_get(data: Any, key: str):
    if hasattr(data, "batch") and key in data.batch:
        return data.batch[key]
    return None


def _non_tensor_get(data: Any, key: str):
    non_tensor = getattr(data, "non_tensor_batch", None)
    if isinstance(non_tensor, Mapping) and key in non_tensor:
        return non_tensor[key]
    return None


def _first_present(data: Any, keys: tuple[str, ...], *, prefer_non_tensor: bool = False):
    for key in keys:
        value = _non_tensor_get(data, key) if prefer_non_tensor else _batch_get(data, key)
        if value is not None:
            return value
        value = _batch_get(data, key) if prefer_non_tensor else _non_tensor_get(data, key)
        if value is not None:
            return value
    return None


def _as_uid_array(data: Any) -> np.ndarray:
    uid = _first_present(data, ("uid",), prefer_non_tensor=True)
    if uid is None:
        raise ValueError("Ranking requires uid in data.non_tensor_batch or data.batch")
    if isinstance(uid, torch.Tensor):
        uid = uid.detach().cpu().numpy()
    uid_arr = np.asarray(uid)
    if uid_arr.ndim != 1:
        raise ValueError(f"uid must be 1D, got shape {uid_arr.shape}")
    return uid_arr


def _decode_ids(
    rows: torch.Tensor,
    mask: torch.Tensor | None,
    tokenizer: Any,
    *,
    field_name: str,
    debug: bool = False,
) -> list[str]:
    if rows.ndim != 2:
        raise ValueError(f"Expected 2D token ids, got shape {tuple(rows.shape)}")

    rows_cpu = rows.detach().cpu()
    mask_cpu = mask.detach().cpu() if isinstance(mask, torch.Tensor) else None
    use_mask = (
        isinstance(mask_cpu, torch.Tensor)
        and mask_cpu.ndim == 2
        and tuple(mask_cpu.shape) == tuple(rows_cpu.shape)
    )

    global _MASK_WARN_DONE
    if debug and mask_cpu is not None and not use_mask and not _MASK_WARN_DONE:
        _MASK_WARN_DONE = True
        print(
            "[composer-debug][rank] mask/ids shape mismatch; decoding ids without mask: "
            f"{field_name}_ids={tuple(rows_cpu.shape)} mask={tuple(mask_cpu.shape)}"
        )

    texts: list[str] = []
    for i in range(rows_cpu.shape[0]):
        row = rows_cpu[i]
        if use_mask:
            row = row[mask_cpu[i].bool()]
        token_ids = row.tolist()
        if tokenizer is None:
            texts.append(" ".join(str(int(tok)) for tok in token_ids))
        else:
            texts.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    return texts


def _group_indices(uid_arr: np.ndarray) -> tuple[list[Any], dict[Any, list[int]]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    order: list[Any] = []
    for idx, key in enumerate(uid_arr.tolist()):
        if key not in groups:
            order.append(key)
        groups[key].append(idx)
    return order, groups


def _rank_shape_guard(group_ranks: list[int], group_size: int, group_key: Any) -> None:
    if len(group_ranks) != group_size:
        raise ValueError(
            f"Ranker returned wrong group length for uid={group_key}: "
            f"expected {group_size}, got {len(group_ranks)}"
        )


def _one_time_debug_dump(data: Any, uid_arr: np.ndarray, groups: dict[Any, list[int]]) -> None:
    global _DEBUG_DUMP_DONE
    if _DEBUG_DUMP_DONE:
        return
    _DEBUG_DUMP_DONE = True

    batch = getattr(data, "batch", {})
    non_tensor = getattr(data, "non_tensor_batch", {})
    shape_parts = []
    if hasattr(batch, "keys"):
        for key in batch.keys():
            value = batch[key]
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            shape_parts.append(f"{key}={shape}")
    group_sizes = [len(v) for v in groups.values()]
    hist = defaultdict(int)
    for size in group_sizes:
        hist[int(size)] += 1
    print(
        "[composer-debug][rank] data snapshot: "
        f"batch_keys={shape_parts} "
        f"non_tensor_keys={list(non_tensor.keys()) if hasattr(non_tensor, 'keys') else type(non_tensor)} "
        f"uid_total={uid_arr.shape[0]} uid_unique={len(groups)} "
        f"group_hist={dict(sorted(hist.items()))}"
    )


def _debug_predecode_shapes(
    *,
    data: Any,
    uid_arr: np.ndarray,
    groups: dict[Any, list[int]],
    response_ids: Any,
    response_mask: Any,
    prompt_ids: Any,
    prompt_mask: Any,
) -> None:
    group_sizes = [len(v) for v in groups.values()]
    hist = defaultdict(int)
    for size in group_sizes:
        hist[int(size)] += 1

    rollout_n = None
    meta_info = getattr(data, "meta_info", None)
    if isinstance(meta_info, Mapping):
        rollout_n = meta_info.get("rollout_n", meta_info.get("n"))
    else:
        getter = getattr(meta_info, "get", None)
        if callable(getter):
            try:
                rollout_n = getter("rollout_n", getter("n", None))
            except Exception:
                rollout_n = None

    def _shape(x: Any):
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, np.ndarray):
            return tuple(x.shape)
        return type(x).__name__

    print(
        "[composer-debug][rank] predecode: "
        f"responses={_shape(response_ids)} response_mask={_shape(response_mask)} "
        f"prompts={_shape(prompt_ids)} prompt_mask={_shape(prompt_mask)} "
        f"uid_total={uid_arr.shape[0]} uid_unique={len(groups)} "
        f"group_hist={dict(sorted(hist.items()))} rollout_n={rollout_n}"
    )
    try:
        expected_n = int(rollout_n) if rollout_n is not None else None
    except Exception:
        expected_n = None
    if expected_n is not None and expected_n > 0:
        bad_sizes = [size for size in group_sizes if size != expected_n]
        if bad_sizes:
            print(
                "[composer-debug][rank] group-size mismatch vs rollout_n: "
                f"expected={expected_n} mismatched_sizes={sorted(set(bad_sizes))}"
            )


def ensure_reward_ranks(
    data: Any,
    ranker: BaseRanker,
    *,
    tokenizer: Any = None,
    config: Any = None,
    debug: bool = False,
) -> None:
    """Attach row-aligned `rrm_ranks` (1=best) to non_tensor_batch."""
    if ranker is None:
        return

    if _first_present(data, ("rrm_ranks", "composer_rrm_ranks"), prefer_non_tensor=True) is not None:
        return

    uid_arr = _as_uid_array(data)
    row_count = int(uid_arr.shape[0])
    uid_order, groups = _group_indices(uid_arr)

    response_ids = _first_present(data, ("responses", "response_ids"), prefer_non_tensor=False)
    response_mask = _first_present(data, ("response_mask",), prefer_non_tensor=False)
    if not isinstance(response_ids, torch.Tensor):
        raise ValueError("Ranking requires `responses` tensor in data.batch")

    prompt_ids = _first_present(data, ("prompts", "prompt_ids"), prefer_non_tensor=False)
    prompt_mask = _first_present(data, ("prompt_mask",), prefer_non_tensor=False)

    if debug:
        _debug_predecode_shapes(
            data=data,
            uid_arr=uid_arr,
            groups=groups,
            response_ids=response_ids,
            response_mask=response_mask,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
        )

    response_texts = _decode_ids(
        response_ids,
        response_mask,
        tokenizer,
        field_name="responses",
        debug=debug,
    )
    if isinstance(prompt_ids, torch.Tensor):
        prompt_texts = _decode_ids(
            prompt_ids,
            prompt_mask if isinstance(prompt_mask, torch.Tensor) else None,
            tokenizer,
            field_name="prompts",
            debug=debug,
        )
        if len(prompt_texts) != row_count:
            if debug:
                print(
                    "[composer-debug][rank] prompt row-count mismatch; falling back to empty prompts: "
                    f"prompt_rows={len(prompt_texts)} response_rows={row_count}"
                )
            prompt_texts = [""] * row_count
    else:
        prompt_texts = [""] * row_count

    group_prompts: list[str] = []
    group_responses: list[list[str]] = []
    for key in uid_order:
        idxs = groups[key]
        group_prompts.append(prompt_texts[idxs[0]] if idxs else "")
        group_responses.append([response_texts[i] for i in idxs])

    ranked_groups = ranker.rank(group_prompts, group_responses)
    if len(ranked_groups) != len(uid_order):
        raise ValueError(
            f"Ranker returned {len(ranked_groups)} groups for {len(uid_order)} uid groups"
        )

    row_ranks = np.zeros((row_count,), dtype=np.int64)
    for group_pos, uid_key in enumerate(uid_order):
        idxs = groups[uid_key]
        group_ranks = [int(x) for x in ranked_groups[group_pos]]
        _rank_shape_guard(group_ranks, len(idxs), uid_key)
        for local_i, row_i in enumerate(idxs):
            row_ranks[row_i] = group_ranks[local_i]

    non_tensor = getattr(data, "non_tensor_batch", None)
    if non_tensor is None:
        data.non_tensor_batch = {}
        non_tensor = data.non_tensor_batch
    non_tensor["rrm_ranks"] = row_ranks
    non_tensor["composer_rrm_ranks"] = row_ranks

    if debug:
        _one_time_debug_dump(data, uid_arr, groups)
