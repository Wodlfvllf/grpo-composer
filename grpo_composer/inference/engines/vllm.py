"""
vLLM Inference Engine

Uses vLLM's LLM class for fast batched generation.
Receives a pre-loaded vLLM LLM instance from models/pretrained.py.
"""

from ...interfaces import InferenceEngine, RolloutRequest, RolloutResult
import torch
from typing import Dict, Any, Optional
from vllm import LLM, SamplingParams


class VLLMInferenceEngine(InferenceEngine):
    def __init__(
        self,
        llm: LLM,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize vLLM engine.

        Args:
            llm: Pre-loaded vLLM LLM instance (from models.load_vllm).
            sampling_params: dict with keys matching vLLM SamplingParams, e.g.:
                {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                 "max_tokens": 512, "logprobs": 1, "stop": None}
        """
        self.llm = llm
        self.sampling_params = sampling_params or {}

    def generate(self, request: RolloutRequest) -> RolloutResult:
        """
        Generate completions from token IDs.

        Args:
            request: RolloutRequest with input_ids (B*G, T) and attention_mask (B*G, T).

        Returns:
            RolloutResult with completions (B*G, L) and log_probs (B*G, L).
        """

        sampling_params = SamplingParams(**self.sampling_params)

        # Build list of TokensPrompt dicts — one per row in the batch
        prompts = [
            {"prompt_token_ids": row.tolist()}
            for row in request.input_ids
        ]

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        all_token_ids = []
        all_log_probs = []

        for output in outputs:
            out = output.outputs[0]  # single sequence per prompt (n=1)
            all_token_ids.append(list(out.token_ids))

            if out.logprobs is not None:
                token_logprobs = [
                    list(tlp.values())[0].logprob
                    for tlp in out.logprobs
                ]
                all_log_probs.append(token_logprobs)

        # Pad to equal length
        max_len = max(len(c) for c in all_token_ids)
        padded_ids = [c + [0] * (max_len - len(c)) for c in all_token_ids]

        if all_log_probs:
            padded_lp = [lp + [0.0] * (max_len - len(lp)) for lp in all_log_probs]
            log_probs_tensor = torch.tensor(padded_lp)
        else:
            log_probs_tensor = torch.zeros(len(padded_ids), max_len)

        return RolloutResult(
            completions=torch.tensor(padded_ids),   # (B*G, L)
            log_probs=log_probs_tensor,              # (B*G, L)
        )

    def get_log_probs(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Score existing tokens via vLLM prompt_logprobs.

        Returns: (B, T-1) log-probs for each token given its prefix.
        """

        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0,
        )

        prompts = [
            {"prompt_token_ids": row.tolist()}
            for row in input_ids
        ]

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        all_log_probs = []
        for i, output in enumerate(outputs):
            tokens = input_ids[i].tolist()
            prompt_lps = output.prompt_logprobs  # list of dicts, length T

            # First token has no log-prob (it's the BOS/start)
            token_log_probs = []
            for t in range(1, len(prompt_lps)):
                lp_dict = prompt_lps[t]
                token_id = tokens[t]
                if lp_dict is not None and token_id in lp_dict:
                    token_log_probs.append(lp_dict[token_id].logprob)
                else:
                    token_log_probs.append(0.0)
            all_log_probs.append(token_log_probs)

        # Pad to equal length
        max_len = max(len(lp) for lp in all_log_probs)
        padded = [lp + [0.0] * (max_len - len(lp)) for lp in all_log_probs]

        return torch.tensor(padded)  # (B, T-1)

    def reload_model(self, new_model_path: str):
        """
        Reload a new model checkpoint.
        Used after policy update stage in RL.
        """
        from ...models import load_vllm

        print(f"Reloading model from: {new_model_path}")

        del self.llm
        torch.cuda.empty_cache()

        self.llm = load_vllm(new_model_path)

        print("Model reloaded successfully.")