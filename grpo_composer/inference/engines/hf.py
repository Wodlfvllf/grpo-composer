"""
HuggingFace Generator

Uses HuggingFace Transformers `.generate()` for text completion.

"""

from ...interfaces import InferenceEngine, RolloutRequest, RolloutResult
from typing import List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFGenerator(InferenceEngine):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype=torch.float32,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def generate(self, request: RolloutRequest) -> RolloutResult:

        input_ids = request.input_ids.to(self.device)
        attention_mask = request.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences  # (B*G, T_total)
        scores = outputs.scores        # list length = new_tokens

        # -------------------------------------------------
        # Compute token log-probs
        # -------------------------------------------------

        logits = torch.stack(scores, dim=0)  # (new_tokens, B*G, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Move batch first
        log_probs = log_probs.permute(1, 0, 2)  # (B*G, new_tokens, vocab)

        T_prompt = input_ids.shape[1]

        generated_tokens = sequences[:, T_prompt:]  # (B*G, new_tokens)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=generated_tokens.unsqueeze(-1),
        ).squeeze(-1)  # (B*G, new_tokens)

        return RolloutResult(
            completions=generated_tokens,   # (B*G, L)
            log_probs=token_log_probs,      # (B*G, L)
        )

    def get_log_probs(self, input_ids, attention_mask) -> torch.Tensor:
        """Single forward pass → log-probs of existing tokens."""

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits  # (B, T, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log-probs at each position for the *next* token
        # Shift: log_probs[:, :-1] aligned with input_ids[:, 1:]
        shifted_log_probs = log_probs[:, :-1, :]
        shifted_ids = input_ids[:, 1:]

        token_log_probs = shifted_log_probs.gather(
            dim=-1,
            index=shifted_ids.unsqueeze(-1),
        ).squeeze(-1)  # (B, T-1)

        return token_log_probs
