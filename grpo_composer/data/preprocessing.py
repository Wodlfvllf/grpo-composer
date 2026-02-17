import logging
from typing import Optional, List, Dict
from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ═══════════════════════════════════════════════════════════
# Template Registry
# ═══════════════════════════════════════════════════════════

TEMPLATES = {
    "alpaca": {
        "with_system": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### System:\n{system}\n\n"
            "### Instruction:\n{prompt}\n\n"
            "### Response:\n"
        ),
        "without_system": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n"
            "### Response:\n"
        ),
    },

    "chatml": {
        "with_system": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "without_system": (
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },

    "llama2": {
        "with_system": (
            "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
            "{prompt} [/INST]"
        ),
        "without_system": "<s>[INST] {prompt} [/INST]",
    },

    "llama3": {
        "with_system": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "without_system": (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
}


# ═══════════════════════════════════════════════════════════
# Prompt Formatter
# ═══════════════════════════════════════════════════════════

@dataclass
class PromptFormatter:
    template_name: str
    tokenizer = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if self.template_name not in TEMPLATES:
            raise ValueError(f"Unknown template: {self.template_name}")

        self.template = TEMPLATES[self.template_name]

    def format_prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:

        if system_message:
            template = self.template["with_system"]
            return template.format(system=system_message, prompt=prompt)
        else:
            template = self.template["without_system"]
            return template.format(prompt=prompt)

    def batch_format(self, batch: List[Dict[str, str]]):
        outputs = []

        for example in batch:
            formatted_prompt = self.format_prompt(example["prompt"])

            # Proper CLM style: prompt + target
            full_text = formatted_prompt + example["target"]

            outputs.append(full_text)

        return outputs

    # -------------------------------------------------------
    # 2. Add Few Shot Examples
    # -------------------------------------------------------
    def add_few_shot_examples(
        self,
        formatted_prompt: str,
        examples: List[Dict[str, str]],
    ) -> str:

        few_shot_block = ""

        for example in examples:
            q = example.get("Q")
            a = example.get("A")

            few_shot_block += f"Q: {q}\nA: {a}\n\n"

        combined = few_shot_block + formatted_prompt

        return combined

    # -------------------------------------------------------
    # 3. Validate Prompt Length
    # -------------------------------------------------------
    def validate_prompt_length(self, prompt: str) -> str:

        if not self.tokenizer or not self.max_tokens:
            return prompt

        tokens = self.tokenizer.encode(prompt)

        if len(tokens) <= self.max_tokens:
            return prompt

        logger.warning(
            f"Prompt too long ({len(tokens)} tokens). "
            f"Truncating to {self.max_tokens}."
        )

        truncated_tokens = tokens[: self.max_tokens]
        truncated_prompt = self.tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=False
        )

        return truncated_prompt


# ═══════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    formatter = PromptFormatter(
        template_name="alpaca",
        tokenizer_name="meta-llama/Llama-2-7b-hf",  # change if needed
        max_tokens=512
    )

    raw_prompt = "What is the sum of 5 and 7?"

    formatted = formatter.format_prompt(
        prompt=raw_prompt,
        system_message="You are a helpful math tutor."
    )

    with_examples = formatter.add_few_shot_examples(
        formatted_prompt=formatted,
        examples=[
            {"Q": "What is 2+2?", "A": "4"},
            {"Q": "What is 10-3?", "A": "7"},
        ]
    )

    final_prompt = formatter.validate_prompt_length(with_examples)

    print("========== FINAL PROMPT ==========")
    print(final_prompt)
