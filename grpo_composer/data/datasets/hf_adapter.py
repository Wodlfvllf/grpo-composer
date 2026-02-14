"""
HuggingFace Datasets Adapter

Loads supervised datasets with prompts and targets for GRPO training.

Purpose:
-------
Load datasets from HuggingFace Hub or local files, returning both:
- **Prompts**: Input text for generation
- **Targets**: Ground truth for reward verification

Return Format:
-------------
All methods return Dict[str, str]:
```python
{
    "prompt": str,  # Input for generation
    "target": str   # Ground truth for verification
}
```

Usage Modes:
-----------

**1. HuggingFace Hub with Built-in Adapter:**
```python
dataset = HuggingFaceDataset(
    name="gsm8k",
    split="train"
)

item = dataset[0]
# {"prompt": "Natalia sold...", "target": "1152"}

batch = dataset.sample(4)
# {"prompt": [...], "target": [...]}
```

**2. Local Files with Custom Field Mapping:**
```python
dataset = HuggingFaceDataset(
    data_format="json",         # json, csv, etc.
    data_files=["data.jsonl"],
    prompt="question",          # Your prompt field
    target="answer",            # Your target field
    split="train"
)
```

Built-in Adapters:
-----------------
- `gsm8k` → question/answer
- `numina-math-cot` → problem/solution
- `cohere-aya-multilingual` → inputs/targets
- `google-mbpp` → text/code

For unlisted datasets, use custom field mapping with `prompt` and `target` params.

Implementation:
--------------
- Uses `datasets.load_dataset()` from HuggingFace
- Adapter pattern for dataset-specific field mappings
- Random sampling with `torch.randint()`
- Targets stored for reward verification (rule-based evaluators)
"""
from .base import Dataset
from datasets import load_dataset
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SupervisedExample:
    prompt: str
    target: str

def gsm8k_adapter(ex):
    return SupervisedExample(
        prompt=ex["question"],
        target=ex["answer"],
    )

def numina_math_cot_adapter(ex):
    return SupervisedExample(
        prompt=ex["problem"],
        target=ex["solution"],
    )

def cohere_aya_adapter(ex):
    return SupervisedExample(
        prompt=ex["inputs"],
        target=ex["targets"],
    )

def google_mbpp_adapter(ex):
    return SupervisedExample(
        prompt=ex["text"],
        target=ex["code"],
    )

  

class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        name: Optional[str] = None,
        split: str = "train",
        data_format: Optional[str] = None,
        data_files: Optional[List[str]] = None,
        prompt : Optional[str] = None,
        target : Optional[str] = None 
    ):
        self.name = name
        self.split = split
        self.prompt = prompt
        self.target = target

        self.adapters = {
            "gsm8k": gsm8k_adapter,
            "numina-math-cot": numina_math_cot_adapter,
            "cohere-aya-multilingual": cohere_aya_adapter,
            "google-mbpp": google_mbpp_adapter,
        }

        if name:
            self.dataset = load_dataset(name, split=split)
        elif data_format and data_files:
            self.dataset = load_dataset(data_format, data_files=data_files, split=split)
        else:
            raise ValueError("Provide either HF dataset name or local dataset files")

        if name and name not in self.adapters:
            raise ValueError(f"No adapter registered for dataset: {name}")

    def __getitem__(self, idx):
        raw_example = self.dataset[idx]
        if self.name:
            adapter = self.adapters[self.name]
        else:
            def custom_adapter(ex):
                return SupervisedExample(
                    prompt=ex[f"{self.prompt}"],
                    target=ex[f"{self.target}"],
                )

            adapter = custom_adapter

        example = adapter(raw_example)

        return {
            "prompt": example.prompt,
            "target": example.target,
        }

    def __len__(self):
        return len(self.dataset)

    def _batch_adapter(self, samples):
        output = {
            "prompt": [],
            "target": [],
        }

        if self.name:
            adapter = self.adapters[self.name]
        else:
            def custom_adapter(ex):
                return SupervisedExample(
                    prompt=ex[f"{self.prompt}"],
                    target=ex[f"{self.target}"],
                )

            adapter = custom_adapter

        # number of samples in batch
        batch_len = len(next(iter(samples.values())))

        for i in range(batch_len):
            single = {k: v[i] for k, v in samples.items()}
            example = adapter(single)

            output["prompt"].append(example.prompt)
            output["target"].append(example.target)

        return output

    def sample(self, batch_size: int):
        dataset_len = len(self.dataset)

        indices = torch.randint(
            low=0,
            high=dataset_len,
            size=(batch_size,)
        ).tolist()

        samples = self.dataset[indices]
        return self._batch_adapter(samples)
