"""
Data Collator

This module implements the data collator for the GRPO model.
Loads the data from the dataset and collates it into batches.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

class DataCollator:
    def __init__(
        self,
        tokenizer,
        template_name,
        max_length : int = 1024,
        max_target_length : int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.formatter = PromptFormatter(
            template_name=template_name
        )
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch : List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        
        input_texts = self.formatter.batch_format(batch)

        # Tokenize all texts
        encodings = self.tokenizer(
            input_texts,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # For CLM, labels = input_ids shifted by 1 (handled by loss function)
        # We mask padding tokens with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    