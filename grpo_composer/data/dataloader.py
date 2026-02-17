import torch
from typing import Optional, List
from torch.utils.data import DataLoader

from .data_collator import DataCollator
from .datasets import HuggingFaceDataset


class GRPODataloader:
    def __init__(
        self,
        tokenizer,
        template_name: str,
        name: Optional[str] = None,
        data_format: Optional[str] = None,
        data_files: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        target: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.name = name
        self.data_format = data_format
        self.data_files = data_files
        self.prompt = prompt
        self.target = target

        # Train dataset
        self.train_dataset = HuggingFaceDataset(
            name=name,
            split="train",
            data_format=data_format,
            data_files=data_files,
            prompt=prompt,
            target=target,
        )

        # Test dataset
        self.test_dataset = HuggingFaceDataset(
            name=name,
            split="test",
            data_format=data_format,
            data_files=data_files,
            prompt=prompt,
            target=target,
        )

        # Collator
        self.collate_fn = DataCollator(
            tokenizer=self.tokenizer,
        )

    def get_dataloaders(self, batch_size: int):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        return train_loader, test_loader
