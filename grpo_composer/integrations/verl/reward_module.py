
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reward_Ranking_module(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 4):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        rank = torch.argsort(probs, dim=-1, descending=True)
        return probs, rank