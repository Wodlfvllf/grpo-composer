
import torch
import torch.nn as nn
from typing import List, Any

class PVPOHookAugmentor:
    @staticmethod
    def _wrap_generate_sequences(original_generate_sequences_method, reference_generate_sequences_method):
        
        def hooked_batch(batch : Any):
            gen_batch_output = original_generate_sequences_method(batch)
            ref_gen_batch_output = reference_generate_sequences_method(batch)
            