import torch
import random
from typing import Any
import os

class InfoGRPORolloutAugmentor:
    
    @staticmethod
    def wrap_generate_sequences(original_generate_sequences_method, tokenizer):
        """
        Wraps the veRL generate_sequences method to intercept the DataProto batch,
        identify the second half of the rollouts (for Info-GRPO), and append
        a tokenized latent seed to the input_ids.
        """
        def hooked_generate(batch: Any):
            # batch is a DataProto instance.
            debug = os.environ.get("GRPO_COMPOSER_DEBUG") == "1"
            if debug:
                print("[Info-GRPO] Intercepting generation to inject latent seeds into augmented prompts...")
            
            input_ids = batch.batch["input_ids"]
            attention_mask = batch.batch["attention_mask"]
            
            B_total = input_ids.shape[0]
            # Get rollout.n (number of paths per prompt) from meta_info
            G_total = batch.meta_info.get("rollout_n", None)
            if G_total is None or G_total <= 1:
                return original_generate_sequences_method(batch)
            
            G = G_total // 2
            B = B_total // G_total
            
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
            # Step 1: Sample seeds and tokenize
            # We generate one seed per unique prompt (B prompts total)
            seed_tokens_list = []
            max_add_len = 0
            
            for _ in range(B):
                z = random.randint(0, 1000)
                # You can customize the latent seed format here
                seed_str = f"\nLatent Seed: {z}"
                encoded = tokenizer.encode(seed_str, add_special_tokens=False)
                seed_tokens_list.append(encoded)
                max_add_len = max(max_add_len, len(encoded))
                
            if max_add_len == 0:
                return original_generate_sequences_method(batch)
                
            seq_len = input_ids.shape[1]
            ext_seq_len = seq_len + max_add_len
            
            # Create new expanded tensors filled with padding
            new_input_ids = torch.full((B_total, ext_seq_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            new_attention_mask = torch.zeros((B_total, ext_seq_len), dtype=attention_mask.dtype, device=attention_mask.device)
            
            # Place active tokens at the right-edge to maintain safe left-padding for generation
            for b in range(B):
                seed_tokens = seed_tokens_list[b]
                
                for g in range(G_total):
                    idx = b * G_total + g
                    row_ids = input_ids[idx]
                    row_mask = attention_mask[idx]
                    
                    # Extract only the actual prompt tokens (ignoring padding)
                    active_ids = row_ids[row_mask == 1]
                    
                    if g < G:
                        # First half: Original trajectories (no seed)
                        final_ids = active_ids
                    else:
                        # Second half: Augmented trajectories (append seed)
                        seed_tensor = torch.tensor(seed_tokens, dtype=row_ids.dtype, device=row_ids.device)
                        final_ids = torch.cat([active_ids, seed_tensor])
                        
                    # Place at the very end
                    L = len(final_ids)
                    new_input_ids[idx, -L:] = final_ids
                    # the attention mask should be 1 for the prompt+seed tokens
                    new_attention_mask[idx, -L:] = 1
            
            # Replace the tensors in the batch safely
            batch.batch["input_ids"] = new_input_ids
            batch.batch["attention_mask"] = new_attention_mask
            
            # Call the real generator!
            return original_generate_sequences_method(batch)
            
        return hooked_generate
