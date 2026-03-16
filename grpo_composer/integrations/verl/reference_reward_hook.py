import copy
from typing import Any

class ReferenceRewardHook:
    """
    Hooks into veRL's `_compute_or_extract_reward` to dynamically generate 
    rollouts from the reference model and compute their rewards before computing 
    the active policy's rewards. The resulting tensor is attached 
    to `batch.batch['reference_rewards']`.
    """

    @staticmethod
    def wrap_compute_or_extract_reward(original_method: Any, trainer: Any):
        def hooked_compute_or_extract_reward(batch, reward_fn, return_dict=False, **kwargs):
            # Only do this if it hasn't been done for this batch yet and we have a reference policy
            if "reference_rewards" not in batch.batch and trainer.use_reference_policy:
                # 1. Create a baseline batch (greedy or sampled depending on algo, usually greedy)
                ref_batch = copy.deepcopy(batch)
                
                # Check config to see if we should sample or use greedy parsing
                do_sample = trainer.config.get("reference_rollout_do_sample", False)
                ref_batch.meta_info["do_sample"] = do_sample
                
                # 2. Generate sequences from the Reference Policy DataParallel WorkerGroup
                ref_output = trainer.ref_policy_wg.generate_sequences(ref_batch)
                
                # 3. Union prompts and generated responses
                ref_eval_batch = batch.union(ref_output)
                
                # 4. Compute reward on the reference rollouts
                # We call the original un-hooked method to avoid infinite recursion
                if trainer.config.reward_model.launch_reward_fn_async:
                    # Async reward not easily hookable synchronously, fallback to synchronous call
                    ref_reward_tensor, _ = original_method(
                        ref_eval_batch, reward_fn=reward_fn, return_dict=False, **kwargs
                    )
                else:
                    ref_reward_tensor, _ = original_method(
                        ref_eval_batch, reward_fn=reward_fn, return_dict=False, **kwargs
                    )
                
                # 5. Attach the reference reward explicitly to the active policy's non_tensor batch
                # veRL's `batch.union` and inter-process Ray RPC frequently strip out unrecognized
                # tensors in `.batch`. However, `.non_tensor_batch` is preserved verbatim.
                import numpy as np
                if isinstance(ref_reward_tensor, torch.Tensor):
                    batch.non_tensor_batch["reference_rewards"] = ref_reward_tensor.cpu().numpy()
                else:
                    batch.non_tensor_batch["reference_rewards"] = np.asarray(ref_reward_tensor)

            # 6. Proceed with the original compute_or_extract_reward for the actor policy
            return original_method(batch, reward_fn=reward_fn, return_dict=return_dict, **kwargs)

        return hooked_compute_or_extract_reward
