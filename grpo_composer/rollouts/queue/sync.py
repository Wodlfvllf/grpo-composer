"""
Sync Queue — Synchronous generation wrapper.

═══════════════════════════════════════════════════════════════════
PURPOSE:
    Wraps Generator.generate() in a blocking call.
    Worker submits requests → waits → gets all results back.

    This is the DEFAULT queue for single-GPU / local generation.

BEHAVIOR:
    enqueue(requests)  →  stores requests
    execute()          →  calls Generator.generate(all_requests), blocks until done
    get_results()      →  returns List[RolloutResult]

    Essentially a pass-through. No concurrency, no async.

WHEN TO USE:
    - Single-GPU training with HuggingFace generate()
    - Local vLLM server on same machine
    - Debugging and testing

WHEN TO UPGRADE TO ASYNC:
    - Multi-GPU rollout generation via Ray
    - Remote vLLM server
    - When generation latency > training latency (overlap them)

═══════════════════════════════════════════════════════════════════
NOTE: For initial implementation, the Worker can call Generator.generate()
directly without going through a queue at all. This file exists for
when you need to decouple request submission from result retrieval.
"""
