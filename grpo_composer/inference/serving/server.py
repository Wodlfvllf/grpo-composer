"""
Inference Server

Serves models (reference, reward, policy) over HTTP/gRPC for remote access.

What it does:
------------
- Loads model into memory
- Exposes REST/gRPC API endpoints
- Handles batching and queueing
- Returns log-probs or rewards

When to use:
-----------
- Separate training and inference clusters
- Reference model on dedicated hardware
- Multiple trainers sharing one inference backend

Example:
-------
```python
# Start server
server = InferenceServer(
    model_path="meta-llama/Llama-2-13b-hf",
    host="0.0.0.0",
    port=8000,
    backend="vllm"  # Use vLLM for serving
)
server.start()
```

API endpoints:
- POST /generate - Generate completions
- POST /compute_log_probs - Get log-probs for tokens
- GET /health - Health check

Used by:
- `RemoteReferenceModel` (client in ref_model/server.py)
- Distributed training setups
"""
