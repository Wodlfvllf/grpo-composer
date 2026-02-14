"""
Inference Client

Connects to remote inference server to request model outputs.

What it does:
------------
- Sends requests to inference server
- Serializes/deserializes data over network
- Handles retries and timeouts
- Returns results as if local

When to use:
-----------
- Calling remote reference model server
- Distributed training with remote inference
- Load balancing across multiple servers

Example:
-------
```python
# Connect to server
client = InferenceClient(
    server_url="http://ref-model-server:8000",
    timeout=30
)

# Request log-probs (looks like local call)
log_probs = client.compute_log_probs(
    token_ids=tensor,
    attention_mask=mask
)
```

Used by:
- `RemoteReferenceModel` (wraps this client)
- Training scripts connecting to remote inference

Protocol:
- HTTP REST API (JSON payloads)
- Optional: gRPC for lower latency
"""
