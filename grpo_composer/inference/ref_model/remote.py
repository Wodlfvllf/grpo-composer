"""
Remote Reference Model Server

Queries reference model on remote server via HTTP/gRPC.

What it does:
------------
- Sends token_ids to remote inference server
- Receives computed log-probs over network
- No local model loading required

When to use:
-----------
- Large models (70B+) on separate inference cluster
- Policy on training GPUs, ref on inference GPUs
- Scaling training and inference independently

Example:
-------
```python
remote_ref = RemoteReferenceModel(
    server_url="http://ref-server:8000"
)

ref_log_probs = remote_ref.get_log_probs(token_ids, mask)
# Network call to remote server
```

Server setup required separately (see `inference/serving/`).
"""
