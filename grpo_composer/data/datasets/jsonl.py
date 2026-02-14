"""
JSONL Dataset Loader

Loads prompts from JSONL (newline-delimited JSON) files.

Purpose:
-------
Support custom datasets stored as JSONL files, which is a common
format for training data in NLP (one JSON object per line).

Role in Training Pipeline:
-------------------------
**Config Specification**: `dataset: file://path/to/prompts.jsonl`
**Startup**: Load JSONL file into memory or create file iterator
**Training Loop**: Sample prompts from loaded data

Key Responsibilities:
--------------------
1. Read JSONL files line by line
2. Extract prompt field from each JSON object
3. Support lazy loading for large files
4. Handle malformed JSON gracefully
5. Provide efficient random access with indexing

JSONL File Format:
-----------------
```jsonl
{"prompt": "What is 2+2?", "metadata": {"difficulty": "easy"}}
{"prompt": "Solve x^2 = 16", "metadata": {"difficulty": "medium"}}
{"prompt": "Prove Fermat's Last Theorem", "metadata": {"difficulty": "hard"}}
```

Configuration Options:
---------------------
```yaml
dataset:
  type: jsonl
  path: "data/math_prompts.jsonl"
  prompt_field: "prompt"      # JSON field containing prompt
  lazy_load: false            # Load on-demand vs. all at once
  encoding: "utf-8"           # File encoding
```

Classes to Implement:
--------------------
1. **JSONLDataset(Dataset)**
   - Reads JSONL files
   - Caches prompts or maintains file handle
   - Supports random access

Key Methods:
-----------
```python
class JSONLDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_field: str = "prompt",
        lazy_load: bool = False,
        encoding: str = "utf-8"
    ):
        '''
        Load JSONL dataset.
        
        Args:
            path: Path to JSONL file
            prompt_field: JSON key containing the prompt text
            lazy_load: If True, don't load entire file into memory
            encoding: File encoding
        '''
        pass
    
    def _load_all(self) -> List[str]:
        '''Load entire file into memory (for small datasets).'''
        pass
    
    def _build_index(self) -> List[int]:
        '''
        Build byte offset index for lazy loading.
        Allows O(1) random access without loading entire file.
        '''
        pass
    
    def __len__(self) -> int:
        '''Return number of lines in file.'''
        pass
    
    def __getitem__(self, idx: int) -> str:
        '''
        Get prompt at index.
        If lazy_load=True, seek to line and parse JSON.
        Otherwise, return from in-memory cache.
        '''
        pass
    
    def sample(self, batch_size: int) -> List[str]:
        '''Sample batch of prompts.'''
        pass
```

Example Usage:
-------------
```python
# Load custom JSONL dataset
dataset = JSONLDataset(
    path="data/custom_math_problems.jsonl",
    prompt_field="question",
    lazy_load=True  # For large files
)

# Sample prompts
prompts = dataset.sample(batch_size=4)
```

Implementation Notes:
--------------------
- For lazy loading, build byte-offset index on init
- Use `json.loads()` to parse each line
- Handle malformed JSON with try/except, log warning, skip line
- Support gzip-compressed JSONL files (.jsonl.gz)
- Validate prompt_field exists in first line during init

Edge Cases:
----------
- Empty file → raise ValueError
- Malformed JSON line → log warning, skip, continue
- Missing prompt_field → raise KeyError with line number
- Very large files → require lazy_load=True
- Unicode issues → handle with encoding parameter
"""
