"""
Tokenizer Abstraction Layer

Provides unified tokenizer interface that works across different backends
(HuggingFace, SentencePiece, etc.).

Purpose:
-------
Abstract away tokenizer differences so the rest of the codebase can work
with a consistent API, regardless of the underlying tokenizer implementation.

Role in Training Pipeline:
-------------------------
**Initialization**: Load tokenizer alongside model
**Generation**: Used by inference engines for encoding/decoding
**Preprocessing**: Validate prompt lengths, count tokens

Key Responsibilities:
--------------------
1. Load tokenizers from HuggingFace, local files, or custom
2. Encode text → token IDs
3. Decode token IDs → text
4. Handle special tokens (BOS, EOS, PAD)
5. Provide token counting utilities
6. Support chat templates

Why an Abstraction Layer?
------------------------
Different models use different tokenizers:
- Llama: SentencePiece BPE
- GPT: Byte-level BPE
- T5: SentencePiece Unigram

We need a unified interface so code doesn't break when switching models.

Classes to Implement:
--------------------
1. **Tokenizer (ABC)**
   - Abstract interface
   
2. **HuggingFaceTokenizer(Tokenizer)**
   - Wraps `transformers.AutoTokenizer`
   - Most common case

Key Interface:
-------------
```python
class Tokenizer(ABC):
    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        '''
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS
            return_tensors: "pt" for torch.Tensor, None for list
            
        Returns:
            Token IDs
        '''
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        '''
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Remove BOS/EOS/PAD
            
        Returns:
            Decoded text
        '''
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        '''Return vocabulary size.'''
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        '''Get PAD token ID.'''
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        '''Get EOS token ID.'''
        pass
    
    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        '''Get BOS token ID.'''
        pass
    
    def count_tokens(self, text: str) -> int:
        '''
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        '''
        return len(self.encode(text, add_special_tokens=False))
```

HuggingFace Implementation:
---------------------------
```python
class HuggingFaceTokenizer(Tokenizer):
    def __init__(
        self,
        model_name_or_path: str,
        padding_side: str = "left",
        truncation_side: str = "right",
        use_fast: bool = True
    ):
        '''
        Load HuggingFace tokenizer.
        
        Args:
            model_name_or_path: HF model name or path
            padding_side: "left" or "right"
            truncation_side: "left" or "right"
            use_fast: Use fast Rust implementation
        '''
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side=padding_side,
            truncation_side=truncation_side,
            use_fast=use_fast
        )
        
        # Ensure PAD token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
```

Example Usage:
-------------
```python
# Load tokenizer
tokenizer = HuggingFaceTokenizer("meta-llama/Llama-2-7b-hf")

# Encode
token_ids = tokenizer.encode("Hello, world!")
# [1, 15043, 29892, 3186, 29991]

# Decode
text = tokenizer.decode(token_ids)
# "Hello, world!"

# Count tokens
count = tokenizer.count_tokens("This is a test prompt.")
# 7
```

Special Token Handling:
----------------------
```python
def setup_special_tokens(tokenizer: Tokenizer) -> None:
    '''
    Ensure tokenizer has all required special tokens.
    
    This is important for models that don't define PAD token
    by default (e.g., Llama, Mistral).
    '''
    if tokenizer.pad_token_id is None:
        # Use EOS as PAD
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("PAD token not defined, using EOS token")
```

Implementation Notes:
--------------------
- Default to HuggingFace tokenizer for compatibility
- Support custom tokenizers via plugin system
- Cache loaded tokenizers to avoid reloading
- Handle missing PAD token gracefully (use EOS)
- Provide utilities for batch encoding/decoding
- Support fast tokenizers (Rust) when available
"""
