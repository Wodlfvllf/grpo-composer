"""
Prompt Preprocessing and Formatting

Handles prompt templating, formatting, and preprocessing before generation.

Purpose:
-------
Convert raw dataset entries into properly formatted prompts that can be
sent to the language model for completion generation.

Role in Training Pipeline:
-------------------------
**After Dataset Loading**: Raw prompts → Formatted prompts
**Before Generation**: Apply templates, add special tokens, format CoT

Key Responsibilities:
--------------------
1. Apply prompt templates (e.g., instruction format, few-shot)
2. Handle chain-of-thought (CoT) prompting
3. Add special tokens and formatting
4. Support multiple prompt formats (Llama, Mistral, custom)
5. Validate prompt lengths

Common Prompt Formats:
---------------------
1. **Instruction Format** (Alpaca, Llama-2-chat):
```
### Instruction:
{prompt}

### Response:
```

2. **ChatML Format** (GPT, Mistral):
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

3. **Few-Shot Format**:
```
Q: What is 1+1?
A: 2

Q: What is 2+2?
A: 4

Q: {prompt}
A:
```

Functions to Implement:
----------------------
1. **format_prompt()**
   - Apply template to raw prompt
   - Return formatted string

2. **add_few_shot_examples()**
   - Prepend few-shot examples
   - Handle context length limits

3. **validate_prompt_length()**
   - Check token count
   - Truncate if necessary

Example Usage:
-------------
```python
# Load raw prompt
raw_prompt = "What is the sum of 5 and 7?"

# Format with Alpaca template
formatted = format_prompt(
    prompt=raw_prompt,
    template="alpaca",
    system_message="You are a helpful math tutor."
)

# Add few-shot examples
with_examples = add_few_shot_examples(
    formatted,
    examples=[
        {"Q": "What is 2+2?", "A": "4"},
        {"Q": "What is 10-3?", "A": "7"}
    ]
)

# Result:
# Q: What is 2+2?
# A: 4
# 
# Q: What is 10-3?
# A: 7
#
# ### Instruction:
# What is the sum of 5 and 7?
# 
# ### Response:
```

Implementation Notes:
--------------------
- Use Jinja2 templates for complex formatting
- Support custom templates via YAML config
- Integrate with HuggingFace tokenizer.apply_chat_template()
- Cache formatted prompts to avoid recomputation
- Log warning if prompt exceeds max_tokens
"""
