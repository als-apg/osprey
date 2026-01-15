# Fix: JSON Boolean Parsing in Structured Output

## Issue

When using the Argo provider (or other LLM providers), task extraction would fail with a JSON parsing error:

```
Failed to parse structured output from argo: 1 validation error for ExtractedTask
  Invalid JSON: expected value at line 3 column 30
Response: {
  "task": "Create a time-series plot of SR beam current data for yesterday",
  "depends_on_chat_history": True,
  "depends_on_user_memory": False
}
```

## Root Cause

Some LLM models return Python-style boolean literals (`True`/`False`) instead of JSON-style (`true`/`false`). This causes `model_validate_json()` to fail since `True` and `False` are not valid JSON values.

## Solution

Updated the `_clean_json_response()` function in `litellm_adapter.py` to:

1. Convert Python-style `True` to JSON-style `true`
2. Convert Python-style `False` to JSON-style `false`
3. Apply this fix to both native structured output and prompt-based fallback paths

### Code Changes

**File:** `src/osprey/models/providers/litellm_adapter.py`

```python
def _clean_json_response(text: str) -> str:
    """Clean markdown code blocks and fix common JSON issues from LLM response."""
    import re

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # Fix Python-style booleans (True/False) to JSON-style (true/false)
    text = re.sub(r':\s*True\b', ': true', text)
    text = re.sub(r':\s*False\b', ': false', text)
    text = re.sub(r',\s*True\b', ', true', text)
    text = re.sub(r',\s*False\b', ', false', text)

    return text
```

Additionally, the cleanup is now applied to both code paths:
- Native structured output path (line 232-233)
- Prompt-based fallback path (line 248-249)

## Testing

Test with CLI:
```bash
osprey chat
# Then: "plot sr beam current of yesterday"
```

The task extraction should now succeed even when the LLM returns Python-style booleans.

## Related Files

- `src/osprey/models/providers/litellm_adapter.py` - Main fix location
- `src/osprey/infrastructure/task_extraction_node.py` - Uses structured output for ExtractedTask
