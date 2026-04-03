# Fixes Applied to the `next` Branch

This document describes bug fixes applied after cloning the `next` branch of osprey.

---

## 1. Provider Loading Failure in Interactive Init

**Symptom:**
```
! No providers could be loaded from osprey registry
```
When running `osprey` (interactive project creation), Step 6 (Provider Selection) shows no providers.

**Root Cause:**
The `next` branch moved provider definitions from the registry config (`FrameworkRegistryProvider.get_registry_config().providers`) to a standalone `ProviderRegistry` in `osprey.models.provider_registry`. However, the CLI init wizard in `interactive_menu.py` still read from the old (now empty) location.

**Fix:**
In `src/osprey/cli/interactive_menu.py`, function `get_provider_metadata()` (~line 326):

Replaced:
```python
from osprey.registry.builtins import FrameworkRegistryProvider
framework_registry = FrameworkRegistryProvider()
config = framework_registry.get_registry_config()
for provider_reg in config.providers:  # config.providers is empty []
    ...
```

With:
```python
from osprey.models.provider_registry import get_provider_registry
pr = get_provider_registry()
for provider_name in pr.list_providers():
    provider_class = pr.get_provider(provider_name)
    ...
```

This reads from `_BUILTIN_PROVIDERS` in `provider_registry.py` (the actual source of truth), which contains all 11 providers.

---

## 2. Code Generator Loading Failure in Interactive Init

**Symptom:**
```
✗ No code generators available
! Osprey could not load any code generators.
Check that osprey is properly installed: uv sync --all-extras
```
When running `osprey`, Step 5 (Code Generator) fails and aborts project creation.

**Root Cause:**
The `get_code_generator_metadata()` function in `interactive_menu.py` had code generators removed from the registry but was never updated with an alternative. The function body was:
```python
generators = {}  # hardcoded empty
```

**Fix:**
In `src/osprey/cli/interactive_menu.py`, function `get_code_generator_metadata()` (~line 423):

Replaced the empty `generators = {}` with entries for the two generators that the scaffolding code (`src/osprey/cli/templates/scaffolding.py`) still expects:

```python
# "basic" is always available
generators["basic"] = {
    "name": "basic",
    "description": "Simple single-pass LLM code generator",
    "available": True,
}

# "claude_code" requires claude-agent-sdk
try:
    import importlib
    importlib.import_module("claude_agent_sdk")
    generators["claude_code"] = {
        "name": "claude_code",
        "description": "Claude Code SDK-based generator with agentic coding",
        "available": True,
    }
except ImportError:
    generators["claude_code"] = {
        "name": "claude_code",
        "description": "Claude Code SDK-based generator with agentic coding",
        "available": False,
        "optional_dependencies": ["claude-agent-sdk"],
    }
```

---

## 3. Argo Base URL Update

**Symptom:**
Argo API calls fail because the old endpoint (`https://argo-bridge.cels.anl.gov`) is no longer active.

**Fix:**
In `src/osprey/models/providers/argo.py`, updated the base URL in two places:

- **Line 73** (fallback in `_execute_argo_structured_output()`):
  `"https://argo-bridge.cels.anl.gov"` -> `"https://apps.inside.anl.gov/argoapi/v1"`

- **Line 145** (`default_base_url` class attribute on `ArgoProviderAdapter`):
  `"https://argo-bridge.cels.anl.gov"` -> `"https://apps.inside.anl.gov/argoapi/v1"`

---

## Verification

After applying all fixes:

```bash
source /home/oxygen/SHANG/next_osprey/.venv/bin/activate
python3 -c "
from osprey.cli.interactive_menu import get_provider_metadata, get_code_generator_metadata

providers = get_provider_metadata()
print(f'Providers loaded: {len(providers)}')
for name in sorted(providers):
    print(f'  {name}')

generators = get_code_generator_metadata()
print(f'Code generators loaded: {len(generators)}')
for name, meta in sorted(generators.items()):
    print(f'  {name}: available={meta[\"available\"]}')
"
```

Expected output:
```
Providers loaded: 11
  als-apg
  amsc
  anthropic
  argo
  asksage
  cborg
  google
  ollama
  openai
  stanford
  vllm
Code generators loaded: 2
  basic: available=True
  claude_code: available=True
```
