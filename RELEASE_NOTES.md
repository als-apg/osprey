# Osprey Framework - Latest Release (v0.10.1)

🎉 **Direct Chat Mode & LiteLLM Migration** - Conversational capability interaction and unified LLM provider interface

## What's New in v0.10.1

### 💬 Direct Chat Mode

A new way to interact with capabilities in a conversational flow!

- **Enter Direct Chat**: `/chat:<capability>` - Start chatting with a specific capability
- **List Available**: `/chat` - See all direct-chat enabled capabilities
- **Exit**: `/exit` - Return to normal orchestrated mode
- **Dynamic Prompt**: See which mode you're in (normal vs capability name)
- **Context Tools**: Save, read, and manage context during conversations

**Built-in Direct Chat Capabilities:**
- `state_manager` - Inspect and manage agent state
- MCP-generated capabilities are direct-chat enabled by default

### ⚡ LiteLLM Migration (#23)

Major backend simplification - all LLM providers now use a unified interface:

- **~2,200 lines → ~700 lines** - Massive code reduction
- **8 Providers**: anthropic, openai, google, ollama, cborg, stanford, argo, vllm
- **100+ Models**: Access to all LiteLLM-supported providers
- **Preserved Features**: Extended thinking, structured outputs, health checks

### 🆕 New Provider: vLLM

High-throughput local inference support:

- OpenAI-compatible interface via LiteLLM
- Auto-detects served models
- Supports structured outputs

### 🔧 LangChain Model Factory

Native integration with LangGraph ReAct agents:

```python
from osprey.models import get_langchain_model

model = get_langchain_model(provider="anthropic", model_id="claude-sonnet-4")
# Use with create_react_agent, etc.
```

### 📚 Documentation Updates

- CLI Reference: Direct chat mode commands and examples
- Gateway Architecture: Message history preservation
- Building First Capability: `direct_chat_enabled` attribute guide

---

## Installation

```bash
pip install --upgrade osprey-framework
```

Or install with all optional dependencies:

```bash
pip install --upgrade "osprey-framework[all]"
```

## Upgrading from v0.10.0

### Direct Chat Mode

No migration needed! Direct chat mode is opt-in:

```python
# Add to your capability to enable direct chat
@capability_node
class MyCapability(BaseCapability):
    direct_chat_enabled = True  # New in 0.10.1
```

### LiteLLM Migration

The API remains the same - `get_chat_completion()` works exactly as before.
Backend providers now use LiteLLM internally.

---

## What's Next?

Check out our [documentation](https://als-apg.github.io/osprey) for:
- Direct chat mode tutorial
- LangChain integration guide
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
