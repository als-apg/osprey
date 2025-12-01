# Integration Guide - Quick Reference

**Quick reference for integrating your API/workflow into the Alpha Berkeley Framework**

---

## 🎯 Three-Step Integration Process

### Step 1: Define Your Data Structure
**File**: `context_classes.py`

```python
class YourDataContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "YOUR_DATA"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

    # Your fields here
    field1: str = Field(description="Description for LLM")
    field2: List[float] = Field(description="Description for LLM")

    def get_access_details(self, key: str):
        # Tell LLM how to access your data
        return {...}

    def get_summary(self):
        # Provide human-readable summary
        return {...}
```

### Step 2: Implement Your Capability
**File**: `capabilities/your_capability.py`

```python
@capability_node
class YourCapability(BaseCapability):
    name = "your_capability"
    description = "What it does"
    provides = ["YOUR_DATA"]
    requires = []  # Dependencies

    @staticmethod
    async def execute(state: AgentState, **kwargs):
        # 1. Get current step
        step = StateManager.get_current_step(state)

        # 2. YOUR CODE HERE - Call your API/perform logic
        result = await your_api.call()

        # 3. Create context object
        output = YourDataContext(
            field1=result['data'],
            field2=result['values']
        )

        # 4. Store and return using helper method
        return self.store_output_context(output)
```

### Step 3: Register Components
**File**: `registry.py`

```python
return extend_framework_registry(
    capabilities=[
        CapabilityRegistration(
            name="your_capability",
            module_path="{{ package_name }}.capabilities.your_capability",
            class_name="YourCapability",
            description="What it does",
            provides=["YOUR_DATA"],
            requires=[]
        ),
    ],
    context_classes=[
        ContextClassRegistration(
            context_type="YOUR_DATA",
            module_path="{{ package_name }}.context_classes",
            class_name="YourDataContext"
        ),
    ],
)
```

---

## 🔄 Common Patterns

### Pattern: Simple API Call

```python
# Context Class
class APIResponseContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "API_RESPONSE"
    data: Dict[str, Any] = Field(description="API response data")

# Capability
@capability_node
class APICapability(BaseCapability):
    name = "api_call"
    provides = ["API_RESPONSE"]
    requires = []

    @staticmethod
    async def execute(state: AgentState, **kwargs):
        response = await api_client.get('/endpoint')
        output = APIResponseContext(data=response.json())
        return StateManager.store_context(...)
```

### Pattern: Data Processing with Dependencies

```python
# Context Class
class ProcessedDataContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "PROCESSED_DATA"
    results: List[float] = Field(description="Processed results")

# Capability
@capability_node
class ProcessorCapability(BaseCapability):
    name = "processor"
    provides = ["PROCESSED_DATA"]
    requires = ["API_RESPONSE"]  # Needs data from above

    @staticmethod
    async def execute(state: AgentState, **kwargs):
        # Extract required input with cardinality constraint
        context_manager = ContextManager(state)
        contexts = context_manager.extract_from_step(
            step, state,
            constraints=[("API_RESPONSE", "single")],
            constraint_mode="hard" # Validates presence and cardinality
        )
        api_data = contexts["API_RESPONSE"]

        # The "single" constraint guarantees api_data is not a list

        # Process it
        results = process(api_data.data)

        # Return processed results
        output = ProcessedDataContext(results=results)
        return StateManager.store_context(...)
```

### Pattern: Knowledge Retrieval

```python
# Context Class
class KnowledgeContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "KNOWLEDGE"
    documents: List[str] = Field(description="Retrieved documents")

# Capability
@capability_node
class KnowledgeCapability(BaseCapability):
    name = "knowledge"
    provides = ["KNOWLEDGE"]
    requires = []

    async def execute(self):
        # Get task objective using helper method
        query = self.get_task_objective()

        docs = await knowledge_base.search(query)

        output = KnowledgeContext(documents=docs)
        return self.store_output_context(output)
```

---

## 🧩 Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│ USER QUERY                                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ FRAMEWORK (LLM Orchestrator)                                │
│ - Classifies task                                           │
│ - Plans execution sequence                                  │
│ - Coordinates capabilities                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ YOUR CAPABILITY                                             │
│ - Receives AgentState with context                         │
│ - Executes your logic (API call, computation, etc.)        │
│ - Creates context object with results                      │
│ - Returns updated state                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ CONTEXT OBJECT                                              │
│ - Type-safe data container (Pydantic)                      │
│ - Stored in agent state                                    │
│ - Accessible to LLM and other capabilities                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM / DOWNSTREAM CAPABILITIES                               │
│ - Access data via context                                  │
│ - Generate responses or perform further processing         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Integration Checklist

### Planning
- [ ] What data does my API/workflow provide?
- [ ] What data does it need as input?
- [ ] Do I need one capability or multiple?
- [ ] What are the dependencies between capabilities?

### Implementation
- [ ] Create context class(es) in `context_classes.py`
- [ ] Copy `example_capability.py.j2` to new file
- [ ] Implement `execute()` with your logic
- [ ] Register in `registry.py`

### Testing
- [ ] Run `framework chat`
- [ ] Test basic functionality
- [ ] Verify error handling
- [ ] Check logs for issues

### Optimization
- [ ] Add `_create_orchestrator_guide()`
- [ ] Add `_create_classifier_guide()`
- [ ] Improve `get_access_details()`
- [ ] Add comprehensive error handling

---

## 🐛 Quick Troubleshooting

| Error | Solution |
|-------|----------|
| Context type X not found | Add `ContextClassRegistration` in `registry.py` |
| Capability X not found | Add `CapabilityRegistration` in `registry.py` |
| Module not found | Check file exists and class name matches |
| LLM doesn't use capability | Improve description and add orchestrator guide |
| Wrong Python code generated | Improve `get_access_details()` in context class |

---

## 💡 Best Practices

### ✓ DO
- Use clear, descriptive names
- Write good descriptions for LLM consumption
- Implement error handling
- Add logging throughout
- Test incrementally
- Study the provided examples

### ✗ DON'T
- Create god capabilities that do everything
- Use circular dependencies
- Hardcode configuration (use config.yml)
- Skip error handling
- Ignore logs during development

---

## 📚 File Reference

| File | Purpose | Key Sections |
|------|---------|--------------|
| `context_classes.py` | Define data structures | CONTEXT_TYPE, fields, get_access_details() |
| `capabilities/*.py` | Implement logic | @capability_node, provides/requires, execute() |
| `registry.py` | Register components | CapabilityRegistration, ContextClassRegistration |
| `config.yml` | Configuration | Models, API keys, services |

---

## 🔗 Quick Links

- **Complete Examples**: See `example_capability.py.j2` and `context_classes.py`
- **Working Apps**: Generate with `osprey init --template hello_world_weather`
- **Framework Docs**: https://als-apg.github.io/osprey/
- **API Reference**: https://als-apg.github.io/osprey/api_reference/

---

**Ready to integrate? Start with `example_capability.py.j2` - it has everything you need!**

