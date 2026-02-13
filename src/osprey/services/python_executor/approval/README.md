# Human-in-the-Loop Approval Subsystem

This subsystem provides human oversight capabilities for Python code execution, integrating with LangGraph's interrupt system for production-ready approval workflows.

## Components

### LangGraph Node
- **`node.py`**: Approval LangGraph node
  - Creates approval interrupts
  - Provides rich approval context
  - Links to review notebooks
  - Handles approval/rejection
  - Resumes workflow on approval

## Approval Flow

```
Analysis Result
(requires_approval=True)
         ↓
┌────────────────────────┐
│  Create Interrupt      │
│  - Approval context    │
│  - Notebook link       │
│  - Safety assessment   │
└───────────┬────────────┘
            ↓
    LangGraph Interrupt
    (waits for human)
            ↓
┌────────────────────────┐
│  User Reviews:         │
│  - Analysis report     │
│  - Generated code      │
│  - Safety assessment   │
└───────────┬────────────┘
            ↓
    User Decision
    (yes / no)
            ↓
    ┌───────┴────────┐
    │                │
┌───▼───┐      ┌────▼─────┐
│ Approve│      │  Reject  │
└───┬───┘      └────┬─────┘
    │               │
    ↓               ↓
Resume          Cancel
Execution      Workflow
```

## Usage

### Via LangGraph Node
```python
from osprey.services.python_executor.approval import create_approval_node

# Create node
approval_node = create_approval_node()

# Add to graph
graph.add_conditional_edges(
    "analyzer",
    lambda state: "approval" if state.requires_approval else "executor",
    {
        "approval": "approval_node",
        "executor": "executor_node"
    }
)
```

### Approval Interrupt Structure
```python
# What users see during approval
"""
⚠️ HUMAN APPROVAL REQUIRED ⚠️

Task: Adjust EPICS beam current setpoints
Execution Mode: EPICS_CONTROL

Python code requires human approval.

📓 Review Code: [Open Jupyter Notebook](http://jupyter/notebook.ipynb)

Safety Assessment:
  • Detected EPICS control operations
  • Code modifies beam parameters
  • Execution mode: EPICS_CONTROL

To proceed:
  • Type 'yes' to approve execution
  • Type 'no' to cancel
"""
```

## Configuration

### Approval Policies
```yaml
osprey:
  execution:
    approval:
      # Which execution modes require approval
      modes_requiring_approval:
        - "WRITE_ENABLED"
        - "EPICS_CONTROL"

      # Auto-approve for certain users/contexts
      auto_approve:
        users: []  # Admin users
        environments: ["development"]
```

## Integration with Executor Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│Generator │ →  │ Analyzer │ →  │ Approval │ →  │ Executor │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                      │               ↑
                      │  requires_    │  user
                      │  approval     │  approves
                      └───────────────┘
```

## Testing

```bash
# Test approval workflow
pytest tests/services/python_executor/approval/

# Test approval node
pytest tests/services/python_executor/approval/test_node.py
```

## See Also

- [Approval Node](node.py) - Main approval implementation
- [Parent Module](../__init__.py) - Main Python executor exports
- [Analysis Subsystem](../analysis/) - Generates approval requirements
- [Executor Service](../service.py) - Complete pipeline integration
