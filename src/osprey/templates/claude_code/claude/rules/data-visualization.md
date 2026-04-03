# Data Visualization

Use `save_artifact(obj, title, description)` to register plots in the artifact
gallery. This helper is available in all execution environments (`execute`,
`create_static_plot`, `create_interactive_plot`). It auto-detects the object
type: matplotlib Figure, Plotly Figure, DataFrame, or string (HTML/markdown).

### Matplotlib

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("x-label")
ax.set_ylabel("y-label")
save_artifact(fig, "Title")
```

### Plotly

```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y))
fig.update_layout(title="Title", xaxis_title="x", yaxis_title="y")
save_artifact(fig, "Title")
```

### Sandbox vs Execute

The workspace visualization tools (`create_static_plot`, `create_interactive_plot`,
`create_dashboard`) run in a sandboxed environment. They auto-import numpy, pandas,
scipy, and their respective visualization library. Additional imports (including `at`)
are allowed — see each tool's description for details.

Use the `execute` tool when code needs EPICS channel access or other packages
not in the sandbox whitelist.
