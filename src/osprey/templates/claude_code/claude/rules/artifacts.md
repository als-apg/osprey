---
summary: Artifact gallery usage and reuse rules
description: How to create, save, and reuse artifacts in the OSPREY gallery
---

# Artifacts

## Reuse First

When the user references previous work or wants to act on it (log it, share it,
re-analyze it), call `data_list()` BEFORE creating anything new.
Use `tool_filter=` or `category_filter=` to narrow results. Reuse existing artifact IDs
rather than recreating content.

## Creating Artifacts

### Inside `execute` — use `save_artifact()`

A `save_artifact(obj, title, description)` function is available in the exec
namespace. It auto-detects the object type:

```python
# Plotly interactive plot
import plotly.express as px
fig = px.scatter(df, x='time', y='current')
save_artifact(fig, title="Beam Current Trend")

# Matplotlib figure
import matplotlib.pyplot as plt
plt.plot(x, y)
save_artifact(plt.gcf(), title="Orbit Distortion")

# DataFrame table
save_artifact(df.describe(), title="BPM Statistics")

# Markdown or HTML string
save_artifact("<h1>Report</h1><p>All clear.</p>", title="Shift Summary")
```

### Standalone — use `artifact_save` MCP tool

Register existing files or inline content without running Python:

- `file_path` — register a screenshot, CSV, or any file already on disk
- `content` + `content_type` — pass markdown/HTML/text/JSON directly

## Notebook Artifacts

Every `execute` call automatically creates a Jupyter notebook artifact
containing the code, stdout, and stderr. These notebooks appear in the gallery
and can be viewed with rendered HTML formatting.

- **Auto-created:** Every execution is saved as a `.ipynb` notebook artifact
- **Pre-execution review:** When approval is required, a pre-execution notebook
  is created and linked in the approval prompt for code review
- **Editable:** Use `NotebookEdit` to modify notebook cells in
  `osprey-workspace/artifacts/` — the gallery re-renders automatically

## Directing User Attention

After creating an artifact, call `artifact_focus(artifact_id)` to select it in
the gallery so the user sees it immediately. The gallery will scroll to the
artifact and show its preview.

The user's current gallery selection is automatically included in your context
via the `UserPromptSubmit` hook (reads `focus_state.txt`). Use this to
understand what the user is looking at.

## Best Practices

- Use descriptive titles — they're the primary identifier in the gallery
- Add descriptions for context (what analysis produced this, what it shows)
- Use `save_artifact()` in Python for computed outputs (plots, tables)
- Use `artifact_save` tool for screenshots, summaries, and file registration
- Use `artifact_focus` to direct the user's attention to a specific artifact
- Use `NotebookEdit` to refine notebook cells before sharing
