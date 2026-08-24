---
summary: Artifact gallery usage and reuse rules
description: How to create, save, and reuse artifacts in the OSPREY gallery
---

# Artifacts

## Reuse First

When the user references previous work or wants to act on it (log it, share it,
re-analyze it), call `artifact_list()` BEFORE creating anything new.
Use `tool=` or `category=` to narrow results. Reuse existing artifact IDs
rather than recreating content.

## Creating Artifacts

Two ways in, chosen by where the thing you are saving lives.

### A live Python object — `save_artifact()` inside `execute`

Plots, DataFrames, dicts of computed numbers: anything that exists only in
the running process. `save_artifact(obj, title, description)` is a function
in the exec namespace; it serializes the object, detecting the type from the
object itself:

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

# Computed results as JSON — a dict saves as json
save_artifact({"tune_x": 0.21, "tune_y": 0.31}, title="Tunes",
              artifact_type="json", category="lattice_analysis")
```

`artifact_type=` overrides the detected type; `category=` groups it in the
gallery. Results computed in code are saved from that code — never copied
into a tool call afterwards.

### A file on disk, or text you already have — `artifact_register` MCP tool

Screenshots, a CSV or PDF a tool left on disk, a markdown summary written in
your reply. Nothing is serialized; the content is stored as given:

- `file_path` — register a file already on disk
- `content` + `content_type` — store literal text; `content_type` is required
  and is one of `markdown`, `html`, `text`, `json`
- `category` — optional gallery category for grouping. See the type registry for valid categories.

`artifact_register` never runs Python and cannot take a live object.

## Notebook Artifacts

Every `execute` call automatically creates a Jupyter notebook artifact
containing the code, stdout, and stderr. These notebooks appear in the gallery
and can be viewed with rendered HTML formatting.

- **Auto-created:** Every execution is saved as a `.ipynb` notebook artifact
- **Pre-execution review:** When approval is required, a pre-execution notebook
  is created and linked in the approval prompt for code review
- **Editable:** Use `NotebookEdit` to modify notebook cells in the `artifacts/`
  directory under the agent-data root (`agent_data.base_dir` in config.yml) —
  the gallery re-renders automatically

## Directing User Attention

After creating an artifact, call `artifact_focus(artifact_id)` to select it in
the gallery so the user sees it immediately. The gallery will scroll to the
artifact and show its preview.

The user's current gallery selection is automatically included in your context
via the `UserPromptSubmit` hook (reads `focus_state.txt`). Use this to
understand what the user is looking at.

## Subagent Hand-Backs

A subagent that files its answer returns a pointer, not the answer:
`**Results** (artifact_id: …)` (or `**Channels found**`), a headline of a few
sentences, and the identifiers the question asked for. The artifact holds the
full answer. Call `artifact_focus(artifact_id)` on it so the user sees the
whole thing in the gallery, and relay the headline — do not re-type the
artifact's tables or listings into your reply. Read it with
`artifact_read(artifact_id)` only when the next step needs its detail.

## Math in Markdown Artifacts

The gallery renders LaTeX math via KaTeX. Use `$...$` for inline math and
`$$...$$` for display equations. **Never use code blocks for equations** — they
render as monospaced plain text without typesetting.

## Best Practices

- Use descriptive titles — they're the primary identifier in the gallery
- Add descriptions for context (what analysis produced this, what it shows)
- Use `save_artifact()` in Python for computed outputs (plots, tables)
- Use `artifact_register` for screenshots, written summaries, and files already on disk
- Use `artifact_focus` to direct the user's attention to a specific artifact
- Use `NotebookEdit` to refine notebook cells before sharing
