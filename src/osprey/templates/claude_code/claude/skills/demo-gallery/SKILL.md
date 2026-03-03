---
name: demo-gallery
description: >
  Generate a showcase set of diverse artifacts to demonstrate the OSPREY Artifact
  Gallery. Produces interactive Plotly plots, matplotlib figures, a LaTeX-rich
  markdown report, and a computed data table — all in parallel. Use for demos,
  onboarding, or verifying that the gallery pipeline works end-to-end.
summary: Showcase demo artifacts for the Artifact Gallery
---

# Demo Gallery — Artifact Showcase Generator

Generate a diverse set of high-quality artifacts to demonstrate every capability of the OSPREY Artifact Gallery. The artifacts should be visually striking, scientifically themed, and exercise all rendering paths.

Follow these two phases in order.

---

## Phase 1 — Generate Artifacts (in parallel)

Launch **all four artifact groups below in parallel**. Each group uses a different creation pathway to showcase the gallery's full range.

### 1a. Interactive Plotly Figure — `create_interactive_plot`

Use the `create_interactive_plot` MCP tool to produce a **multi-panel Plotly figure** with at least two subplots. The plot should be visually rich and interactive.

**Suggested content** (pick one or combine creatively):

- Damped harmonic oscillator with envelope + phase-space portrait
- Synchrotron radiation spectrum (intensity vs photon energy) + cumulative flux
- Particle beam phase space (x-x', y-y') with color-coded amplitude
- Resonance diagram with tune footprint overlay

Use `plotly.subplots.make_subplots` for multi-panel layout. Apply a clean template (`plotly_white` or `plotly_dark`). Add descriptive axis labels and a figure title. Call `save_artifact(fig, "Title")` at the end.

### 1b. Matplotlib Static Plot — `execute`

Use the `execute` MCP tool to create a **publication-quality matplotlib figure** and save it as an artifact. The figure should demonstrate matplotlib's strengths (colormaps, contours, annotations).

**Suggested content** (pick one):

- 2D heatmap of a beam distribution in phase space with contour overlay
- Twiss parameter evolution ($\beta_x$, $\beta_y$, $\eta_x$) along a beamline
- Frequency map analysis colored by diffusion rate
- Mountain range plot of bunch profiles over multiple turns

Use `plt.subplots()`, apply `plt.style.use("seaborn-v0_8-whitegrid")` or similar, and call `save_artifact(fig, "Title", "description")`.

### 1c. Markdown Report with LaTeX — `artifact_save`

Use the `artifact_save` MCP tool to create a **rich markdown document** that exercises KaTeX rendering and table formatting.

The report MUST include:

1. **A title and introduction** with inline math (e.g., $E = \gamma m_0 c^2$)
2. **A data table** with numeric values (at least 4 rows, 4 columns)
3. **Display equations** using `$$...$$` blocks — at least two, such as:
   - Hill's equation: $x''(s) + K(s)\, x(s) = 0$
   - Beam emittance: $\epsilon = \sqrt{\langle x^2 \rangle \langle x'^2 \rangle - \langle x x' \rangle^2}$
   - Synchrotron radiation critical energy: $E_c = \frac{3}{2} \frac{\hbar c \gamma^3}{\rho}$
4. **Multiple heading levels** (H1, H2, H3)
5. **A bulleted list** summarizing key observations

Use `content_type: "markdown"`. Give it an informative title like "Accelerator Physics Reference Card" or "Beam Dynamics Summary".

### 1d. Computed Data Table — `execute`

Use the `execute` MCP tool to **compute a numeric result** and save a markdown table as an artifact.

**Suggested content:**

- Generate a lattice element table with columns: Element, Type, Length [m], K1 [1/m²], Angle [mrad]
- Compute a parameter scan (e.g., tune vs. quadrupole strength) and present it as a table
- Calculate beam parameters at different energies and tabulate them

Use `save_artifact(summary_string, "Title", "description")` where the string is a formatted markdown table. Include descriptive column headers and realistic numeric values.

---

## Phase 2 — Focus & Confirm

After all artifacts are created:

1. **Focus the most visually striking artifact** (usually the Plotly figure) using `artifact_focus`
2. **List all created artifacts** in a summary table for the user:

| # | Title | Type | Pathway |
|---|-------|------|---------|
| 1 | ... | Interactive plot | `create_interactive_plot` |
| 2 | ... | Static plot | `execute` + matplotlib |
| 3 | ... | Markdown report | `artifact_save` |
| 4 | ... | Data table | `execute` + `save_artifact()` |

3. **Note** that additional notebook artifacts were auto-generated from each `execute` call

---

## Content Guidelines

- **Domain**: Use accelerator physics, beam dynamics, or synchrotron science content. This fits the OSPREY context and produces visually interesting results.
- **Data**: Generate synthetic data with `numpy`. Use realistic parameter ranges (e.g., beam energies 1-8 GeV, tunes 10-30, beta functions 1-30 m).
- **Quality**: Plots should look publication-ready. Use proper axis labels, units, legends, and colorbars where appropriate.
- **Variety**: Each artifact should look distinctly different — vary color schemes, plot types, and content.
- **LaTeX**: Use proper accelerator physics notation. The gallery renders LaTeX via KaTeX — use `$...$` for inline and `$$...$$` for display math. Inside Python string literals in `execute`, escape backslashes (`\\\\`) for LaTeX commands.

## Anti-Patterns

Do NOT:
- Create all artifacts sequentially — Phase 1 groups must run in parallel
- Use trivial or placeholder content — make it visually impressive for demos
- Skip the LaTeX in the markdown report — math rendering is a key gallery feature
- Create more than 4-5 artifacts total (excluding auto-notebooks) — keep it focused
- Use the same color palette across all plots — vary the visual style
- Skip `artifact_focus` — the demo should open to the most striking visual
