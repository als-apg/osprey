---
name: demo-ui
description: >
  Run a short, scripted demonstration of the agent driving the web workspace —
  switching and reshaping panel tabs, creating an artifact and focusing it,
  composing a named layout — so an audience watches the agent and the UI move
  together. Use this whenever someone asks for a demo, walkthrough, tour, or
  showcase of the web terminal, the panels, or the workspace; wants to "show
  what the agent can do" to a visitor, a new operator, or a review committee;
  or is rehearsing a presentation — even when they never say the word "demo".
summary: Scripted UI demos — panel choreography, artifacts, layouts
---

# Demo UI — Agent-Drives-the-Workspace Showcase

The point of these demos is not the content the agent produces. It is the **visible
coupling**: the operator says something in the chat, and the workspace around the
chat reacts — a tab activates, the tab bar reshapes, the gallery jumps to a new
figure, the activity indicator flashes on the surface being touched.

An audience only sees that coupling if they are looking at the right place when it
happens. Everything below is built around that.

## Pick a workflow

Ask which one, unless the request already names it or implies it:

| # | Workflow | Runs | Shows |
|---|----------|------|-------|
| 1 | **Panel tour** | ~90 s | The tab bar as something the agent plays, not a fixed frame |
| 2 | **Artifact drop** | ~90 s | Agent makes a thing → workspace jumps to it → activity flashes |
| 3 | **Layout switch** | ~60 s | Task-shaped layouts composed from the same primitive a human's click uses |
| 4 | **Grand tour** | ~4 min | 1 → 2 → 3 back to back, with a recap |

"Give me the quick one" → Panel tour. "Show them everything" → Grand tour.
For a *wide* spread of artifact rendering paths (Plotly, matplotlib, LaTeX,
tables) use the **demo-gallery** skill instead — this skill deliberately makes
only one or two artifacts, because here the artifact is a prop for the handoff.

## The rhythm every workflow follows

**Say it, then do it.** One short line naming the move goes out *before* the tool
call, so the audience's eyes are on the tab bar when it changes. A move that lands
silently reads as a glitch.

**One move per beat.** Resist batching three panel calls into one turn — the
audience cannot follow three simultaneous changes, and the demo's whole claim is
that each step is legible.

**Narrate what it means, not what it is.** "Switching to CHANNELS — this is where
I search the channel database" beats "calling switch_panel with panel_id
channel-finder." The audience is watching a colleague work, not reading a log.

**Leave the workspace as you found it.** Record the active panel and which tabs are
visible at the start, and restore them at the end. Demos get run twice in a row,
and often on someone's real working session — a demo that rearranges an operator's
layout and walks away is a demo they won't let you run again.

**Adapt to the deployment.** Panel IDs vary — `lattice` or `okf` may not exist here.
Always read the live inventory first and demo the panels this deployment actually
has. Never guess an ID; a failed call mid-demo is the one error the audience *will*
notice.

## Before starting

Call `list_panels`. It returns the active panel, every panel with its `visible`
flag, and any `presets` (named layouts) the deployment defines. This is both the
precondition check and the script you are about to improvise against.

If it reports the Web Terminal is not running, say so plainly and stop — these
demos have nothing to show without it. Do not narrate moves that aren't landing.

---

## 1. Panel tour

Open by saying what the workspace is: a chat with panels docked around it, and the
agent can reach every one of them.

1. **Inventory** — from `list_panels`, name the panels that are here and what each
   is for, in one line each. Say which one is active now.
2. **Switch** — `switch_panel` through two or three of them, pausing on each with a
   sentence about what an operator uses it for. Choose panels that look
   different from each other; three tabs of similar-looking text is a dull tour.
3. **Reshape** — `hide_panel` one tab, then `show_panel` it back. Say what this is
   for: the tab bar is not a fixed frame, it is the set of surfaces relevant to
   what you are doing right now, and it can be trimmed to fit the task.
4. **Restore** — `switch_panel` back to whatever was active at the start, and
   confirm the workspace is as it was.

## 2. Artifact drop

Open by framing the handoff: the agent computes something, and it appears in the
workspace as a real object the operator can open, keep, and come back to.

1. **Make one figure** — `create_interactive_plot` with a compact, visually clear
   Plotly figure. Accelerator-physics content fits the setting (a beam profile, an
   orbit, a tune scan); a single well-labelled panel beats a dense multi-panel grid
   on a projector. Call `save_artifact(fig, "Title")` at the end of the code.
2. **Focus it** — `artifact_focus` on the returned artifact id. Point out that the
   gallery jumped and the activity indicator flashed on the workspace surface: the
   UI is showing the audience where the agent's attention is.
3. **Pin it** — `artifact_pin` and say why an operator would: pinned artifacts stay
   at the top through a long shift.
4. **Add a short note** — `artifact_save` with `content_type: "markdown"`, a few
   lines interpreting the figure, with one piece of inline math so the KaTeX
   rendering shows. Keep it to a paragraph and a short list; this is a companion
   to the plot, not a report.

## 3. Layout switch

Open with the idea: a layout is a task. "Machine setup" and "logbook review" want
different surfaces on screen.

1. Read `presets` from `list_panels`.
2. **If the deployment defines presets** — pick one, name it, and apply it by
   composing `show_panel` / `hide_panel` so the member panels are visible and the
   others are not. Say the line that lands: this is exactly what a human's click on
   that layout resolves to — same primitive, same result, just reached by asking.
3. **If it defines none** — compose a plausible one from the panels that exist
   (e.g. logbook review = the logbook panel plus the workspace) and mention the
   deployment can name it in `config.yml` under `web.panels.presets` so it becomes
   one click for everyone.
4. **Restore** the starting visibility set.

## 4. Grand tour

Run 1 → 2 → 3 in order with a one-sentence bridge between them, then close with a
short recap of what the audience saw the UI do — not a list of tools called, but
three or four plain statements ("the agent moved between panels", "it produced a
figure and the workspace opened it", "it reshaped the tab bar to match a task").
Restore the starting state once, at the very end.

---

## Optional beat — the approval gate

Only when the operator explicitly asks to show the human-in-the-loop gate, and only
on a deployment where writes are armed. Ask before running it, every time.

A write demo is a real machine action — it is the one beat here with consequences.
Pick a channel the operator names, let the approval prompt appear, and let *them*
answer it. That prompt, appearing unbidden in front of the audience, is the whole
point; never pre-approve or bypass it to keep the demo moving.

If writes are not armed, say so — "this deployment is read-only, so the write would
be refused before it reached the machine" is itself a good thing for an audience to
hear — and skip the beat.

## Anti-patterns

- **Silent moves.** A tool call with no line before it. The audience misses it and
  the demo lands as "nothing happened."
- **Guessed panel IDs.** Always from `list_panels`. A demo that opens with an error
  never recovers.
- **Walking away from a rearranged workspace.** Restore before you finish.
- **Turning it into demo-gallery.** One or two artifacts here. Four is the other
  skill's job.
- **Log-reading narration.** Tool names and parameters in the spoken line.
- **Padding.** These are 60–90 second pieces. If a workflow is running long, cut a
  panel from the tour rather than talking faster.
