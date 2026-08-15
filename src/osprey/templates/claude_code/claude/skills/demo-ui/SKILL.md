---
name: demo-ui
description: >
  Run a short, scripted demonstration of the agent driving the web workspace —
  swapping which panel holds the tile, creating an artifact and focusing it,
  composing a named layout on the panel rail — so an audience watches the agent
  and the UI move together. Use this whenever someone asks for a demo,
  walkthrough, tour, or showcase of the web terminal, the panels, or the
  workspace; wants to "show what the agent can do" to a visitor, a new
  operator, or a review committee; or is rehearsing a presentation — even when
  they never say the word "demo".
summary: Scripted UI demos — panel choreography, artifacts, layouts
---

# Demo UI — Agent-Drives-the-Workspace Showcase

The point of these demos is not the content the agent produces. It is the **visible
coupling**: the operator says something in the chat, and the workspace around the
chat reacts — the tile swaps to a new panel, the gallery jumps to a new figure, and
agent-driven moves that touch a rail entry (switch, show, register) flash a brief
accent-colored glow on it — the UI's own proof that the agent, not the presenter,
made the move. A hidden panel's entry simply vanishes, no glow.

An audience only sees that coupling if they are looking at the right place when it
happens. Everything below is built around that.

## The workspace, in the words you'll narrate

The left **rail** is a membership list: an entry exists only while its panel is a
member; an off-rail panel waits in the "+" catalog. The workspace shows **one
panel per tile**, each with its own header bar.

Two axes, and the verb names which one it moves. `open_panel` puts a panel on
screen — the tile appears, the active marker moves on the rail, and the panel it
replaced stays one rail-click away with its state intact; `close_panel` takes the
tile away again and leaves the entry. `add_panel_to_rail` and
`remove_panel_from_rail` move membership instead: adding brings the entry back
with a glow but puts nothing on screen, and removing takes the entry away (and
with it any open tile — an unlaunchable panel is not left stranded). The chat is
the SESSION tile, operator-only — never script the agent driving it.

## Pick a workflow

Ask which one, unless the request already names it or implies it:

| # | Workflow | Runs | Shows |
|---|----------|------|-------|
| 1 | **Panel tour** | ~90 s | The rail as something the agent plays, not a fixed frame |
| 2 | **Artifact drop** | ~90 s | Agent makes a thing → workspace surfaces it → the gallery jumps |
| 3 | **Layout switch** | ~60 s | Task-shaped layouts composed from the same primitive a human's click uses |
| 4 | **Grand tour** | ~4 min | 1 → 2 → 3 back to back, with a recap |

"Give me the quick one" → Panel tour. "Show them everything" → Grand tour.
For a *wide* spread of artifact rendering paths (Plotly, matplotlib, LaTeX,
tables) use the **demo-gallery** skill instead — this skill deliberately makes
only one or two artifacts, because here the artifact is a prop for the handoff.

## The rhythm every workflow follows

**Say it, then do it.** One short line naming the move goes out *before* the tool
call, so the audience's eyes are on the rail and the tile when they change. A move
that lands silently reads as a glitch.

**One move per beat.** Resist batching three panel calls into one turn — the
audience cannot follow three simultaneous changes, and the demo's whole claim is
that each step is legible.

**Narrate what it means, not what it is.** "Switching to CHANNELS — this is where
I search the channel database" beats "calling open_panel with panel_id
channel-finder." The audience is watching a colleague work, not reading a log.

**Leave the workspace as you found it.** Record the active panel and the rail
membership at the start, and restore them at the end. Demos get run twice in a
row, and often on someone's real working session — a demo that rearranges an
operator's layout and walks away is a demo they won't let you run again.

**Adapt to the deployment.** Panel IDs vary — `lattice` or `okf` may not exist here.
Always read the live inventory first and demo the panels this deployment actually
has. Never guess an ID; a failed call mid-demo is the one error the audience *will*
notice.

## Before starting

Call `list_panels`. It returns the active panel, every panel with its `visible`
flag (visible = has a rail entry), and any `presets` (named layouts) the
deployment defines. This is both the precondition check and the script you are
about to improvise against.

Notice which mode the audience screen is in: Expert shows the full workspace;
Simple may still be chat-only — that changes the Artifact drop's reveal beat.

If it reports the Web Terminal is not running, say so plainly and stop — these
demos have nothing to show without it. Do not narrate moves that aren't landing.

---

## 1. Panel tour

Open by saying what the workspace is: a chat with a rail of panels beside it, and
the agent can reach every one of them.

1. **Inventory** — from `list_panels`, name the panels that are here and what each
   is for, in one line each. Say which one holds the tile now.
2. **Switch** — `open_panel` through two or three of them, pausing on each with a
   sentence about what an operator uses it for. Tell the audience to watch the
   tile swap and the active marker move on the rail. Choose panels that look
   different from each other; three similar-looking pages is a dull tour.
3. **Reshape** — `remove_panel_from_rail` on one panel: its rail entry vanishes
   and its tile closes — say it now waits in the "+" catalog. Then
   `add_panel_to_rail` it back and point at the glow as its entry returns. The
   rail is the set of surfaces relevant to the task at hand, and it can be
   trimmed to fit.
4. **Restore** — `open_panel` back to whatever was active at the start, and
   confirm the workspace is as it was.

## 2. Artifact drop

Open by framing the handoff: the agent computes something, and it appears in the
workspace as a real object the operator can open, keep, and come back to.

1. **Make one figure** — `create_interactive_plot` with a compact, visually clear
   Plotly figure. Accelerator-physics content fits the setting (a beam profile, an
   orbit, a tune scan); a single well-labelled panel beats a dense multi-panel grid
   on a projector. Call `save_artifact(fig, "Title")` at the end of the code.
2. **Surface the gallery** — in Expert mode, `open_panel` on the artifacts panel
   so the gallery is on screen before you point at it. In Simple mode on a
   chat-only page, that same call *is* the beat: the
   workspace column arrives, already showing the gallery — the single strongest
   moment these demos have, so announce it first.
3. **Focus it** — `artifact_focus` on the returned artifact id. The gallery jumps
   to the figure and opens its preview. Off screen, the only cue is a one-line
   note in the bottom activity strip that clears in seconds — hence step 2 first.
4. **Pin it** — `artifact_pin` and say why an operator would: pinned artifacts stay
   at the top through a long shift.
5. **Add a short note** — `artifact_save` with `content_type: "markdown"`, a few
   lines interpreting the figure, with one piece of inline math so the KaTeX
   rendering shows. Keep it to a paragraph and a short list; this is a companion
   to the plot, not a report.

## 3. Layout switch

Open with the idea: a layout is a task. "Machine setup" and "logbook review" want
different surfaces on the rail.

1. Read `presets` from `list_panels`.
2. **If the deployment defines presets** — pick one, name it, and apply it by
   composing `add_panel_to_rail` / `remove_panel_from_rail` so the member panels are on the rail and
   the others are not — the audience watches entries appear and vanish, one glow
   at a time. Say the line that lands: this is exactly what a human's click on
   that layout resolves to — same primitive, same result, just reached by asking.
3. **If it defines none** — compose a plausible one from the panels that exist
   (e.g. logbook review = the logbook panel plus the workspace) and mention the
   deployment can name it in `config.yml` under `web.presets` so it becomes one
   click for everyone.
4. **Restore** the starting rail membership and active panel.

## 4. Grand tour

Run 1 → 2 → 3 in order with a one-sentence bridge between them, then close with a
short recap of what the audience saw the UI do — not a list of tools called, but
three or four plain statements ("the agent moved between panels", "it produced a
figure and the workspace opened it", "it reshaped the rail to match a task").
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
- **`add_panel_to_rail` as a reveal in Expert mode.** It only adds a rail entry;
  the audience sees content when `open_panel` follows.
- **Walking away from a rearranged workspace.** Restore before you finish.
- **Turning it into demo-gallery.** One or two artifacts here. Four is the other
  skill's job.
- **Log-reading narration.** Tool names and parameters in the spoken line.
- **Padding.** These are 60–90 second pieces. If a workflow is running long, cut a
  panel from the tour rather than talking faster.
