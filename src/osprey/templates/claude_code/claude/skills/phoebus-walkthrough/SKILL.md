---
name: phoebus-walkthrough
description: >
  Run a guided, end-to-end demonstration of OSPREY's native Phoebus interaction:
  list and perceive live Phoebus displays, snapshot a widget, drive a control via
  synthetic GUI events and the semantic PV path, and cross-check the effect through
  OSPREY's own EPICS connector. Use when the user asks to "demo Phoebus", "show the
  Phoebus capabilities", "walk through the bridge", or wants to see the
  perceive→snapshot→drive→readback loop. Requires the demo stack (soft IOC + a
  Phoebus product with the agent bridge) to be running.
summary: Guided demo of native Phoebus perceive + drive capabilities
---

# Phoebus Walkthrough — Native Interaction Demo

Demonstrate the full set of capabilities OSPREY has over a **live** Phoebus control
panel through the agent bridge. The headline: OSPREY can both **drive the GUI** the
way an operator would (synthetic events that run confirm dialogs and scripts) **and**
read/write the **same PVs directly** through its EPICS connector — and prove the two
agree.

The demo control system is a small soft IOC serving `DEMO:*` PVs (Setpoint, Readback,
Current, Enable, Valve, Status). The demo panel binds its widgets to those PVs.

Tools you will use:
- `phoebus_list_displays`, `phoebus_open_panel`, `phoebus_perceive`,
  `phoebus_perceive_region`, `phoebus_snapshot`, `phoebus_drive` (the `phoebus`
  MCP server)
- `channel_read`, `channel_write` (the `controls` MCP server, in `epics` mode against
  the same soft IOC)

**Targeting a Phoebus instance:** each running Phoebus process has its own MCP
server, and the server you call IS the target — every tool acts on the instance
its server is bound to. When a `phoebus2` server exists, the mapping is:

| User says | MCP server | Web-terminal tab | Bridge |
|---|---|---|---|
| "first window", "the panel", default | `phoebus` | PHOEBUS | 7979 |
| "second window", "2nd instance", "other Phoebus" | `phoebus2` | PHOEBUS2 | 7980 |

Displays, handles, and the "active" display are **per instance** — a handle from
`phoebus` means nothing to `phoebus2`. Both instances watch the same demo soft
IOC, so a direct PV write surfaces on BOTH panels; only GUI state (tabs, focus,
open displays) is independent. Run the walkthrough phases against instance 1
unless the user targets the second.

Follow the phases in order. Narrate each step to the user in plain language — this is a
*demonstration*, so explain what each capability is as you exercise it.

---

## Phase 1 — Preflight

Confirm the demo stack is up before driving anything.

1. Call `phoebus_list_displays`.
2. If it returns a `phoebus_unreachable` error, the Phoebus product (with the bridge)
   is not running. Tell the user to start the stack and stop:
   > "The demo stack isn't running. From the OSPREY repo, run
   > `bash demos/phoebus/run_demo.sh` (starts the soft IOC and Phoebus with the demo
   > panel), then ask me to run the walkthrough again."
   Do NOT try to launch Phoebus yourself — it is a desktop application.
3. If it returns a display list, confirm the demo display is present and note its name.
   Read `DEMO:Readback` and `DEMO:Status` with `channel_read` to confirm the IOC is
   reachable too. Report both checks succeeded.

---

## Phase 2 — Perceive

Show what OSPREY can *see*.

1. Call `phoebus_perceive` on the active display.
2. Summarise the widget tree for the user: the widget names/types, and for each
   widget with a PV, the PV name, value, and severity. Call out `Setpoint` (text
   entry), `SetButton` (action button), `Readback`, `Enable`, and `Status`.
3. Optionally call `phoebus_perceive_region` with a rectangle around the control
   cluster to show region-filtered perception.

Explain: perception walks the *live* JavaFX scene graph, so values, alarm severities,
and on-screen geometry are real-time — this is how the agent grounds itself before
acting.

---

## Phase 3 — Snapshot

Show what OSPREY can *capture*.

1. Call `phoebus_snapshot` for the `Setpoint` widget with `highlight=true`.
2. Use the **Read** tool on the returned file path to view the PNG, and describe what
   the highlighted widget looks like.

Explain: the snapshot is registered to screen coordinates (origin + scale headers), so
the agent can map image pixels back to the display.

---

## Phase 4 — Drive (synthetic) and verify two ways

This is the core capability. Drive the GUI, then verify via **both** paths.

1. Record the current `DEMO:Readback` value (`channel_read`).
2. Call `phoebus_drive` on `SetButton` with `verb="click"`, `mode="synthetic"`. This
   fires the button's real `ActionEvent` — exactly as an operator click — so the
   button's WritePV action runs.
3. Wait ~1s, then verify **two independent ways**:
   - `phoebus_perceive` again and read the new `Readback` widget value (the GUI's view).
   - `channel_read DEMO:Readback` (OSPREY's own EPICS view of the same PV).
4. Confirm both report the new value (the button writes 42) and that they agree.

Explain: synthetic drive proves the agent actuated the *real control*, not a back-door
PV poke — confirm dialogs, enable-gates, and bound scripts all ran. The two-way check
proves the GUI and the control system are consistent.

---

## Phase 5 — Drive (semantic) and contrast

Show the labelled fallback path and how it differs.

1. Call `phoebus_drive` on `Setpoint` with `verb="type"`, `value="7"`,
   `mode="semantic"`. Note the result `detail` is prefixed `[BYPASSED GUI]`.
2. Verify with `channel_read DEMO:Setpoint` / `DEMO:Readback`.

Explain the contrast: semantic mode writes the PV directly through the widget runtime,
**bypassing** the GUI chain (no confirm dialog / scripts). It is the labelled fallback
for widgets with no resolvable interactive control — always tagged so a semantic write
is never mistaken for a real operator action.

---

## Phase 6 — Direct control, no GUI

Close the loop by showing OSPREY operating the control system *without* the panel.

1. Use `channel_write` to set `DEMO:Setpoint` to a value ≥ 95.
2. `channel_read DEMO:Status` — show it derived `FAULT`.
3. `phoebus_perceive` once more — show the panel's `Status` widget now reflects `FAULT`
   too, i.e. a direct PV write is visible on the live GUI.

Explain: OSPREY is a first-class control-system client (EPICS) *and* a first-class GUI
driver (bridge) over the very same PVs.

---

## Phase 7 — Open a site panel on demand

Show that the agent can bring up a *new* display by logical name, not just use what
is already open. Run this phase **last among the interactive phases** — the newly
opened panel becomes the active tab, so earlier phases that address the demo display
would otherwise need explicit handles.

1. Check the project config for registered panels (`phoebus.panels.*` in config.yml).
   A site build profile registers these logical names, each mapping to a facility
   `.bob` display (e.g. a `site_overview` panel). If no site panel is registered,
   say so and skip this phase.
2. Call `phoebus_open_panel` with a registered name (e.g. `"site_overview"`). It
   returns a deterministic handle like `"handle:d-2"` once the display reports ready.
3. Call `phoebus_perceive` with `display=<that handle>` — summarise the scale (widget
   count, sections) rather than listing everything.
4. Point out the expected **disconnected** state of the panel's PVs: this is a real
   production display, but the demo Phoebus is deliberately isolated (Channel Access
   scrubbed to the loopback soft IOC), so real-machine PVs cannot resolve. The agent
   can *see* the panel; it physically cannot touch the machine. Frame this as the
   safety invariant working, not as an error.
5. Tell the user the new panel is now visible live in the PHOEBUS tab of the web
   terminal (it opened as a new tab in the streamed Phoebus window).
6. Do **not** drive anything on this panel.

Explain: open → handle → perceive is the full agent loop for displays it has never
seen; handles make targeting unambiguous even with several displays open.

---

## Phase 8 — Wrap up

Summarise for the user what was demonstrated, as a short bulleted recap:
- perceived a live display and its PVs,
- snapshotted a widget,
- drove a control via a synthetic GUI event and verified via both the GUI and the PV,
- contrasted the semantic (GUI-bypassing) path,
- showed a direct PV write surfacing on the panel,
- and opened a site display on demand by logical name, perceiving it by handle.

If the `session-report` skill is available, offer to log a session report.
