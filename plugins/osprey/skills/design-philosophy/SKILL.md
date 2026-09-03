---
name: design-philosophy
description: >-
  OSPREY's design and architecture principles — the rules that define how OSPREY code should be
  structured, and the anti-pattern each one prevents. Consult before designing, adding, or reviewing
  any feature (connectors, MCP servers, providers, capabilities, CLI, registry, runtime, config),
  and when redirecting work that is drifting from these principles. Trigger when the user says "is
  this the right direction", "does this fit OSPREY", "this feels wrong", "check the design
  philosophy", or points an agent at how OSPREY code should look. Apply before adding a config knob,
  a new abstraction, a new subsystem, or anything touching hardware-write safety.
---

# OSPREY Design Philosophy

> **Working draft.** Principles are still being collected and refined with the maintainer. The seven
> below are confirmed. Each states a rule, its rationale, and the guidance an agent should apply.
> Principles are written generically: do not hard-code lists of current features or subsystems into
> this document — they go stale. Refer to "existing peer subsystems" and let the reader inspect the
> code.

OSPREY runs agentic AI in safety-critical control systems (accelerators, fusion experiments,
beamlines), where an incorrect hardware write can damage equipment or endanger people. These
principles exist to keep the framework trustworthy on real machines and adaptable as the field
changes. Apply judgment: principles guide decisions, they are not mechanical rules to satisfy.

---

## 1. The safe state is the default

The cautious behavior is what happens unless configuration explicitly enables the riskier one.

- Hardware writes are disabled until config opts in; validation fails closed (e.g. an empty limits
  database blocks every write rather than allowing them); when configuration cannot be read, assume
  the safe path, not the convenient one.
- Wire safety in structurally, not per-call, so it cannot be forgotten. New connectors inherit the
  writes-enabled guard automatically via `ControlSystemConnector.__init_subclass__`
  (`src/osprey/connectors/control_system/base.py`).
- A guard that currently never triggers is not dead code. It documents an invariant and protects
  against the day the assumption changes. Do not remove it.

## 2. Nothing facility-specific belongs in the core

Code in `src/osprey` must run unchanged at a different facility. Anything tied to one accelerator,
detector, or beamline lives behind a connector, in config, or in a preset/template — never in the
framework.

- Test each change against: "Would this be wrong at a different facility?" If yes, it is in the
  wrong layer.
- Facility values go in config; protocol differences go in connectors; facility narrative and agent
  wiring go in presets and templates.
- Do not create domain-prefixed sibling artifacts to hold facility variations. Fold the variation
  into the relevant preset or configuration and reuse the generic components.

## 3. Reach for symmetry, measured

When building a new subsystem, follow the structure that existing peer subsystems already use before
inventing a new one. Consistency lowers design cost, shortens the learning curve, and makes later
refactoring tractable because subsystems resemble each other.

- Look across the codebase first. If peers integrate through a subagent, an MCP server, and a
  service layer, a new subsystem of the same kind should do likewise unless it genuinely differs.
- This is a default, not a mandate. Divergence is allowed when a feature does not fit the existing
  shape, but it must be justified, not assumed. Forcing an ill-fitting feature into the common mold
  is as wrong as inventing a new pattern needlessly.
- This principle and Principle 4 both serve future changeability and can pull against each other —
  symmetry favors resembling neighbors, swappability favors not entangling with them. Balance them.

## 4. Keep components swappable

Separate a feature from the dependencies it relies on so either can be replaced independently. In a
fast-moving field, tight coupling to a volatile dependency is technical risk, not convenience.

- Isolate the parts most likely to change — the model, the agent harness, external MCP/protocol
  standards — behind a boundary (interface, adapter, or config). Do not reference them inline
  throughout the codebase.
- **Target state:** the agent harness is a replaceable dependency. The current code does not yet meet
  this. New work moves toward it, not away from it.

## 5. A user-facing feature isn't done until it's discoverable

If a change alters what an operator or deployer sees or does, the user-facing surface ships with it: a
docs how-to, CLI `--help` text, and a changelog fragment in `changelog.d/`. Code that works but cannot
be found is incomplete, not done.

- Test each change against: "Could a user discover and use this without reading the source?" If no,
  the feature is unfinished.
- Match the documentation shape peers already use — if comparable features have a how-to page, this
  one does too.
- Internal-only or framework-internal changes are exempt; the bar is reader-facing impact, not line
  count.

## 6. Every fact has one producer

A datum — which channels exist, which providers are registered, what a limit is — is produced in
exactly one place, and every other consumer derives from that place rather than keeping its own
enumeration. Parallel lists do not stay in sync; they diverge silently and the divergence surfaces
as a wrong answer on a real machine, not as a failing test.

- A projection is not a source. A file or table derived from the authoritative datum (a limits
  export, a cache, a generated manifest) answers the question it was built for and nothing else.
  Reading it back as if it were the original is a bug even when the two currently agree.
- Test each change against: "If this list and its source disagree tomorrow, which one is wrong, and
  would anyone find out?" If the answer is not immediate, derive instead of enumerating.
- Deriving costs a lookup; duplicating costs a silent inconsistency. Prefer the lookup, and where a
  cached copy is genuinely needed, make its derivation explicit rather than re-deriving by hand.

## 7. Keep the maintained surface small

Every example, preset, demo, and doc page OSPREY ships is a promise to keep it working. A surface
the development loop does not exercise day to day rots silently: its config drifts, its docs lie,
and its tests pin yesterday's behavior. So the set of shipped surfaces must stay small enough that
the ordinary development cycle touches all of them.

- Before adding a parallel artifact (a demo preset, a second example stack, a tutorial variant),
  fold the demonstration into a surface that is already exercised, with its tests, its docs, and
  its CI lane, rather than shipping a lighter clone beside it.
- If a dedicated artifact is genuinely needed, wire it into CI and the docs in the same change that
  ships it. An untested example is a liability, not an asset.
- Test each change against: "Which existing workflow will exercise this next month?" If none, either
  wire one up or do not ship it.
- When two shipped surfaces drift toward duplicating each other, merge them and delete the
  redundant one. Deleting a surface nobody's workflow exercises is a feature, not a loss.

---

## How to apply

When a feature feels wrong but the reason is hard to name, identify which principle it violates and
state it plainly: name the principle, point at the specific drift, and propose the change that brings
the work back in line. The questions behind the principles: Is the unsafe path harder to reach than
the safe one? Would this be wrong at another facility? Does this follow the shape of its peers? Can
this dependency be swapped later without a rewrite? Could a user discover and use this without
reading the source? Does this fact have exactly one producer? Which existing workflow will exercise
this next month?
