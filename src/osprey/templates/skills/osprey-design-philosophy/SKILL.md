---
name: osprey-design-philosophy
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

> **Working draft.** Principles are still being collected and refined with the maintainer. The five
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
docs how-to, CLI `--help` text, and a changelog entry. Code that works but cannot be found is
incomplete, not done.

- Test each change against: "Could a user discover and use this without reading the source?" If no,
  the feature is unfinished.
- Match the documentation shape peers already use — if comparable features have a how-to page, this
  one does too.
- Internal-only or framework-internal changes are exempt; the bar is reader-facing impact, not line
  count.

---

## How to apply

When a feature feels wrong but the reason is hard to name, identify which principle it violates and
state it plainly: name the principle, point at the specific drift, and propose the change that brings
the work back in line. The questions behind the principles: Is the unsafe path harder to reach than
the safe one? Would this be wrong at another facility? Does this follow the shape of its peers? Can
this dependency be swapped later without a rewrite? Could a user discover and use this without
reading the source?
