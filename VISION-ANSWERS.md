# Vision answers

Calibration record for `VISION.md`.
Keep this file next to the vision: it records which hypotheticals the draft was tested against, how each was answered, and what changed in the text as a result.

**Status: awaiting the author's sign-off.**
The `/vision` skill puts these verdicts to the author on a review board.
That board was built and launched, but its server is only reachable inside the container this ran in, so the eleven verdicts below were derived from merged history instead of answered by Thorsten Hellert.
Each verdict names the evidence it rests on.
Where the evidence does not settle a question, the entry says so rather than supplying an answer (see H-3).
Treat every line as a proposal until the author confirms or overrides it.

Evidence base: 90 merged pull requests (#442-#607), 9 read in full, plus
`src/osprey/templates/skills/osprey-design-philosophy/SKILL.md` and
`src/osprey/interfaces/design_system/DESIGN.md`.

---

## H-1 Closed-loop orbit correction on a shift approval

**Conditional.**
The draft said a person arms every write. The history does not: #594 starts a whole Bluesky queue on a single approval, and #574 describes the human's click as "the arming decision" for a queue, not for each write in it. What the practiced rule actually protects is that the human reads the full extent of what they are arming before it runs. A queue is enumerable in advance; a feedback loop whose next write depends on unseen state is not. #565 supports the same reading from the other side: `execute` stays gated specifically so the agent cannot bypass the queue with raw writes, which only matters if the queue is the legitimate arming unit.

## H-2 Estimated archive bounds for a backend that reports none

**Conditional.**
#568 is explicit that bounds come from `get_metadata` and that a backend reporting none yields `coverage_unknown` rather than a guess, but its stated principle is "facts only", not "no derivation" - #564 derives freshness from what the store actually holds. Probe-derived bounds are measurement, so they are admissible under their own name; what is inadmissible is returning them in the field a store's own report would occupy, since #568 exists precisely so the agent can tell four situations apart.

## H-3 A facility color family in the core token set - OPEN, needs the author

**Not answered.**
This is a genuine contradiction in the evidence and only the author can resolve it. Design principle 2 forbids facility-specific code in the core and explicitly forbids "domain-prefixed sibling artifacts". `tokens/core.json` nonetheless carries a `desy` family owning four color families no other facility uses, documented in `DESIGN.md` as shipped fact.

Three readings are available, and they yield different visions:

1. `desy` is an acknowledged exception to be retired, and the rule stands unqualified.
2. Theming is exempt from principle 2, in which case the principle should say so rather than be silently violated.
3. Facility identity in the core is acceptable generally, which would weaken principle 2 well beyond theming.

`VISION.md` currently states the rule unqualified, which matches reading 1. If the answer is 2, the "Nothing facility-specific reaches the core" section needs a line naming theming as the exception and bounding it.

## H-4 An override for the stale-store preflight

**Off mission.**
#587 measured what proceeding costs: `compose up` recreates the store container, which is the only host-side record of the credential its volume was initialized with, so the run destroys the recovery path for the very store it then cannot authenticate to. The same PR declines to add `--recreate-stores` and supplies `--reuse-stores` and `init --reset` instead. The project already believes a refusal needs an exit; it does not accept an exit that costs the operator the thing the refusal was protecting.

## H-5 Negative controls only where they are cheap

**Off mission.**
The proposal's premise is refuted by #565: the guard that silently rotted was CI wiring, not a safety boundary. Every agent-gated test in CI skipped for months behind a merge-gating lane that went green in about thirty seconds, and only a zero-skip check caught it. #501 spends 16 of 32 tests on negative controls for a config-key linter, which is exactly the "cheap" category the proposal would exempt.

## H-6 A read-only analysis deployment with no control system

**In vision, and the draft was wrong.**
`channel_finder_standalone` and `ariel_standalone` ship as app templates (#501), so read-only deployments without a control system are already supported and already serve users. The draft's scope line - "capability that does not serve an operator at a machine belongs elsewhere" - would have excluded shipped, working deployments. Reading honestly about a machine is half of what the project does; the real boundary is capability serving neither an operator nor a machine.

## H-7 Refusing a mock machine paired with a real archive

**Off mission.**
#564 states the rule's limit directly: "A mock control system with the mock archiver is untouched: nothing is claimed to be real there, so nothing lies." The refusal targets a component that *synthesizes at read time* being presented as history. A real archive holding a real past invents nothing, so the pairing is honest even though the two halves describe different machines. The draft's "worlds" framing was broader than the rule the code enforces.

## H-8 A deployment that cannot attribute actions to people

**Conditional.**
The safety-relevant distinction from #576 is agent versus human versus system, and it survives a shared account intact. Per-user login (#539) is a capability, not a precondition. But the honesty principle applies to the audit surface as much as to an archiver read: the #568 pattern is to name what cannot be determined rather than let a response read as complete, so such a deployment must record that human actions are not individually attributable.

## H-9 An expert mode that drops the caveats

**Off mission, with the underlying need granted.**
#588 states the goal as plain language "while every operational caveat stays stated", and enumerates the caveats that had to survive. The practiced answer to verbosity is not a second copy of the output: #603, #606 and #607 built a single CLI output hierarchy that promotes facts and warnings into levels. Ranking is the tool; a quieter mode that drops content is a second surface to keep true, and `test_printed_copy_style.py` guards only one of them.

## H-10 Freezing agent-facing names for a release year

**Off mission, but the migration need is real.**
#595 shows the renames were not cosmetic: `show_panel` moved rail membership despite its name, and five panels published URLs that returned 502. An alias window preserves exactly the two-names-for-one-thing state the principle exists to end. The project's own migration idiom is the tombstone - #583 deliberately keeps "tombstone errors for removed flags" - which names the change and fails, rather than keeping the old spelling working.

## H-11 Deleting guards proven unreachable

**Off mission.**
Design principle 1 states it flatly: a guard that never triggers is not dead code, it documents an invariant, do not remove it. #576 is the case in point - an unvalidated `execution_mode` string outside `readonly`/`readwrite` satisfied neither write gate and ran write-pattern code with `writes_enabled=false`. "Proven unreachable" is a claim about today's call sites, and the guard's value is realized on the day that stops being true.

---

## Changelog

Each line maps a verdict to the edit it produced in `VISION.md`.

- **H-1 conditional** -> "The human holds the arming decision" no longer claims a person arms every write. It now names the bounded, enumerable plan as the unit of arming, admits many writes under one approval, and excludes open-ended loops. Two lines added.
- **H-2 conditional** -> "The system says only what it knows" gains a line separating measurement from assertion and forbidding a derived value in the field a store's own report would occupy.
- **H-3 unanswered** -> No edit. "Nothing facility-specific reaches the core" states the rule unqualified, pending the author's decision on the `desy` token family.
- **H-4 off mission** -> "Refuse before you touch" gains "and that way preserves what the refusal was protecting", plus a line rejecting an override that costs the recovery path.
- **H-5 off mission** -> "A claim is kept by a guard that can fail" now says the cheap guards carry negative controls too, and names unwatched surfaces as where guards rot.
- **H-6 in vision** -> Scope rewritten. The line excluding capability that does not serve an operator at a machine is replaced by one admitting read-only deployments and excluding capability serving neither operator nor machine.
- **H-7 off mission** -> The honesty section's pairing rule is narrowed from mismatched worlds to fabrication presented as record, and now states that a simulated present may read a genuine recorded past.
- **H-8 conditional** -> Attribution gains a line: a deployment that cannot tell people apart records that it cannot.
- **H-9 off mission** -> A line added that legibility is bought by ranking output, never by a second quieter mode.
- **H-10 off mission** -> "One name, one home, one rule" gains the tombstone line: a removed name fails loudly and names its replacement rather than surviving as a working alias.
- **H-11 off mission** -> The existing guard line is sharpened to say why: unreachable today is a statement about today's call sites.
