# Vision

`osprey` exists so that an operator can ask a scientific control system a question in plain language and act on the answer without ever handing over the decision to act.
It serves the control-room operator who runs the machine and the deployer who stands the system up at a facility.
It turns a spoken request into an auditable chain of steps against real hardware.
It owns exactly one thing: the boundary between an agent and a facility, and everything that crosses it in either direction.

## The human holds the arming decision

A person arms every action that moves hardware, and the agent is structurally unable to arm one for itself.
The unit a person arms is a bounded plan they can read in full before it runs, not necessarily a single write.
A plan may cover many writes when its extent is enumerable in advance and a human holds a stop; an open-ended loop whose next write depends on state nobody has seen is not such a plan.
A deployed agent never holds the credential that starts motion; it files a request, and a person's click is what starts it.
Config surgery is never the fix for a refused action, so deployed agents cannot reach the surfaces that edit their own harness configuration.
Gates are wired into the type hierarchy rather than repeated at each call site, because a gate that must be remembered will eventually be forgotten.
An unrecognized value at a safety boundary is rejected, not interpreted; a mode that is neither read-only nor read-write is an error, not a write.
A guard that has never fired stays, because unreachable today is a statement about today's call sites and not about the change that lands next.
Convenience is not a reason to widen a gate, and neither is a test that would be easier to write without one.

## The system says only what it knows

An answer is built from what the system actually holds, and a missing answer is reported as missing.
Absence of data is a fact with a name: a window before the archive, a window after it, a channel never recorded, a gap inside coverage, or bounds the backend cannot report.
A backend that cannot say is allowed to say so, and is never made to guess so a response looks complete.
A measurement and an assertion are different claims and are never returned in the same field; a value derived from probing may be offered under its own name, never in the place a store's own report would sit.
A setpoint is reported as moved only when a readback confirmed it.
An action is attributed to whoever took it, and the agent is not credited with what a person or a scheduled sweep did.
Where a deployment cannot tell one person from another, it records that it cannot, rather than presenting a shared identity as though it were an individual.
What is refused is fabrication presented as record, not simulation as such.
Simulated components may be paired freely, and a simulated present may read a genuine recorded past, because neither claims to be what it is not.
A component that invents at read time may not be paired with one that presents its output as history, and that pairing is refused rather than documented as a caveat.
Shipped prose describes the system as it is, not as a diff against an arrangement no reader has seen.
User-facing text is written for someone who does not know the framework's internal vocabulary, and every operational caveat survives the rewrite.
Legibility is bought by ranking what is printed, never by a second quieter mode that drops what the first one had to say.
The project does not claim capabilities it does not have, including in its own README.

## Refuse before you touch

A run that cannot succeed stops before it changes anything, and says so in the seconds it takes to check rather than the minutes it takes to fail.
A refusal names every affected resource at once, not the first one discovered.
A refusal states the reason and the remedy, and no refusal escapes as a bare runtime error.
Proceeding is refused when continuing would destroy the evidence needed to recover, because a diagnosis that arrives after the recovery path is gone is not a diagnosis.
Ownership that cannot be proved is not assumed; the system declines and names what it found rather than sweeping resources that may not be its own.
A refusal ships with the way past it, and that way preserves what the refusal was protecting.
An override that costs the operator the recovery path is not an exit, however loudly it warns.
Refusal grounds are chosen narrowly and pinned by test, so a normal state is never mistaken for a broken one.

## One name, one home, one rule

One concept has one name across connectors, tools, panels and documentation.
Two concepts never share a name, and a name that describes the wrong axis is changed rather than explained.
A removed name fails loudly and names its replacement; it is never kept alive as a working alias, because two spellings of one thing is the state this principle exists to end.
A value is derived from one source, never declared in two places that can drift apart.
Each secret has exactly one writable home, and ownership is derived from what the build actually copied rather than from a list someone maintains.
A rule reachable from several entry points is decided in one place and asked at each of them, so a configuration cannot read as valid at one site and invalid at another.
A new subsystem follows the shape its peers already use unless it genuinely differs, and the difference is argued rather than assumed.
Deliberate inconsistencies are recorded with their reasons, and known open items are listed rather than quietly carried.

## Nothing facility-specific reaches the core

Code in the framework runs unchanged at a different facility.
Facility values live in configuration, protocol differences live in connectors, and facility narrative lives in presets and profiles.
The deployment repository is the source of truth for a facility, and the rendered project is a derivation that may be thrown away and rebuilt.
The model, the agent harness, the control protocol and the compute backend are replaceable by configuration.
Physics belongs to upstream packages behind a model boundary, not to this repository.
A change that would be wrong at another facility is in the wrong layer, whoever needs it and however soon.

## A claim is kept by a guard that can fail

A property the project claims is pinned by a check, and a check that cannot go red does not count as one.
Guards carry negative controls, and the cheap guards carry them too, because the surfaces nobody watches are exactly where a silent guard rots unnoticed.
A test that silently skips is a failure, not a pass, and a lane that reports success without running is treated as an outage.
Evidence is captured before teardown, since a failure that removed its own logs cannot be diagnosed.
Behavior is verified against a real deployment when only a running system can settle the claim.
Coverage of a hazard is derived from the code rather than from a hand-maintained list, so a new instance cannot quietly opt out.
Measured numbers are recorded as measurements, and a single green run is not treated as a calibration.

## Scope

Osprey is not a control system, not an archiver, not a physics code, and not a facility's deployment.
It is not an autonomous operator, and no roadmap item may make it one.
A deployment that only reads is within scope, because reading honestly about a machine is half of what this exists to do.
What is out of scope is capability serving neither an operator nor a machine, however well it would run here.
Facility-specific data, credentials and narrative stay in the deployment repository, never in the framework.
The repository holds itself to its own standard: its own claims about its own configuration, tests and prose are guarded the same way.

A change aligns when it narrows the distance between what the system says and what is true, when it makes the unsafe path harder to reach than the safe one, when it would be correct at a facility that has never been seen, and when the claim it makes is pinned by a check that could fail.
A change should be resisted when it lets the agent arm its own action, when it produces a plausible answer in place of an absent one, when it proceeds past a condition it could have detected, when it adds a second name or a second home for something that already has one, when it puts one facility's assumption in the framework, or when it asserts a property no guard enforces.
