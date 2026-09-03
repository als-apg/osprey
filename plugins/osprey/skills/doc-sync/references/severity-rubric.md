# Severity rubric and the two gates

The failure mode of a doc audit is noise: a wall of "I would phrase this differently"
that buries the one claim that is actually broken. A finding is something that would
**mislead or block a reader following the page**, and that you can **prove** against the
code or a captured run. Everything else is dropped, not appended.

A candidate must pass both gates.

## Gate 1 — Correctness: did we actually verify it?

Evidence, not an impression:

- A static claim (a signature, a default, a config key, a return field, a named symbol):
  a `file:line` citation that confirms or contradicts the page.
- A runtime claim (a command works, a mode falls back, output is truncated at N): captured
  output from actually running it. "It looks like it would raise" is not evidence.

Then one honest attempt to refute it. Re-read the source as if you were defending the
page: is the verifier looking at the wrong file, a stale layer, a test double, a
deprecated alias that still works? Only a finding that survives is filed. Record how many
candidates died; a healthy count is the self-correction working.

A claim that cannot be verified with what the run has (a container runtime, a live
control system, a kernel, a credential) is **not** a finding. It goes under **Not run**
with the specific blocker, after whatever static check was possible. Seen and
deliberately not judged, never guessed.

## Gate 2 — Impact: would a reader care?

Ask literally: a reader opens this page and follows it as written. Are they misled or
blocked? If yes, file it with a tier. If no, drop it, and not into an appendix either.

Fails Gate 2 by construction:

- Style and wording preferences.
- "The page could also mention X." OSPREY's standing preference is the opposite: a page
  earns each addition by clearing a real foot-gun bar. A missing mention is a finding only
  when its absence actively misleads, such as a now-required step the reader fails
  without.
- Re-explaining a trade-off the page chose not to cover.
- Broken markup, dead `:doc:`/`:ref:` targets, orphan pages. The Sphinx build with
  warnings as errors and the guards under `tests/docs/` own those; the skill's step 1
  runs them and reports the count.
- Anything not tied to a concrete code or runtime fact.

When unsure, drop it. A short trustworthy report beats a long one the maintainer learns
to skim.

## Tiers

| Tier | Definition | Reader's experience |
|---|---|---|
| **breaking** | Following the page produces a failure: a command or flag the CLI rejects, a key the loader ignores or rejects, a documented call that raises, a documented feature that does not work. Confirmed by running it. | "I did what it said and it failed." |
| **wrong** | A factual claim that is false and builds a wrong model: wrong default, wrong signature, wrong return field, a symbol or flag that does not exist, behaviour that differs from the claim. No crash, but the reader is now wrong about the system. | "It does not behave the way the page told me." |
| **stale** | True but behind: a step that is now required and missing, a renamed symbol still under its old name, a caveat whose absence will bite a careful reader. | "Technically works, but the page is behind reality." |

There is no cosmetic tier. A typo in prose fails Gate 2; a typo inside a command literal
is **breaking** and is filed as such.

## The side question

A drift between page and code can be fixed on either side. The page says the default
timeout is 600 seconds and the code says 300: maybe the page is stale, or maybe the
default was changed by accident and the page records the intent. Every finding states
which side the verifier believes is wrong and why. Picking the side is the maintainer's
ruling, recorded in the decisions file, and only doc-side rulings are applied by the
skill.
