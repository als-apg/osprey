---
name: osprey-build-interview
description: >
  Interactive interview to create a custom OSPREY build profile for a new accelerator, detector,
  or beamline application. Use when someone says "interview me", "create a build profile",
  "set up my agent", "configure my detector", "onboard me", or needs to create an OSPREY project
  tailored to their specific control system. Also trigger when onboarding a new colleague, or
  when anyone needs help figuring out what their OSPREY agent should look like.
---

# OSPREY Build Profile Interview

Interview someone who works on an accelerator, beamline, or detector — an operator or
scientist, not a software engineer — and turn what they tell you into a **build profile**:
the small set of files `osprey build` consumes to render a working OSPREY project for
their system. Plan on roughly **three batched `AskUserQuestion` calls** and about **five
minutes** of their time.

## Write down only what cannot be discovered

This skill deliberately holds no catalog of OSPREY's features. Presets, providers, config
keys, artifacts, and schemas all change from release to release, and any list copied in
here is wrong within weeks. Whenever you need one, **discover it at runtime** — every
command and path for that is in `references/osprey-map.md`, this skill's only reference
file. Read the map before you generate anything, and go back to it the moment you catch
yourself about to state a fact about OSPREY from memory.

## What you produce

A `build-profile/` directory holding a `profile.yml` (started from a bundled preset and
then edited — the map has the command that emits an editable one), a plain-language
`README.md`, and a channel database plus channel limits when the person gave you signal
details. Generation is covered in full further down this file.

## What the interview must establish

These are **goals, not a question script.** Compose questions that suit the person in
front of you, in whatever order the conversation wants, and follow up wherever an answer
is thin. New OSPREY capabilities should reach the interview through discovery, never
through someone editing a question menu here.

1. **What the system is** — the kind of system, a short project name (lowercase, hyphens,
   no spaces), a one-line plain-English description, and the facility.
2. **How it connects** — simulated data to start with (the safe default for anyone
   unsure), or a live control-system connection. If live, collect the connection details
   that connection needs.
3. **Which signals** — the process variables the assistant will work with: names,
   descriptions, units, typical ranges, and which are read-only versus writable. If they
   have no list yet, capture the *shape* instead — signal types, rough count, naming
   convention, a few examples — and generate a skeleton they can fill in later. Also
   establish whether historical or archived data matters, and where it lives if so.
4. **What privilege level** — see below.
5. **Which AI service** — see below.

### The privilege question

Ask it the way an operator thinks about it: *should the assistant only look at things, or
also change things?* Read-only is the default and the recommended starting point. Do not
lecture them about approval flows, limit checking, or verification — the preset you
select encodes the safety posture, and restating it here would only go stale. Your job is
to pick the preset that delivers what they asked for.

If they do want the assistant to change things, gather what the profile needs in order to
do that safely: exactly which signals should be writable, and the safe operating range
for each.

### The provider question

Ask which AI service they have access to. Most people know this as "whatever my lab gives
me", and some genuinely do not know. Build the answer options from the **live provider
registry** — the map points at it — never from a list written down here. Present the
discovered names plainly, and keep an "I'm not sure" option that does not block progress.
Model choice defaults to whatever the preset supplies; only ask about it if they raise
it, and take the valid values from the same registry.

## Running the conversation

- Open with a one-line welcome and an honest estimate: a few minutes, "I'm not sure" is
  always an acceptable answer, and everything can be changed later.
- Batch related questions into a single `AskUserQuestion` call instead of asking one at a
  time. Aim for three batches: what the system is; how it connects and which signals;
  then privilege level and AI service.
- Recap in a sentence between batches — "so far I have…" — so they stay oriented and can
  catch a misunderstanding early.
- Explain *why* a question matters before you ask it. A question with no visible purpose
  feels like a form.
- Have a default ready for everything. If they hesitate, offer the simple option, say it
  can be extended later, and move on.
- Prefer the minimal setup. Someone starting with simulated data and their main signals
  has a working assistant today; anything else can be layered on any time.
- Plain language throughout. Skip the framework vocabulary; say what a thing does.

## Consistency review before generating

Once the five goals are covered, review the collected requirements yourself — as a
skeptic hunting for gaps and contradictions — and resolve whatever you find *with the
person* rather than guessing. Categories worth checking:

- Write access wanted, but no safe operating ranges given for the writable signals.
- "Read-only" stated, but the work they described requires changing values.
- A live control-system connection chosen, but the connection details are missing.
- Historical data expected, but no archive source identified.
- Signals implied by the use case that never made it into the list, or missing units and
  ranges on signals they intend to analyze.
- Scope much narrower, or much broader, than what they said they wanted.

## Generating the profile

Pick the starting preset first. `osprey build --list-presets` reports what this
installation ships; open the ones that sound close — the map says where they live — and
take the one whose privilege level and connection mode match what the interview
established. The `control-assistant` family is the canonical modern example and a
sensible default when nothing else stands out.

Then emit an editable profile:

```
osprey build --emit-profile build-profile/ --preset <closest-preset>
```

`--preset` is required. The command refuses if `build-profile/` already exists, which a
second pass through the interview will hit — move the old directory aside or emit into a
fresh one, and tell the person which you did. It also refuses the project-render flags;
the map lists them.

What you get back is a self-documenting `profile.yml`: `extends: <preset>` plus
commented-out sections for skills, rules, agents, config, env, and overlays. Read it
before you edit it. It is the current, authoritative statement of what a profile can
say, which is exactly why no copy of it lives in this file.

Now edit **only the deltas the interview actually decided** — the project name and
description, the connection mode, the AI service, the signals. Everything that never
came up in the conversation is already answered sensibly by the preset, and leaving
those sections commented out is the whole point of extending a preset instead of
authoring a profile from scratch. Under `config:`, use dotted keys
(`system.timezone: "America/Los_Angeles"`); nested YAML there does not merge the way
people expect.

Write `README.md` in the same plain language you used in the interview: what this
profile builds, what was decided and why, what was left at preset defaults, and what to
do next.

## The signals

Generate the channel database and the channel limits from what they gave you — names,
descriptions, units, ranges, and which ones are writable. Do not work from a schema
written down anywhere in this skill. Open the two live examples the map points at (a
channel-database template and a channel-limits file, both shipping in the wheel, both
documenting themselves inline) and follow their shape. If they described a device family
and a naming convention rather than listing every signal, the template shows how to
express that without typing out hundreds of names.

Channel limits only matter when the assistant may change things. Every writable signal
needs its safe operating range; if one is missing, go back and ask rather than inventing
a number.

## Verify it builds before you hand it over

Build the profile yourself before you tell anyone it works:

```
osprey build <throwaway-project-name> build-profile/profile.yml --skip-deps
```

Exit 0 is required. `--skip-deps` keeps it quick — you are checking that the profile
renders, not installing anything. Delete the throwaway project directory afterwards.

If it exits non-zero, read the actual error, correct the profile, and run it again.
Never hand over a profile that does not build, and never describe a failed build as a
success. If it still fails after a few honest attempts, say plainly what the error says
and what you tried; that is far more useful to them than a confident handover of
something broken.

## When something does not fit

**The build verification fails.** As above — read the error, fix the profile, retry, and
be straight about it if you cannot get it green.

**`osprey` is not on PATH.** Check this before you ask the first question. Everything
here depends on asking a live installation what exists, so without the CLI you cannot do
this honestly. Tell them exactly what to run — `pip install osprey-framework`, or
whatever their facility's install instructions say — and then stop cleanly. Do not fall
back to answering from memory and do not fabricate preset, config, or service names to
keep the conversation moving; answering from recall is how the previous version of this
skill went stale.

**They have no signal list yet.** Do not block on it and do not emit an empty database.
Write a skeleton in the shape they described, with placeholders that are obviously
placeholders, and put instructions in the README for replacing them and rebuilding. They
can come back with the real list any time.

**No preset is a good fit.** Start from the closest one anyway. Record what it does not
cover as stub files under `overlays/` and as notes in the README, so the gap is visible
and someone can fill it in place. Authoring a profile from scratch to avoid an imperfect
preset costs far more than it saves.

## Handing over

They build their project from the finished profile with:

```
osprey build <project-name> build-profile/profile.yml
```

Then point them at the deploy phase:

```
osprey skills install osprey-build-deploy
```

This skill settles *what* to build. That one covers *how to ship it* — dependencies,
credentials, and getting it running. Leave that work to it.
