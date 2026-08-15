---
name: osprey-build-interview
description: >
  Interactive interview to create a custom OSPREY build profile for a new accelerator, detector,
  or beamline application. Use when someone says "interview me", "create a build profile",
  "set up my agent", "configure my detector", "onboard me", or needs to create an OSPREY project
  tailored to their specific control system. Also covers someone who already has something —
  trigger on "I already have a project", "bring my existing setup forward", "my profile is in
  the old layout", or converting an existing web-terminal variant into a persona of the
  profile it belongs to. Also trigger when onboarding a new colleague, or when anyone needs
  help figuring out what their OSPREY agent should look like.
---

# OSPREY Build Profile Interview

Interview someone who works on an accelerator, beamline, or detector — an operator or
scientist, not a software engineer — and turn what they tell you into a **build profile**:
the small set of files `osprey build` consumes to render a working OSPREY project for
their system. Plan on roughly **three batched `AskUserQuestion` calls** — a fourth when
the assistant is headed for a shared machine — and about **five minutes** of their time.

## Write down only what cannot be discovered

This skill deliberately holds no catalog of OSPREY's features. Presets, providers, config
keys, artifacts, and schemas all change from release to release, and any list copied in
here is wrong within weeks. Whenever you need one, **discover it at runtime** — every
command and path for that is in `references/osprey-map.md`, this skill's only reference
file. Read the map before you generate anything, and go back to it the moment you catch
yourself about to state a fact about OSPREY from memory.

The same rule governs the past. This file carries no account of how earlier releases were
arranged and no steps for moving off one, because a remembered layout is the one kind of
fact you can never check. When someone arrives with an existing setup, the installed
version and the files in front of you are the evidence; there is nothing else.

## What you produce

A **facility repo**: a git repository whose root holds the `profile.yml` (started from a bundled preset and then edited — the map has the command
that emits one), a plain-language `README.md`, and a channel database plus channel
limits when the person gave you signal details. One command lays the whole repo out; the
CI pipeline is emitted once the `deploy:` block is filled in. Generation is covered in
full further down this file.

## What the interview must establish

These are **goals, not a question script.** Compose questions that suit the person in
front of you, in whatever order the conversation wants, and follow up wherever an answer
is thin. New OSPREY capabilities should reach the interview through discovery, never
through someone editing a question menu here.

1. **Whether they are starting from scratch** — see below. Ask this one first; it
   changes what the rest of the conversation is for.
2. **What the system is** — the kind of system, a short project name (lowercase, hyphens,
   no spaces), a one-line plain-English description, and the facility.
3. **How it connects** — simulated data to start with (the safe default for anyone
   unsure), or a live control-system connection. If live, collect the connection details
   that connection needs.
4. **Which signals** — the process variables the assistant will work with: names,
   descriptions, units, typical ranges, and which are read-only versus writable. If they
   have no list yet, capture the *shape* instead — signal types, rough count, naming
   convention, a few examples — and generate a skeleton they can fill in later. Also
   establish whether historical or archived data matters, and where it lives if so.
5. **What privilege level** — see below.
6. **Which AI service** — see below.
7. **Whether it gets deployed, and where** — see below.
8. **Who uses it, and what runs** — see below.

### The existing-setup fork

Open with it, in one plain question: *are we starting from scratch, or do you already
have something running?* "Something" is deliberately loose — an OSPREY project from an
earlier release, a profile in a layout that no longer matches, hand-written instructions
for the assistant, a tool someone wrote for it, a set of scripts, a web-terminal
deployment built before the current arrangement existed. People rarely volunteer any of
this, because it does not feel like part of "setting up an assistant" to them.

If they are starting fresh, that is the whole of this goal. Carry on with the rest.

If they already have something, the goal is **not** to run a migration procedure. It is
to find out what exists, and then decide *with them*, item by item, what carries forward
into the new profile and what retires. You work that out from evidence on their machine,
not from anything remembered about how OSPREY used to be arranged.

Gather the evidence first, before you ask a single migration question:

- **What version is installed here** — `osprey --version`. Everything below is relative
  to it, and it is the only version whose behaviour you can check.
- **What a current setup looks like** — materialize a profile into a scratch directory
  from a bundled preset (the canonical one, unless something they said points
  elsewhere) and read what it writes. That is the shape their setup is moving towards,
  defined by the installation in front of you rather than by memory. Delete it
  afterwards; it exists to be read, and the real one gets materialized later.
- **What they actually have** — open it. The old project or profile, the instructions,
  the tools, the scripts, the deployment. Read the files rather than asking them to
  describe the contents; they usually cannot, and it is the fastest part of this.

Now derive your own questions by comparing the two. For each thing they have, the
question is which of three things it is: a fact that belongs in the new profile, an
artifact that carries across as a file the profile owns, or something the framework
does natively and that should retire. Ask one at a time, in plain language, and say what
each answer costs them — a thing that retires is work they no longer maintain, which is
usually welcome news once it is put that way. Where you cannot tell whether the
framework covers something, check the installation instead of guessing; the map
lists the commands that answer that.

Two cases are common enough to name, though neither gets a procedure here:

- **A deployment whose extra terminals were built as separate variants.** OSPREY's own
  messages send people here for this one. The goal is to end with a single profile that
  every terminal renders from, so that a change to the facility's data or conventions
  reaches all of them at once instead of being copied around. A materialized profile
  that configures terminals *is* the shape, and it shows you both halves: the per-terminal
  files it writes beside `profile.yml`, and the way the profile's own terminal catalog
  points at them. Give each existing variant that same shape, carrying over only its
  *real* differences from the shared profile, and point the catalog at the result. If a
  runtime message is what sent them here, it names that setting outright — take it from
  the message or from the materialized example, never from memory.
- **Instructions, tools, and scripts written against an older arrangement.** These are
  usually the valuable part: they encode how the facility actually works. Move the
  content, not the wiring — the profile's convention directories are where facility-owned
  artifacts live, and the materialized `README.md` says which directory each kind lands
  in on this version.

**Do not write, follow, or invent a version-specific migration recipe.** No numbered
upgrade steps, no "in the old layout, X was called Y" lore, and nothing in this file will
ever grow into a migration guide — a recipe here would be describing a release nobody in
front of you is running. Read what is installed, read what they have, and reason about
the difference in the open where they can correct you.

Either path ends in the same place: one facility repo, described below.

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

### The deployment questions

Deployment coordinates are opt-in, so the first thing to settle here is the fork: *is
this destined for a shared machine that other people log in to, or does it run where
you are?* A profile that only ever builds on one person's laptop needs none of what
follows, and "not yet" is a perfectly good answer — it can be added the day a server
appears. Do not walk someone through server details for a system that has no server.

If there is a shared machine, establish how the software gets there and who meets it:

- **How it is built and published** — which CI platform builds the images, where the
  built images are kept, and how the machine running them gets hold of them: pulling
  what CI built, or building its own on the spot. A facility whose deploy host cannot
  reach the registry is the ordinary reason for the second answer, not an exotic case.
- **Which machine runs it** — how to reach it, which account owns the checkout there,
  and where on the machine that checkout lives.
- **Anything else it pulls** — some deployments run another project's image alongside
  their own; if theirs does, get that project's coordinates too.
- **Who uses it** — one person, or a room of operators who each want their own
  terminal. A multi-operator deployment needs the roster, because each person gets
  their own workspace on the host.
- **What actually runs** — which of the framework's optional services this deployment
  turns on. Services nobody asked for still cost startup time and attention on the
  host, so take the answer rather than assuming the preset's.

Ask these the way you asked the rest: plainly, with defaults ready, and without a
lecture on pipelines. The keys these answers become are not written down here — the
deploy schema in the map is the source of truth for the block's shape, and you read it
when you write the block, not when you ask the questions.

## Running the conversation

- Open with a one-line welcome and an honest estimate: a few minutes, "I'm not sure" is
  always an acceptable answer, and everything can be changed later.
- Ask the fresh-or-existing fork before the batches, on its own. If they already have
  something, go and read it before you ask anything else — the batches below are worth
  more once you know what is already there, and some of their answers will be sitting in
  the files.
- Batch related questions into a single `AskUserQuestion` call instead of asking one at a
  time. Aim for three batches: what the system is; how it connects and which signals;
  then privilege level and AI service. Add a fourth only if the deployment fork says
  there is a shared machine — someone building on their laptop should never see it.
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

Once the goals are covered, review the collected requirements yourself — as a
skeptic hunting for gaps and contradictions — and resolve whatever you find *with the
person* rather than guessing. Categories worth checking:

- Write access wanted, but no safe operating ranges given for the writable signals.
- "Read-only" stated, but the work they described requires changing values.
- A live control-system connection chosen, but the connection details are missing.
- Historical data expected, but no archive source identified.
- Signals implied by the use case that never made it into the list, or missing units and
  ranges on signals they intend to analyze.
- Scope much narrower, or much broader, than what they said they wanted.
- A shared machine described, but no answer for how it gets the images it runs.
- Several operators expected, but nothing said about who they are.
- Something they already had that never got a decision — neither carried into the
  profile nor deliberately retired. An item that quietly fell off the list is the one
  they will miss first.

## Generating the profile

Pick the starting preset first. `osprey profile presets` reports what this
installation ships; open the ones that sound close — the map says where they live — and
take the one whose privilege level and connection mode match what the interview
established. The `control-assistant` family is the canonical example and a
sensible default when nothing else stands out.

Then lay out the facility repo:

```
osprey init <facility-name> --preset <closest-preset>
```

`--preset` is required. The command refuses to write into a directory that already holds
a deployment, which a second pass through the interview will hit — move the old one aside
or write into a fresh name, and tell the person which you did.

What you get back is a git repository rather than a loose directory, laid out in four
zones. The repo ROOT is the editable source: `profile.yml` sits directly at the top,
beside its `data/` tree, its convention directories, and the CI pipeline that lands there
once the `deploy:` block is filled in. `.env` holds the secrets — provider keys and the
tokens `osprey up` mints — and is kept out of git alongside `build/`, which holds what
`osprey build` renders. `var/` holds runtime state. The map records the layout and what each
part is for. One consequence runs through everything below: every path you edit is a path
at the repo root, and no command needs to be told where the repo is — they all find it by
walking up from wherever you are standing.

`profile.yml` is standalone and self-documenting: the preset's full
configuration written out explicitly — no `extends:` — with the preset's own comments,
next to a `data/` tree copied from the preset and the convention directories its
`README.md` walks through. Read it before you edit it. It is the current, authoritative
statement of what a profile can say, which is exactly why no copy of it lives in this
file.

Now edit **only the deltas the interview actually decided** — the project name and
description, the connection mode, the AI service, the signals. Everything that never
came up in the conversation already carries the preset's answer, so leave those keys as
materialized rather than second-guessing them. Under `config:`, use dotted keys
(`system.timezone: "America/Los_Angeles"`); nested YAML there does not merge the way
people expect.

If the interview established a shared machine, the deployment coordinates go in the
profile's `deploy:` block. Its schema — the map points at it — is the source of truth
for the block's shape, and it reports every problem in one pass, so writing the block
and then running the profile validator is the quickest way to get it right.

One trap belongs to that block. How the machine obtains its images has two possible
homes, and some presets already answer it under `config:` — so a profile that has never
been touched can carry the answer, and a profile someone else wrote almost certainly
does. The build refuses outright when both homes are filled in, because two homes for
one fact are free to disagree, and this disagreement decides whether the host builds its
own images or pulls them. So before you write the block, look for that existing answer
under `config:` and remove it: the `deploy:` block is the home, and the build carries
its value across into the rendered config for you. The refusal names the offending entry
if you get there first.

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
osprey build --repo <facility-name> --skip-deps
```

Exit 0 is required. `--skip-deps` keeps it quick — you are checking that the profile
renders, not installing anything. The render lands in the repo's `build/` zone, which is
kept out of git, so there is nothing to clean up afterwards. `--repo` is only needed
because you are standing outside the repo; from inside it, plain `osprey build` does the
same thing.

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
keep the conversation moving; answering from recall is exactly how this skill goes
stale.

**They describe an existing setup you cannot open.** It is on another machine, or behind
a login, or they only half remember it. Say so plainly and work from what you *can* see:
build the profile from the interview as though it were fresh, and write down in the
README which of their existing pieces still need a decision. Do not reconstruct the old
setup from their description plus a guess at how that release was arranged — a migration
built on a remembered layout silently drops the parts nobody thought to mention. They can
bring the files to the next session, and picking it up then costs almost nothing.

**They have no signal list yet.** Do not block on it and do not emit an empty database.
Write a skeleton in the shape they described, with placeholders that are obviously
placeholders, and put instructions in the README for replacing them and rebuilding. They
can come back with the real list any time.

**No preset is a good fit.** Start from the closest one anyway. Record what it does not
cover as stub files in the profile's convention directories — the materialized
`README.md` names them and says where each one lands — and as notes in that README, so
the gap is visible and someone can fill it in place. Authoring a profile from scratch to
avoid an imperfect preset costs far more than it saves.

## Handing over

They rebuild their project from the finished profile at any time with:

```
osprey build
```

Run it from anywhere in the repo; the render lands in `build/` either way. Drop
`--skip-deps` for a project that actually runs — that is the difference between the
verification build above and a usable one.

This skill settles *what* to build. Running what was built is the CLI's own job and
has no skill of its own: `osprey scaffold ci` emits the pipeline and health check
from the profile's `deploy:` block, `osprey up` brings the stack up, and `osprey
status` / `osprey logs` are where triage starts. Each verb's `--help` is the current
catalog; do not reproduce it here.
