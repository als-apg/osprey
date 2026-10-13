:html_theme.sidebar_secondary.remove: true

.. _architecture-testing:

=======
Testing
=======

OSPREY is a harness around an agent that can reach a machine, so what has to be proved is
not one thing. The code has to be correct, the hooks have to refuse what they must, the
machine the agent is tested on has to behave like one, the agent has to do the job with a
real model, and all of it has to come up as containers the way a facility runs it. The
suite is built as five rings, each adding one real thing to the last. All five gate a pull
request, and a red in any of them blames the change. Two further uses of the same machinery
judge something else: the model, on the benchmark board, and the facility's own deployment,
in production.

This page is the map. How to run the suites, the fixture house style and the full marker
vocabulary are in the repository's ``tests/README.md`` and ``tests/e2e/README.md``; the
contributor's view is :doc:`/contributing/development-setup`.

.. raw:: html
   :file: ../_diagrams/testing-rings.html

The five rings
==============

**1 · The code, in process.** The unit lane: about 1,800 test files over the framework, the
CLI, the MCP servers and the interfaces, run under four parallel workers with every file
kept on one worker. Nothing here reaches a model or a service the test did not start
itself, and a test that changes process-global state resets it on the way in and on the way
out. Two static tiers sit beside the lane on every push. *Tier 0* holds the shipped presets
to the artifact registry and every skill's tool references to the MCP tools that exist;
*Tier 1* builds each preset for real and boots every server in its ``.mcp.json`` through a
JSON-RPC handshake. A set of guard tests holds the repository to its own rules --- the agent
SDK is imported only behind the harness adapter, a credentialed test lives only under
``tests/e2e/`` --- and holds the published documentation to the code.

**2 · The hooks, without a model.** The :doc:`safety chain <safety-chain>` is tested as
code before any agent runs it: each PreToolUse hook on its own, the chain end to end against
the mock control system, and the text the deployed agent is given --- its safety rules,
output style and memory guard --- pinned by template tests. A red here is a hole in the
harness that no model could have papered over, which is why this ring exists apart from
ring 4.

**3 · The machine.** An agent cannot be tested against nothing, and a real accelerator is
not an option. Two substrates stand in. The data-driven simulation engine serves channel
reads, writes and archiver history from scenario bundles, each a self-contained fault with
its own telemetry overlay and logbook narrative, and the statistical signature of every
bundle is pinned deterministically, without a model, before any agent sees it. The
:doc:`Virtual Accelerator <virtual-accelerator>` puts a whole facility on real EPICS with
physics behind the storage-ring channels; its own suite proves that the wire agrees with the
namespace it meant to serve, that a corrector write moves the downstream BPM, and that the
approval and limits gates hold against a live container. One rule follows. When a scenario
test with an agent in it regresses, run the substrate's contract tests first: if they pass,
the miss is the agent's; if they fail, fix the bundle, never the prompt.

**4 · The model.** The e2e suite runs a real agent through the SDK against a deployment repo
it built with ``osprey init``, with one pinned Claude Code and one named provider; a run that
names none is refused before its first test. Grading is in two layers. A deterministic floor
over the tool trace decides whether the agent did the thing, and an LLM judge is given only
what a trace cannot show: whether the agent described what it did without inventing
findings. Both layers are verifiable offline, the floor against hand-built traces and the
judge against one passing conclusion and one failing control per criterion. *Tier 3* is the
simplest member: for each shipped preset, one agentic flow from build to answer. This ring
spends model tokens, so on a pull request it runs only under the ``full-ci`` label, and on
``main`` by a maintainer's dispatch; there is no schedule, because a red nobody is waiting
for reaches nobody. The lanes are gates all the same. There is no advisory tier anywhere in
the suite: every lane that runs must be green.

**5 · The stack.** A facility runs OSPREY as containers --- with an archiver, a dispatch
worker, bridges to its chat and an auth perimeter in front --- so the last ring builds those
images and deploys those stacks for real. About forty lanes, one per stack, each a proof that
needs no model: the dispatch pair boots in both network shapes, the privilege split holds at
the level of file ownership inside the image, a persona behind nginx sees the identity
headers it should and no others, a Bluesky run survives a bridge restart. Each lane is a
hand-written job, and a wiring test parses the workflow to prove that every container-backed
e2e file has one and that the shared e2e lane never collects it by accident.

Each ring adds one real thing to the one before: real code, then real hooks, then a real
machine, then a real model, then real containers. A test belongs to the innermost ring that
can prove its claim. A claim that needs a model is not pushed into ring 2 by mocking the
model, and a claim ring 2 can pin is not left to the judge.

Two more defendants
===================

The rings judge a change. The same machinery, pointed differently, judges two other things.

**6 · The model, on the benchmark board.** Ring 4's suite is the test; the model is what is
on trial. The board runs the suite against a matrix of models and seeds, forcing every alias
the agent uses onto the model under test, and scores two lanes that every e2e test in scope
declares by marker. The *capability lane* holds the agentic tasks a weak model plausibly
fails, and is the headline number. The *harness lane* holds the tests where an agent runs
but the assertion is OSPREY's own behaviour, scored separately so that its near-constant
passes never pad a model's score. A reference model should score close to everything, so a
miss by a strong model is read first as a bug on OSPREY's side, and the board has found
several. *Tier 2* is the board's home in CI: a small set of tasks that are fair but that a
capable model may still miss, kept out of every pull-request lane because a miss there says
something about the model, not the change. A Tier 2 task may deploy the real stack when the
task needs one.

**7 · The facility, in production.** A deployment is not done being tested when it ships.
Every deployment carries ``osprey health``, a composed suite of diagnostics over its
configuration, environment, containers and enabled services, plus the probes the facility
declares, read in the browser's SYSTEM panel, from the command line, or by a CI job; each
check carries a cost class, so a browser can never trigger one with side effects. ``osprey
audit`` turns an agent on a built profile and reports the safety risks it finds in the
permissions, hooks and server configuration. Both test the facility's deployment rather
than OSPREY, and both are documented under :doc:`/how-to/health-and-monitoring/index`.

Tiers and lanes
===============

Two words recur on this page and are easy to conflate. A **tier** numbers the cost of a proof in
CI: Tier 0 is static, Tier 1 builds and boots, Tier 2 runs a few hard agentic tasks, Tier 3
runs a whole agentic flow per preset. Whether a tier gates a pull request is a separate
property: Tiers 0, 1 and 3 do, and Tier 2 measures instead and runs only on demand. A
**lane** is a CI job and, on the benchmark board, one of the two scores. The board speaks of
lanes, never of tiers.

Markers are hops
================

A pytest marker in this suite is not a label. It is an instruction about where a test runs
instead of where its directory would put it. ``slow`` takes a test out of the fast local
loop and nothing else. ``pty`` and ``wall_clock`` take a test out of the parallel sweep and
into a serial step that runs it alone, because a repaint cadence and a time budget both
measure the machine as much as the code; a budget is never loosened for a slow runner and
never skipped on one, and the test prints what it measured so a red says which of the two it
was. ``dockerbuild`` and ``qmd`` keep a container-backed e2e file out of the shared e2e lane
and in one of its own. ``model_free`` names the e2e tests that reach no model, which is
exactly the set the credential-free e2e lane runs. ``agentic_benchmark`` and
``harness_benchmark`` say which lane the board scores a test in, and every e2e test in the
board's scope carries one of them. The ``requires_*`` family declares a resource and skips
without it, which is why a credentialed test may live only under ``tests/e2e/``: the unit
lane must never reach a gateway, whatever keys a developer has in their shell.

Each of these pairs a marker with a lane, and a wiring test over the parsed workflow holds
every pair: the parallel lane must deselect what the serial step selects, the serial step
must exist, a marked file must live where its step looks. Losing either half of a pair is
silent --- a suite nobody runs under a green check --- and that silence is what the wiring
tests exist to make impossible.

.. seealso::

   :doc:`/contributing/development-setup`
      Running the suites, the local check scripts and the front-end rails.

   :doc:`/contributing/workflow`
      When the model-spending lanes run and what a pull request needs.

   :doc:`safety-chain`
      The hooks that ring 2 tests and ring 4 exercises.

   :doc:`virtual-accelerator`
      The machine that ring 3 stands up.
