:html_theme.sidebar_secondary.remove: true

.. _architecture-testing:

=======
Testing
=======

OSPREY is a harness around an agent that can reach a machine, so the suite has to
prove several different things: that the code is correct, that the presets it ships
build and boot, that the simulated machine behaves like one, that an agent with a
real model does the job, and that the whole thing comes up as containers the way a
facility runs it. This page is the inventory: what tests exist, which CI tier and
lane each belongs to, what the markers do, how a fault is staged for the agent to
find, and what the benchmarks measure. How to run the suites and the fixture house
style are in the repository's ``tests/README.md`` and ``tests/e2e/README.md``.

.. raw:: html
   :file: ../_diagrams/testing-suite-map.html

What the suite is made of
=========================

**The unit suite** is everything under ``tests/`` that the other families do not
claim: the framework, the CLI, the MCP servers, the interfaces, the hooks. It runs in
process under four parallel workers with every file kept on one worker, reaches no
model and no service it did not start itself, and resets any process-global state it
touches. Two things run after the parallel sweep, each alone: the real-terminal
``pty`` tests, which read the bytes an operator would see, and the ``wall_clock``
tests, which hold a time budget. A set of guard tests holds the repository to its
own rules, for example that the agent SDK is imported only behind the harness
adapter, that a credentialed test lives only under ``tests/e2e/``, and that the
published documentation still matches the code.

**The hooks** are tested as code before any agent runs them. Each PreToolUse hook has
its own tests under ``tests/hooks/``, the chain is run end to end against the mock
control system, and the text a deployed agent is given, its safety rules, output
style and memory guard, is pinned by template tests. A red here is a hole in the
:doc:`safety chain <safety-chain>` that no model could have papered over.

**The presets** have two static tiers under ``tests/integration/``. Tier 0 checks
every shipped preset against the artifact registry and every skill's tool
references against the MCP tools that exist. Tier 1 builds each preset for real and
boots every server in its ``.mcp.json`` through a JSON-RPC handshake. Both run on
every push.

**The machine** is two substrates and their contracts. ``tests/simulation/`` pins the
data-driven simulation engine: how scenario bundles load, compose and apply, and the
statistical signature of every shipped fault, deterministically and without a model.
``tests/va/`` is the :doc:`Virtual Accelerator <virtual-accelerator>`'s own suite: the
manifest, the lattice, the physics bridge, the write path, plus live Channel Access
suites that run one process per module behind a gate that refuses a skipped module.
``tests/va/e2e/`` proves the container from where a client stands: the wire agrees
with the namespace it meant to serve, a corrector write moves the downstream BPM,
the limits and approval gates hold, and a scenario switch reaches the running IOC.

**The agent** is ``tests/e2e/``. Each test builds a deployment repo with ``osprey
init`` and ``osprey build``, then runs a real Claude Code session through the SDK
against it, with one pinned Claude Code version and one named provider; a run that
names no provider is refused before its first test. Grading is in two layers: a
deterministic floor over the tool trace decides whether the agent did the thing, and
an LLM judge is given only what a trace cannot show, whether the agent described
what it did without inventing findings. Both layers are verified offline, the floor
against hand-built traces and the judge against one passing conclusion and one
failing control per criterion. The shared ``e2e-tests`` lane runs the bulk of these;
Tier 3 runs one canonical flow per preset; the scenario tests described below are
the hardest members. This family spends model tokens, so on a pull request it runs
only under the ``full-ci`` label, and on ``main`` by a maintainer's dispatch.

**The stacks** are the container-backed e2e files, each with a CI lane of its own:
the dispatch pair in both network shapes, the privilege split inside the image, the
auth perimeter and persona headers behind nginx, the Bluesky, Tiled and archiver
stacks, the chat bridges, the two-shape boot under Docker and Podman. Some forty
lanes, and none needs a model. A wiring test parses the workflow and proves that
every ``dockerbuild`` file has a lane and that the shared e2e lane never collects it
by accident. The ``e2e-no-model`` lane runs the ``model_free`` files on every pull
request, fork pull requests included, and fails on a single skip.

**The board** is ``scripts/benchmark/``: the e2e suite run against a matrix of models
and seeds, with Tier 2 tasks and the channel-finder benchmarks beside it. It measures
and never gates; see `Benchmarks`_.

**In deployment**, a facility keeps testing itself. ``osprey health`` composes
diagnostics over configuration, environment, containers and enabled services, plus
the probes the facility declares, readable in the browser's SYSTEM panel, from the
command line or by a CI job. ``osprey audit`` turns an agent on a built profile and
reports the safety risks it finds in permissions, hooks and server configuration.
Both are documented under :doc:`/how-to/health-and-monitoring/index`.

Tiers
=====

A tier numbers the cost of a proof about a shipped preset. Whether a tier gates a
pull request is a separate property.

.. list-table::
   :header-rows: 1
   :widths: 6 44 26 14 10

   * - Tier
     - What it proves
     - Where
     - Runs
     - Gates
   * - 0
     - The preset YAML names only hooks, rules, skills, agents and panels the
       registry has; every skill's MCP tool reference names a tool a server
       exposes.
     - ``tests/integration/test_preset_static.py``
     - every push, seconds
     - yes
   * - 1
     - ``osprey build`` succeeds for the preset and every server in the rendered
       ``.mcp.json`` answers ``initialize`` and ``tools/list`` with the documented
       tools.
     - ``tests/integration/test_build_boot.py``, one job per preset
     - every push, minutes
     - yes
   * - 2
     - A fair agentic task that a capable model may still miss, such as reading
       a value off a plot nothing told it to open. A miss says something about
       the model, not the change.
     - ``tests/e2e/tier2/``
     - on demand (``run_benchmarks``) and on the board
     - no
   * - 3
     - One canonical agentic flow per preset: the agent reaches for the
       documented MCP tool and the response passes the judge.
     - ``tests/e2e/test_preset_agentic.py``, one job per preset
     - ``full-ci`` label, model tokens
     - yes

There is no advisory tier: every lane that runs on a pull request must be green.
Tier 2 is the one rung that measures instead of gating, and it is kept out of
every pull-request lane for that reason.

Two other things in the repository are called tiers and are unrelated to these.
A channel-finder **build tier** (``--set tier=N`` on ``osprey init``) sets how much
of the facility's channel corpus a render carries, which is why the scenario tests
build at build tier 3: the vacuum gauges exist only there. The multi-user **web
tier** is the nginx-fronted persona layer of a deployment. The benchmark board
speaks of lanes, never of tiers.

Markers
=======

A marker here is an instruction about where a test runs instead of where its
directory would put it, or about a resource it needs. A marked test and the lane
that reads the marker are a pair, and a wiring test over the parsed workflow holds
every pair, because losing either half is silent: a suite nobody runs under a
green check.

.. list-table:: Hops: the marker moves the test to another step or lane
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``slow``
     - Too long for the fast local loop. Still runs in CI.
     - ``scripts/quick_check.sh`` deselects it
   * - ``pty``
     - Drives an ``osprey`` verb on a real terminal and asserts on repaint
       cadence, which three other workers loading the box would distort.
     - the parallel lane deselects it; a serial step runs ``tests/pty/`` alone
   * - ``wall_clock``
     - Holds a time budget: a build that must finish in so many seconds, a
       search whose p95 must stay under one. Prints what it measured before it
       asserts, so a red says whether the code regressed or the host could not
       hold the budget. Never loosened for a slow runner, never skipped on one.
     - the parallel lane deselects it; a serial step after it selects it back
   * - ``dockerbuild``
     - Runs a real container build. Skips when Docker is absent.
     - kept out of the shared e2e lane; each file has a lane of its own
   * - ``qmd``
     - Needs the qmd sidecar image and refuses to skip without it, because a
       lane that skipped on a missing image would report success having
       proved nothing.
     - a lane that builds the image; every other lane deselects it
   * - ``model_free``
     - An e2e test that reaches no model.
     - the ``e2e-no-model`` lane, whose selection equals this set

.. list-table:: Resources: the test needs something the host may not have
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``requires_e2e_provider``
     - Builds its deployment with the provider the run named in
       ``OSPREY_E2E_PROVIDER`` and needs that provider's credential.
     - the e2e lanes, which name a provider; refused without one
   * - ``requires_als_apg``, ``requires_cborg``, ``requires_anthropic``,
       ``requires_openai``, ``requires_google``
     - Pins a gateway itself and needs its key. ``cborg`` is local-only: its
       IP allowlist blocks GitHub runners.
     - ``tests/conftest.py`` turns a missing key into a skip
   * - ``requires_ollama``, ``requires_matlab``, ``browser``
     - A local Ollama, a MATLAB with the AT toolbox, a Playwright chromium.
     - the same conftest check; skips when absent

.. list-table:: The board: which lane scores the test
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``agentic_benchmark``
     - A genuine capability task: the agent must choose and sequence tools,
       reason about results and produce an answer the assertions or the judge
       evaluate. A weak model plausibly fails it.
     - the board's capability lane, the headline number
   * - ``harness_benchmark``
     - An agent runs, but the assertion is OSPREY's own behaviour: a hook
       fires, a write is refused, an audit record lands. Any responding model
       passes it.
     - the board's harness lane, scored apart so it never pads a score
   * - ``channel_finder_benchmark``
     - The channel-finder quality benchmarks, with an LLM scoring lane.
     - the ``channel-finder-benchmarks`` job, on demand

Every e2e test in the board's scope carries exactly one of the first two, or its
file is named in the board's exclusion list with a reason; a lane gate fails on an
unmarked test at matrix start and as a unit test.

.. list-table:: Opt-outs and control
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``real_server_launch``, ``real_workspace_watcher``,
       ``real_http_posters``, ``real_graph_index``, ``real_fetch``
     - Opts one test out of a conftest stub that otherwise keeps a
       process-global side effect inside the test that caused it: a companion
       server, a filesystem observer, a live POST, an index build, a fetch.
     - the conftest that installs the stub
   * - ``flaky``
     - Rerun on failure, for nondeterminism outside the process: model sampling
       in e2e, OS event delivery in a watcher test. Not for a flaky unit test,
       and never on a Tier 2 task, where every attempt is scored.
     - pytest-rerunfailures
   * - ``xdist_group``
     - Pins a set of files to one worker, for example everything that starts a
       container.
     - the ``--dist loadgroup`` scheduler
   * - ``e2e_smoke``, ``e2e_tutorial``, ``e2e_services``
     - Categories inside ``tests/e2e/``: the quick smoke set, the tutorial
       walkthroughs, the tests that need local MCP service infrastructure.
     - selected by hand with ``-m``

There is no ``e2e`` marker: the directory is the selector, and ``pytest tests/e2e/``
is how the suite is named.

Scenarios: how a fault is staged
================================

A scenario is what makes an agentic test more than a smoke test. The agent is given
an operator's question and a machine with a fault in it, and has to find the fault
from the instruments, the history and the logbook, the way an operator would. The
machinery that stages the fault is the same one a facility uses to run demos and
training sessions on the :doc:`Virtual Accelerator <virtual-accelerator>`.

.. raw:: html
   :file: ../_diagrams/testing-scenario-pipeline.html

**The bundle.** A simulated machine is one ``machine.json`` that defines every
channel with its baseline, noise, bounds and slow texture, and one directory per
scenario under ``data/simulation/scenarios/<name>/``. The scenario's ``scenario.json``
says how the machine departs from nominal: ``overrides`` replace live values;
``archiver`` events place a step, a ramp or a spike on a channel's history; a
``drivers`` block with ``couple`` and ``noise`` gives several channels one shared
slow cause, so that a cavity's temperature, reflected power and frequency move
together the way they do when one cooling manifold is blocked; and a ``physics``
block seeds lattice faults into the Virtual Accelerator, a BPM with inverted
polarity or a corrector at half gain. An optional ``logbook.json`` carries the
narrative, entries with authors, tags and attached plots, dated in days ago rather
than on a calendar. The exact schema, every rule and every refusal are in the
:doc:`/reference/contracts/simulation-bundle`.

**Composition and one clock.** ``osprey sim apply <names>`` activates a set.
``nominal`` is always active and always first; the other names keep their order.
No two active scenarios may touch one channel, so two faults compose only when they
are about different instruments, and the shipped pair that both script the beam
current is refused rather than composed order-dependently. Apply fixes one anchor
time, T0, and resolves everything against it: an archiver event placed with
``at_when`` and the logbook entry that narrates it land on the same instant, a
recurring ``at_time`` event fires at that time of day on every date in the window
read, and a relative ``at_offset`` counts from T0. A test freezes T0 with ``--now``
so a run is reproducible.

**What apply writes.** Four things, in this order, and nothing if any check
refuses. The active set goes into ``var/agent_data/simulation/active_scenarios``,
written whole by rename, which every reader re-reads when it changes. A stored
archive, the deployment's memory, is rewritten so that history shows the fault and
switching away leaves nothing behind. The ARIEL logbook is purged and reseeded from
the active scenarios' own entries, plots drawn first so a picture of the week before
an entry carries that week's dates. And a ``physics`` block is rendered into the
deployment's ``.env`` as ``VA_BPM_ERRORS`` and ``VA_CORR_GAIN``, which the Virtual
Accelerator reads when its container is created, so a lattice fault takes effect at
the next ``osprey up`` rather than at the next poll.

**Two substrates, one engine.** The mock connector composes live reads and
synthesized history from the bundle in process. The Virtual Accelerator container
mounts the same ``data/simulation/`` and the same state directory, polls the state
file about once a second, and drives every channel that has no lattice physics and no
setpoint echo from the same ``SimulationEngine`` class the mock uses; there is no
second implementation. The two differ in one visible way. On the mock, applying a
scenario clears the values written during the session. On the Virtual Accelerator a
setpoint lives in the IOC's own record and survives a switch, because a real machine
does not forget its setpoints when the world changes; only the engine-composed
channels refresh. ``tests/va/e2e/test_scenario_reload.py`` proves that a switch
reaches the running IOC within its documented bound.

**How a scenario test is built.** The prompt is what an operator would say and
nothing more: *"We lost about 5 mA of beam yesterday around 14:32. Did the vacuum do
anything weird around then?"* No channel names, no time-window shape, no hint which
subsystem to look at. The test builds its repo at build tier 3 so every simulated
instrument is discoverable, calls ``activate_scenarios`` so telemetry and narrative
are seeded against one clock, and then closes the cheap route to a right answer: the
bundle's ``description`` names the fault outright and the agent's working directory
is the render, so the main agent loses ``Bash``, ``Glob`` and ``Grep`` for the run,
and a physics scenario's bundle is deleted from both trees once its fault is
materialized in the container's environment. A telemetry scenario's bundle cannot
be deleted, because it *is* the live overlay, so those tests rely on the withheld
tools alone. Grading is the two layers every agentic test uses: the floor checks the
trace, that a channel-finder call and an archiver read on a vacuum gauge happened,
and the judge checks the prose, that the answer commits to sector 7 rather than to
"a gauge somewhere". The scenario tests carry ``flaky(reruns=2)`` to absorb the rare
stochastic bail-out.

**The contract comes first.** Every shipped bundle's statistical signature is pinned
under ``tests/simulation/``, deterministically and without a model: the SR07 gauge
is the single sector anti-correlated with the beam current and leads the runner-up
by a wide margin; the cavity's reflected power tracks its temperature; the three
excursions land on the days the logbook narrates. When a scenario test regresses,
run those first. If they pass, the data substrate is sound and the miss is the
agent's; fix the preset's agent context or tool descriptions. If they fail, fix the
bundle, never the prompt.

.. list-table:: The shipped bundles (``control_assistant`` preset)
   :header-rows: 1
   :widths: 18 36 22 24

   * - Scenario
     - The fault
     - Mechanism
     - Staged by
   * - ``nominal``
     - None. Carries the ambient logbook, including decoy entries that mention a
       Sector 4 vacuum spike and "Sector 7" in a magnet context, weeks back.
     - always active
     - every apply
   * - ``vacuum-burst``
     - A Sector-7 pressure spike with a coincident 5 mA beam loss at 14:32:08,
       recurring daily. Pure telemetry, no logbook.
     - ``archiver`` events, ``at_time``
     - ``test_vacuum_burst_scenario.py``
   * - ``rf-thermal``
     - Three CAVITY01 thermal excursions from a blocked cooling manifold, the
       last a beam dump four days back; a three-entry logbook arc from trip to
       repair. Telemetry and narrative must converge on one cavity.
     - ``archiver`` events, ``at_when``, ``logbook.json`` with plots
     - ``test_rf_cavity_correlation_scenario.py``
   * - ``rf-thermal-live``
     - Live thermal wander on cavity 01: one slow driver moves body
       temperature, reflected power and frequency together, with drifting
       gain.
     - ``drivers``, ``couple``, ``noise``
     - ``test_scenario_drivers.py`` pins the correlation bands
   * - ``bpm-polarity``
     - BPM 17's horizontal readback is inverted. The orbit is right; only the
       reading lies, so single-channel telemetry never shows it.
     - ``physics``, applied at Virtual Accelerator boot
     - ``osprey sim apply`` on a Virtual Accelerator deployment; the render
       step is pinned by ``test_scenario_physics_render.py``
   * - ``orm-dual-fault``
     - Two faults at once, each invisible to single-channel telemetry: the
       inverted BPM and corrector HCM01 at half gain. Only an orbit-response
       measurement resolves both.
     - ``physics``
     - as above

A bundle need not ship with a preset. The Tier 2 chart-reading task carries its own
two bundles under ``tests/e2e/fixtures/`` and plants them into a stack it deploys,
one a recent beam-current dip in the archive and one a year-old loss that survives
only as a plot on a logbook entry, so the answer can be read only off the picture.

Benchmarks
==========

**The model matrix.** ``scripts/benchmark/matrix.py`` runs the e2e suite against
every model in ``matrix.yaml``, three seeds each, forcing every model alias the
agent uses onto the model under test and routing OpenAI-protocol models through a
translation proxy. The dashboard scores two lanes by marker: the **capability
lane** is the headline, the **harness lane** is reported beside it so that its
near-constant passes never pad a score. Claude models run as the reference ceiling:
they should score close to everything on a clean stack, so a miss by a reference
model is read first as a bug on OSPREY's side, and the board has found several that
way. The judge is always a reachable remote model and never the model under test.

**Tier 2** is the board's home in CI: the stack-free tasks run in the
``tier2-model-benchmark`` job on the ``run_benchmarks`` dispatch, and the task that
deploys a real stack runs on the board only, where parallel cells take turns on a
host-wide lock.

**The channel-finder benchmarks** exercise each channel-finder paradigm against the
preset's channel corpus with the query set the build materialized into the render,
and assert aggregate quality thresholds. They run in the ``channel-finder-benchmarks``
job on the same dispatch, never on a pull request, because the LLM scoring lane is
too expensive to gate on.

.. seealso::

   :doc:`/contributing/development-setup`
      Running the suites, the local check scripts and the front-end rails.

   :doc:`/contributing/workflow`
      When the model-spending lanes run and what a pull request needs.

   :doc:`/reference/contracts/simulation-bundle`
      Every key, rule and refusal of a scenario bundle.

   :doc:`virtual-accelerator`
      The machine the scenarios reach over real EPICS.

   :doc:`safety-chain`
      The hooks tested as code before any agent runs them.
