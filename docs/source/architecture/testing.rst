.. _architecture-testing:

=======
Testing
=======

OSPREY is a harness around an agent that can reach a machine, so the suite has to
prove several different things: that the code is correct, that the presets it ships
build and boot, that the simulated machine behaves like one, that an agent with a
real model does the job, and that the whole thing comes up as containers the way a
facility runs it. This page is the inventory: what tests exist, when each of them
runs, how a fault is staged for the agent to find, what the markers do, and what
the benchmarks measure. How to write a test and the fixture house style are in the
repository's ``tests/README.md``; how to run the agentic suite is in
``tests/e2e/README.md``.

.. raw:: html
   :file: ../_diagrams/testing-suite-map.html

Four words carry the rest of the page:

- A **check** is one command declared in a ``repo_checks.py`` module and run by
  ``scripts/check.py``: a linter, the parallel test sweep, the docs build.
  ``scripts/check.py list`` prints them all.
- A **lane** is one CI job that is not a call of the check command: a preset
  build, a container stack, an agentic run. The ``LANES`` table in
  ``scripts/repo_checks.py`` lists every lane, what it needs and which files it
  hosts.
- A **marker** says where a test runs instead of where its directory would put
  it, or names a resource it needs.
- A **tier** is one of two things, never a CI stage. ``scripts/check.py`` has two
  tiers, ``quick`` and ``full``. An agentic test is in Tier 1 (basic) or Tier 2
  (benchmark).

What the suite is made of
=========================

**The unit suite** is everything under ``tests/`` outside ``tests/e2e/``, plus the
connectors package's own suite under ``packages/osprey-connectors/tests``. It is
one check, ``pytest``, and runs under parallel workers, four in CI, with every
file kept on one worker. It reaches no model, and resets any process-global state it
touches. CI splits it into five shards. Three kinds of test leave the parallel
sweep and run alone, in the ``serial-suites`` job: the real-terminal ``pty``
tests, which read the bytes an operator would see; the ``wallclock`` tests,
which hold a time budget; and the live Channel Access suites, one process per
module. A set of guard tests holds the repository to its own rules, for example
that the agent SDK is imported only inside ``src/osprey/agent_runner/``
(``tests/test_harness_fence.py``), that a test needing a credential lives only
under ``tests/e2e/`` (``tests/test_api_marker_placement.py``), and that the
published documentation still matches the code (``tests/docs/``).

**The hooks** are tested as code before any agent runs them. Each hook has its own
tests under ``tests/hooks/``, and ``test_hook_chain_integration.py`` runs the
write chain in the order Claude Code would: the writes check, the limits check,
then the approval prompt. A red here is a hole in the
:doc:`safety chain <safety-chain>` that no model could have papered over.

**The presets** have three files under ``tests/integration/`` that the parallel
sweep leaves to two lanes. The ``static-checks`` lane runs
``test_preset_static.py``, which holds every shipped preset to the artifact
registry and every skill's tool references to the MCP tools that exist, and
``test_preset_build.py``, which initialises and builds each preset without
installing it. The ``build-boot-smoke`` lane runs ``test_build_boot.py``, one
case per preset and one for the demo with an application turned on: a real
``osprey build`` that installs offline, then every server in the rendered
``.mcp.json``, the application's own included, must answer ``initialize`` and
``tools/list`` with the documented tools. Each case is a job on Python 3.11;
the application case, the only install of an application's wheel beside
osprey, is also a job on 3.12, the unit suite's Python on pull requests. Both
lanes run on every push and every pull request.

**The machine** is the simulator and its contracts. ``tests/simulation/`` and
``tests/facility/`` hold the scenario machinery: how scenarios build into the
simulator view, compose and apply. ``tests/connectors/`` reads the shipped
scenarios through the mock archiver and pins the signature each one documents,
without a model. ``tests/va/`` is the
:doc:`Virtual Accelerator <virtual-accelerator>`'s own suite; its live Channel
Access modules run through ``scripts/va/live_ca/gate.py``, which fails on a
skipped module. ``tests/va/e2e/`` proves the container from where a client
stands, in the ``va-live-e2e`` lane: the limits and approval gates hold, a
corrector write moves a downstream BPM, the container serves the same
values as the in-process simulator, and a scenario switch reaches the running
IOC.

**The agent** is ``tests/e2e/``. A test builds a deployment repo with ``osprey
init`` and ``osprey build``, then runs a real Claude Code session through the
SDK against it, with the pinned Claude Code version and one named provider; a
run that names no provider is refused before its first test. Grading is in two
layers: a deterministic floor over the tool trace decides whether the agent did
the thing, and an LLM judge is given only what a trace cannot show, whether the
agent described what it did without inventing findings. ``e2e-tests`` is the
shared lane: it runs every module no other lane claims. The
``agentic-per-preset`` lane runs one canonical flow per preset
(``test_preset_agentic.py``). These lanes spend model tokens, so they run only
when asked; see `When each part runs`_.

**The scripted gate tests** are the part of ``tests/e2e/`` that needs no model. A
test whose subject is a safety gate needs one forbidden tool call to be made,
and a well-behaved model will not make it. ``tests/e2e/scripted_model.py`` is a
local endpoint that makes exactly the tool calls a test lists, and the real
Claude Code delivers them through the rendered settings, the hook chain and the
MCP servers. These modules carry ``model_free`` and run in the ``e2e-no-model``
lane on every pull request, fork pull requests included; the lane fails on a
single skip.

**The stacks** are the container-backed e2e files, each hosted by a lane of its
own or by one cell of the ``stack-e2e`` matrix: the Bluesky, Tiled, archiver and
optimizer stacks, the dispatch stacks, the privilege split inside the image, the
auth perimeter behind nginx, the chat bridges, and the two-shape boot (bridge
and host network) under podman-compose. The ``LANES`` table is the one list of
which lane hosts which module: the shared lane derives its ``--ignore`` list
from it, and the wiring test, ``tests/deployment/test_ci_workflow_wiring.py``,
parses the workflow and proves that no ``dockerbuild`` file is left in the
shared lane.

**The front end** has its own checks: ``vitest``, ``tsc``, ``eslint`` and
``prettier`` in the ``frontend-js`` job, and the real-browser Playwright tests
under ``tests/interfaces/`` (``-m browser``) in the ``visual-theming`` lanes.

**The board** is ``scripts/benchmark/``: the agentic tests run against a matrix of
models and seeds. It measures and never gates; see `Benchmarks`_.

**In deployment**, a facility keeps testing itself. ``osprey health`` runs
diagnostics over configuration, the Python environment, containers and the
enabled services, plus the probes the facility declares, read in the browser's
Health panel, on the command line or by a CI job. ``osprey audit`` turns an AI
reviewer on a built profile and reports the safety risks it finds in
permissions, hooks and MCP server configuration. See
:doc:`/how-to/health-and-monitoring/index` and :doc:`/reference/cli`.

When each part runs
===================

Every job is required once it runs: the ``All CI Checks Passed`` gate has no
advisory lane. What differs is the event that starts a lane.

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Runs on
     - What
     - Why there
   * - every push to ``main`` and every pull request
     - the checks (lint, types, the unit shards, ``serial-suites``, the docs and
       package builds, the front-end checks), ``static-checks``,
       ``build-boot-smoke``, ``dependency-floor``, ``optimizer``, the connectors
       lanes, the ``visual-theming`` lanes, ``teams-slack-bridge-e2e``,
       ``plugin-smoke``, ``canopy-contract``
     - no secret, no model
   * - every pull request, forks included
     - ``e2e-no-model``, ``agent-record-e2e``
     - containers, but no secret and no image push
   * - a pull request from a branch of the repository
     - ``deploy-e2e``, ``stack-e2e``, ``va-live-e2e``, ``orm-roundtrip-e2e``,
       ``privilege-split-e2e``, ``dockerfile-e2e``, ``full-chain-auth-e2e``,
       ``two-shape-boot-e2e``, ``qmd-sidecar-e2e``,
       ``multi-user-deploy-lifecycle-e2e`` (only when the pull request's base
       is ``main``)
     - they build and deploy images; house policy keeps that to same-repo
       branches
   * - a pull request labelled ``full-ci``
     - ``e2e-tests``, ``agentic-per-preset``, ``dispatch-stack-e2e``,
       ``target-switch-agentic-e2e``, ``dockerfile-agent-e2e``,
       ``nextcloud-talk-bridge-e2e``, ``gchat-bridge-e2e``
     - they spend model tokens; ``e2e-tests`` also runs on a manual dispatch
       of the workflow that does not set ``run_benchmarks``
   * - the ``run_benchmarks`` dispatch
     - ``benchmarks``
     - it measures the model, not the change

A pull request without the label runs the unit suite on Python 3.12 only;
``main``, a ``full-ci`` pull request and a Dependabot pull request run 3.11,
3.12 and 3.14. A push to ``main`` runs the first row only. ``main`` gets the
other lanes from the ``revalidate_secret_lanes`` dispatch, which
``.github/workflows/main-full-ci.yml`` starts once a day when ``main`` has
moved. :doc:`/contributing/workflow` has the commands and the cost.

Locally, ``scripts/check.py quick`` runs the fast checks and the parallel sweep
without the ``slow`` tests. ``scripts/check.py full`` adds the slow tests, each
serial suite and the builds. ``scripts/check.py run <id>`` runs one check as CI
does. Before opening a pull request, ``scripts/check.py quick --changed
origin/main`` runs the checks and tests the diff touches and names the CI lanes
it touches. Neither tier runs a lane: the preset, container and agentic lanes
are CI's, so a green ``quick`` says nothing about them.
:doc:`/contributing/workflow` and :doc:`/contributing/development-setup` have
the commands.

Scenarios: how a fault is staged
================================

A scenario is what makes an agentic test more than a smoke test. The agent is given
an operator's question and a machine with a fault in it, and has to find the fault
from the instruments, the history and the logbook, the way an operator would. The
machinery that stages the fault is the same one a facility uses to run demos and
training sessions on the :doc:`Virtual Accelerator <virtual-accelerator>`.

.. raw:: html
   :file: ../_diagrams/testing-scenario-pipeline.html

**The scenario file.** A scenario is one YAML file,
``data/facility/simulation/scenarios/<name>.yaml``, and says how the machine departs from
its baseline: ``overrides`` set live values, ``faults`` write a fault into a
physics model, ``channel_faults`` make a channel stuck, frozen or disconnected,
``archiver`` events shape a channel's history, ``drivers`` and ``couple`` give
several channels one slow shared cause, and ``logbook`` entries narrate it, dated
in days ago and not on a calendar. ``osprey build`` puts the scenarios into the
simulator view under ``build/data/simulator/``. Every key is in
:ref:`reference-scenario-files`.

**Composition and one clock.** ``osprey sim apply <names>`` makes the names the
active set; ``nominal`` is always in it. Two scenarios compose only when no two
of them write the same target, a channel or a model's fault, and a set that
breaks the rule is refused before anything is written. Archive events are not a
target, so ``vacuum-burst`` and ``rf-thermal``, which both script the beam
current, compose. Apply fixes one anchor time and resolves everything against
it: an archive event placed with ``at_when`` and the logbook entry that narrates
it land on the same instant. ``--now`` freezes the anchor, so a seeded run is
reproducible.

**What apply writes.** The active set goes into
``var/agent_data/simulation/active_scenarios``. The ARIEL logbook is purged and
reseeded from the active scenarios' own entries, and the affected windows of the
stored archive are rewritten from their archive events, so live values, history
and narrative tell one story.

**Two venues, one simulator.** The simulator serves the same view in process and
from its container over Channel Access (:ref:`va-two-venues`). Either re-reads
the state file when it changes and starts again at the new set's start state;
setpoints written during the session are dropped in both.
``tests/simulation/test_sim_apply_live.py`` holds the in-process half,
``tests/va/e2e/test_scenario_reload.py`` and ``test_sim_apply_va.py`` the
container, and ``tests/va/e2e/test_substrate_parity.py`` holds the two to each
other, channel for channel.

**How a scenario test is built.** The prompt is what an operator would say and
nothing more: *"We lost about 5 mA of beam yesterday around 14:32. Did the vacuum
do anything weird around then?"* No channel names, no time-window shape, no hint
which subsystem to look at. The test builds a project from the ``demo`` preset
and calls ``activate_scenarios``, which applies the scenario and seeds the
logbook against one clock. Then it closes the cheap route to a right answer. The
scenario file's ``description`` names the fault outright and the agent's working
directory is the render, so the agent loses ``Bash``, ``Glob`` and ``Grep`` for
the run. A test whose fault already lives in a running container's own view also
deletes the scenario's files from the repo (``conceal_scenario_ground_truth``);
a scenario the in-process simulator serves cannot be deleted, because the
simulator reads it on every pass. Grading is the two layers every agentic test
uses: the floor checks the trace, that a channel-finder call and an archive read
on a vacuum gauge happened, and the judge checks the prose, that the answer
commits to sector 7.

The two demo scenario tests need a running ARIEL logbook database, which a
GitHub runner does not have. They skip in CI and run locally and on the
benchmark board, and carry ``flaky(reruns=2)`` for a local run; the board turns
reruns off.

**The contract comes first.** ``tests/connectors/test_archiver_scenario_signatures.py``
pins each demo scenario's signature without a model, read through the mock
archiver the agent's tools use: sector 7 is the most anti-correlated of the
twelve sectors with the beam current, by a set margin; cavity 1 has three
excursions in the week before the investigation entry, the last one the trip;
nothing happens after the repair. When a scenario test regresses, run that file
first. If it passes, the data is sound and the miss is the agent's: fix the
preset's agent context or tool descriptions. If it fails, fix the scenario file,
never the prompt.

.. list-table:: The demo preset's scenarios and what tests them
   :header-rows: 1
   :widths: 18 44 38

   * - Scenario
     - The fault
     - Tested by
   * - ``nominal``
     - None. It carries the background logbook of the weeks before.
     - every apply
   * - ``vacuum-burst``
     - A vacuum burst in sector 7 with a beam-current drop at 14:32:08,
       recurring daily in the archive. No logbook narrative.
     - the signature file; ``tests/e2e/test_vacuum_burst_scenario.py``
   * - ``rf-thermal``
     - Three thermal excursions on cavity 1, the last a beam dump, with a
       logbook arc from trip to repair.
     - the signature file;
       ``tests/e2e/test_rf_cavity_correlation_scenario.py``
   * - ``rf-thermal-live``
     - Live thermal wander on cavity 1 from one slow shared driver.
     - the signature file, which pins the correlation bands
   * - ``bpm-polarity``
     - BPM 17 reads inverted. No single channel is out of range.
     - ``tests/simulation/test_sim_apply_live.py``,
       ``tests/va/e2e/test_sim_apply_va.py``
   * - ``orm-dual-fault``
     - The BPM inversion and corrector HCM01 at half gain at once.
     - ``tests/facility/test_scenarios.py``
   * - ``demo-history``
     - Three past agent sessions with the archive events and logbook entries
       behind them.
     - ``tests/simulation/test_demo_history_honesty.py``

A scenario need not ship with a preset. The Tier 2 chart-reading task carries its
own two scenario files under ``tests/e2e/fixtures/chart_value_reading/scenarios/``
and plants them into a stack it deploys: one a recent beam-current dip in the
archive, one an older loss that survives only as a plot on a logbook entry, so
the answer can be read only off the picture.

Every e2e stack built from ``demo`` also states the scenarios it starts in,
because its assertions read live values off the simulator and the preset's
default set moves on its own. ``tests/deployment/test_e2e_scenario_pin.py``
holds that.

Markers
=======

Markers are declared in ``pyproject.toml``. A marked test and the lane or step
that reads the marker are a pair, and the wiring test holds the pairs that take
a test out of the parallel sweep, because losing either half is silent: a suite
nobody runs under a green check.

.. list-table:: Hops: the marker moves the test to another step or lane
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``slow``
     - Too long for the fast local loop. Still runs in CI.
     - ``scripts/check.py quick`` deselects it
   * - ``pty``
     - Runs an ``osprey`` verb on a real terminal and asserts on repaint
       cadence, which three other workers loading the box would distort.
     - the parallel sweep deselects it; the ``pytest-pty`` check runs it alone
   * - ``wallclock``
     - Asserts an absolute time budget. A clock read beside loaded workers
       measures the workers.
     - the parallel sweep deselects it; the ``pytest-wallclock-unit`` check
       times the image-free budgets alone, and ``qmd-sidecar-e2e`` the ones
       also marked ``dockerbuild``
   * - ``canopy``
     - Needs CANOPY and PyDMConverter at the pinned commits, and refuses to
       skip without them.
     - the parallel sweep deselects it; the ``canopy-contract`` job installs
       both and runs it
   * - ``dockerbuild``
     - Runs a real container build. Skips when Docker is absent.
     - kept out of the shared e2e lane; each file is hosted by a lane of its
       own
   * - ``qmd``
     - Needs the qmd sidecar image and refuses to skip without it, because a
       lane that skipped on a missing image would report success having
       proved nothing.
     - ``qmd-sidecar-e2e``, which builds the image; a lane without one
       deselects it
   * - ``model_free``
     - An e2e test that reaches no model, so it runs in a session that names
       no provider.
     - the ``e2e-no-model`` and ``agent-record-e2e`` lanes, whose selection
       equals this set
   * - ``browser``
     - A real-browser Playwright test. Skips without chromium.
     - the ``visual-theming`` lanes select it

.. list-table:: Resources: the test needs something the host may not have
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``requires_e2e_provider``
     - Builds its deployment with the provider the run named in
       ``OSPREY_E2E_PROVIDER`` and needs that provider's credential.
     - the e2e lanes, which name a provider; a run naming none is refused
   * - ``requires_api``, ``requires_als_apg``, ``requires_cborg``,
       ``requires_anthropic``, ``requires_openai``, ``requires_google``
     - Needs a provider key: any of them, or the one it names. ``cborg`` is
       local-only: its IP allowlist blocks GitHub runners.
     - ``tests/conftest.py`` turns a missing key into a skip
   * - ``requires_typesafe``, ``requires_openrouter``
     - Calls a live API and needs its key exported to the run.
     - the same conftest check
   * - ``requires_ollama``, ``requires_matlab``
     - A local Ollama, a MATLAB with the AT toolbox.
     - the same conftest check; skips when absent

A ``requires_*`` marker is refused outside ``tests/e2e/`` unless its resource is
keyless and local, so the unit suite never makes a credentialed call whatever
keys a developer has exported.

.. list-table:: The agentic markers: who is on trial
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``agentic(tier, title, asks, passes_if)``
     - The model under test drives the agent and can fail the task. Tier 1,
       basic: the stack works and a simple workflow goes through. Tier 2,
       benchmark: a harder task a capable model may still miss. The text
       fields say in plain words what the operator asks and what passes.
     - the benchmark board, which runs only these tests; collection refuses an
       incomplete marker
   * - ``harness_benchmark``
     - An agent runs, but the assertion is OSPREY's own behaviour: a hook
       fires, a write is refused, an audit record lands.
     - CI; the board leaves these out
   * - ``channel_finder_benchmark``
     - The channel-finder quality benchmarks.
     - the ``benchmarks`` job, on demand

Every e2e test in the board's scope carries exactly one of the first two, or its
file is named in ``scripts/benchmark/matrix_e2e_config.json`` with a reason. A
lane gate fails on an unmarked test at the start of a board cell and as a unit
test (``tests/benchmark/test_matrix_lanes.py``). Every test under
``tests/e2e/tier2/`` is Tier 2; not every Tier 2 test lives there.

.. list-table:: Opt-outs and control
   :header-rows: 1
   :widths: 22 44 34

   * - Marker
     - What it says
     - Who reads it
   * - ``real_server_launch``, ``real_workspace_watcher``,
       ``real_http_posters``, ``real_graph_index``, ``real_fetch``,
       ``real_bounded_run``
     - Opts one test out of a conftest stub that otherwise keeps a
       process-global side effect inside the test that caused it: a companion
       server, a filesystem observer, a live POST, an index build, a fetch, a
       bounded subprocess run.
     - the conftest that installs the stub
   * - ``flaky``
     - Rerun on failure, for nondeterminism outside the process: model sampling
       in e2e, OS event delivery in a watcher test. Not for a flaky unit test.
     - pytest-rerunfailures; the board turns reruns off
   * - ``xdist_group``
     - Pins a set of files to one worker, for example everything that starts a
       container.
     - the ``--dist loadgroup`` scheduler
   * - ``e2e_smoke``, ``e2e_tutorial``, ``e2e_services``
     - Categories inside ``tests/e2e/``: the quick smoke set, the tutorial
       walkthroughs, the tests that need local MCP service infrastructure.
     - selected by hand with ``-m``

The e2e suite is selected by path, ``pytest tests/e2e/``. An ``e2e`` marker is
declared, but nothing selects by it.

Benchmarks
==========

**The model matrix.** ``scripts/benchmark/matrix.py`` runs the agentic tests against
every model in ``matrix.yaml``, three seeds each by default, forcing every model
alias the agent uses onto the model under test and routing OpenAI-protocol models
through a translation proxy. Each test runs once, with reruns off, so every
result is a first attempt. The dashboard groups results by tier: Tier 1 reports
time and tokens per task with an all-green check, and Tier 2 shows each test's
result per seed. Claude models run as the reference ceiling: they should score
close to everything on a clean stack, so a miss by a reference model is read
first as a bug on OSPREY's side, and the board has found several that way. The
judge is always a reachable remote model and never the model under test.

**Tier 2 in CI.** The stack-free Tier 2 tests run in the ``benchmarks`` job on the
``run_benchmarks`` dispatch. The two that deploy a real stack,
``test_chart_value_reading.py`` and ``test_channel_directory_agentic.py``, run on the
board only, where parallel cells take turns on a host-wide lock. A miss in
Tier 2 says something about the model, not the change, and no pull-request lane
runs the directory.

**The channel-finder benchmarks** run each channel-finder pipeline against the
preset's channels and assert aggregate quality thresholds. They run in the same
``benchmarks`` job, never on a pull request, because scoring them with a model
on every push costs too much to gate on.

.. seealso::

   :doc:`/contributing/development-setup`
      Running the checks and the front-end rails.

   :doc:`/contributing/workflow`
      When the model-spending lanes run and what a pull request needs.

   :ref:`how-to-run-scenarios`
      Applying and writing scenarios.

   :ref:`reference-scenario-files`
      Every key of a scenario file.

   :doc:`virtual-accelerator`
      The machine the scenarios reach over real EPICS.

   :doc:`safety-chain`
      The hooks tested as code before any agent runs them.
