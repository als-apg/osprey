# Project Data Directory

Everything the agent reads from disk lives here: benchmark query sets and the
facility's sources and scenarios. These are your files — edit them freely.

## Directory Structure

As shipped by the preset:

```
data/
├── benchmarks/
│   └── cross_paradigm/queries/           # Benchmark query sources, one per channel-finder pipeline
├── canopy/
│   └── library/screens/                  # CANOPY screens (facility overview and sections)
├── ariel/
│   ├── vocabulary.yml                    # Logbook shorthand -> the words entries use
│   └── README.md                         # Vocabulary format walkthrough
├── landing/
│   └── working-safely.md                 # A notice on the web terminal's landing page
└── facility/                             # The facility's authored sources
    ├── identity.yaml                     # The facility's code and display name
    ├── records/                          # places, devices, channels, groups
    ├── models.yaml                       # Each simulated model, its engine and wiring
    ├── decks/                            # The deck each model runs
    ├── measurement/                      # The measurement kinds each model offers
    ├── limits.yaml                       # Write limits, one record per channel
    ├── seeds.yaml                        # Simulated channels' nominal values and noise
    ├── scenarios/                        # Simulation scenarios, one YAML file each
    └── knowledge/                        # Markdown knowledge bundle
```

An importer writes its own layer under `facility/imported/<layer>/`:
`osprey facility import mml` writes `facility/imported/mml/`. Edit the
authored files; an import rewrites its own directory.

`osprey build` reads `facility/` and writes the facility file and its views
into the render, never back here: `facility.json` at the render root, and under
its `data/` the limits view `channel_limits.json`, the facts view
`facility_facts.json`, the simulator view `simulator/`, the graph view
`graph/facility.ttl` and the channel-finder view `channel_finder/<mode>.json`.
`osprey facility show` prints the facility the tree builds and each view's
path.

`osprey build` copies the benchmark query file matching `channel_finder_mode`
to `benchmarks/queries.json`: `in_context_queries.json` for `in_context`,
`tree_queries.json` for every other mode. Each channel-finder index is the view
the build writes at its own path; nothing is flattened. The render drops the
`benchmarks/cross_paradigm/` subtree.

## CANOPY screens

`canopy/library/` is the deployment's CANOPY screen library: the control
system screens the web terminal's CANOPY panel lists and opens. Screens live
under `canopy/library/screens/` as Screen-IR documents (`*.screen.json`), one
file per screen, addressed by their path without the suffix. The template
ships:

- `facility/overview` — the facility at a glance: the plan of gun, linac,
  booster and storage ring with BPM status, the storage-ring status, RF,
  magnet supplies, injector and booster; each storage-ring sector, the
  booster, BTS and GTL (gun, linac and LTB) opens its screen;
- `br/overview` — the booster: its BPM status, the dipole and quadrupole
  ramps, the septa and kicker, and the booster DCCT;
- `sr/sector-01` to `sr/sector-12` — one storage-ring sector each: its
  straight and arc in beam order, every magnet, corrector and BPM with its
  status and readbacks, and the sector's vacuum;
- `sr/orbit-feedback` — storage-ring orbit feedback: the BPM orbit and
  corrector readbacks, the DCCT, the orbit RMS per plane as CANOPY Formula Engine
  formulas, and one corrector setpoint input;
- the transfer-line panels `gtl/GTL` and `bts/BTS`, with the screens and
  images they embed under `common/`.

The screens bind the simulator's channels (see the simulator view the build
writes under `data/simulator/`);
readouts show their units and alarm state.

The library is baked into the image: after editing a screen here, run
`osprey build` and then `osprey up`; `osprey up` alone does not refresh it.

## Database Paradigms

`channel_finder_mode` in the build profile picks one of three ways to organize
the same channel namespace as a file; all three are views of the same
facility records. The mode's fourth value, `graph`, is not one of them: it
answers from the facility knowledge graph rather than a channel database. Its
corpus is `data/graph/facility.ttl`, the graph view the build writes from
`facility/`, seeded into the `services.graphdb` store.

### `in_context` — flat structure

Best for fewer than about 1,000 channels. The whole database fits in the
agent's context, so lookup is direct semantic search over a flat list of
channels.

### `hierarchical` — nested structure

Best for more than about 1,000 channels. The agent navigates the hierarchy
level by level instead of loading everything at once: the facility's place
words by depth, then class, device and leaf.

### `middle_layer` — functional structure

An MML-organized functional hierarchy: System / Family / Field / Subfield.
Navigation mirrors the way operators reason about devices rather than the way
the control system names them.

## Database Tools

Database tools are `osprey channel-finder` CLI subcommands. Each reads the
active database from `config.yml` unless you pass `--database`.

Validate database format and structure:

```bash
osprey channel-finder validate
osprey channel-finder validate --database data/channel_finder/hierarchical.json
```

Preview database contents:

```bash
osprey channel-finder preview
osprey channel-finder preview --database data/channel_finder/hierarchical.json
```

## Benchmarks

Evaluate channel-finder accuracy against the query set in
`data/benchmarks/queries.json`:

```bash
# Full query set
osprey channel-finder benchmark --model anthropic/claude-haiku-4-5

# A slice of the query set
osprey channel-finder benchmark --model anthropic/claude-haiku-4-5 --queries 0:10

# Repeat each query to measure run-to-run variance
osprey channel-finder benchmark --model anthropic/claude-haiku-4-5 --runs-per-query 3
```

Results are written to `data/benchmarks/results/` as JSON reports carrying
per-query outcomes, accuracy, timing, and cost.
