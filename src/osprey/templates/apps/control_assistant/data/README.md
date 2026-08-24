# Project Data Directory

Everything the agent reads from disk lives here: channel databases, benchmark
query sets, facility knowledge, and simulation scenarios. These are your files —
edit them freely.

## Directory Structure

As shipped by the preset, the channel-finder artifacts are staged for all tiers
and all three file-backed paradigms:

```
data/
├── raw/                                   # CSV address data (in_context build path)
│   ├── CSV_EXAMPLE.csv                   # Example CSV format
│   └── address_list.csv                  # Sample address list
├── channel_databases/
│   ├── tiers/tier{1,3}/<paradigm>.json   # Staged databases, one per paradigm
│   ├── examples/                         # Hierarchy-shape examples
│   └── TEMPLATE_EXAMPLE.json             # Database format example
├── benchmarks/
│   └── cross_paradigm/queries/           # Staged query sets, one per tier
├── channel_limits.json                    # Per-channel write limits
├── machine_state_channels.json            # Channels shown in the machine-state view
├── demo_machine.ttl                       # Knowledge-graph corpus (graph paradigm)
├── ariel/
│   ├── vocabulary.yml                    # Logbook shorthand -> the words entries use
│   └── README.md                         # Vocabulary format walkthrough
├── facility_knowledge/                    # Markdown knowledge bundle
├── lattice/                               # Accelerator lattice files
└── simulation/                            # Mock-connector scenarios
```

`osprey build` collapses the staged sets down to the ones your build profile
selected. It copies the active paradigm's database to a flat
`channel_databases/<paradigm>.json`, copies the tier-matching query file to a
flat `benchmarks/queries.json`, and removes the `tiers/` and
`benchmarks/cross_paradigm/` subtrees. `raw/` survives only for `in_context`
builds — the CSV format cannot express a nested database, so it is dead weight
for the other paradigms.

## Database Paradigms

`channel_finder_mode` in the build profile picks one of three ways to organize
the same channel namespace as a file. All three describe addresses in the
`RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD` grammar. The mode's fourth value,
`graph`, is not one of them: it answers from the facility knowledge graph
rather than a channel database. Its corpus is `demo_machine.ttl` in this
directory, seeded into the `services.graphdb` store.

### `in_context` — flat structure

Best for fewer than about 1,000 channels. The whole database fits in the
agent's context, so lookup is direct semantic search over a flat list of channel
and template entries. This is the only paradigm with a CSV build path.

### `hierarchical` — nested structure

Best for more than about 1,000 channels. The agent navigates the
`RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD` hierarchy level by level instead of
loading everything at once.

### `middle_layer` — functional structure

An MML-organized functional hierarchy: System / Family / Field / Subfield.
Navigation mirrors the way operators reason about devices rather than the way
the control system names them.

## Database Tools

Database tools are `osprey channel-finder` CLI subcommands. Each reads the
active database from `config.yml` unless you pass `--database`.

Build a database from CSV (`in_context` only):

```bash
osprey channel-finder build-database --csv data/raw/address_list.csv
osprey channel-finder build-database --csv data/raw/address_list.csv --use-llm
```

Validate database format and structure:

```bash
osprey channel-finder validate
osprey channel-finder validate --database data/channel_databases/hierarchical.json
```

Preview database contents:

```bash
osprey channel-finder preview
osprey channel-finder preview --database data/channel_databases/hierarchical.json
```

## CSV Format

The CSV consumed by `build-database` has these columns:

```csv
channel,address,description
ChannelName1,PV:ADDRESS:1,Description of the channel
ChannelName2,PV:ADDRESS:2,Description of the channel
```

For template-based channels (with multiple instances):

```csv
template_base_name,base_address,description,instances_start,instances_end,sub_channels
QuadrupoleMagnet,Q{instance:02d},Quadrupole magnets,1,17,SetPoint|ReadBack
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
