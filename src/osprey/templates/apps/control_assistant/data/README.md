# Project Data Directory

Everything the agent reads from disk lives here: channel databases, benchmark
query sets, and the facility's sources and scenarios. These are your files —
edit them freely.

## Directory Structure

The channel-finder indexes are not here: each is a view the build writes from
`facility/` into the render. As shipped by the preset:

```
data/
├── channel_databases/
│   └── examples/                         # Hierarchy-shape examples
├── benchmarks/
│   └── cross_paradigm/queries/           # Benchmark query sources, one per channel-finder pipeline
├── ariel/
│   ├── vocabulary.yml                    # Logbook shorthand -> the words entries use
│   └── README.md                         # Vocabulary format walkthrough
├── facility/                              # The facility's authored sources
│   ├── knowledge/                         # Markdown knowledge bundle
│   └── scenarios/                         # Simulation scenarios
```

`osprey build` copies the benchmark query file matching `channel_finder_mode`
to `benchmarks/queries.json`: `in_context_queries.json` for `in_context`,
`tree_queries.json` for every other mode. Each channel-finder index is the view
the build writes at its own path; nothing is flattened. The render drops the
`benchmarks/cross_paradigm/` subtree.

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
