# Channel Finder Data Directory

This directory contains channel databases, benchmark datasets, and tools for managing them.

## Directory Structure

```
data/
├── raw/                          # Raw address data (CSV files)
│   ├── CSV_EXAMPLE.csv          # Example CSV format
│   └── address_list.csv         # Sample address list data
├── channel_databases/            # Processed channel databases
│   ├── in_context.json          # In-context pipeline database
│   ├── hierarchical.json        # Hierarchical pipeline database
│   └── TEMPLATE_EXAMPLE.json    # Database format example
└── benchmarks/                   # Benchmark datasets and results
    ├── datasets/                # Test query datasets
    │   ├── basic_queries.json
    │   └── comprehensive.json
    └── results/                 # Benchmark result files
```

## Database Tools

Database tools are available as native ``osprey channel-finder`` CLI subcommands:

### 1. Build Channel Database

Build a channel database from CSV files:

```bash
osprey channel-finder build-database --csv data/raw/address_list.csv
osprey channel-finder build-database --csv data/raw/address_list.csv --use-llm
```

### 2. Validate Database

Validate database format and structure:

```bash
osprey channel-finder validate
osprey channel-finder validate --database data/channel_databases/in_context.json
```

### 3. Preview Database

Preview database contents:

```bash
osprey channel-finder preview
osprey channel-finder preview --database data/channel_databases/in_context.json
```

## CSV Format

Your CSV file should have these columns:

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

## Database Formats

### In-Context Database (Flat Structure)
Best for < 1,000 channels. Direct semantic search.

### Hierarchical Database (Nested Structure)
Best for > 1,000 channels. Multi-level navigation.

## Benchmarks

Run benchmarks to evaluate channel finder performance:

```bash
# Run all benchmarks
python scripts/run_benchmarks.py

# Run specific dataset
python scripts/run_benchmarks.py --dataset basic_queries

# Run specific queries
python scripts/run_benchmarks.py --queries 0:10
```

Results are saved in `data/benchmarks/results/` with:
- Detailed JSON report
- Summary text file
- Performance metrics
- Cost analysis

