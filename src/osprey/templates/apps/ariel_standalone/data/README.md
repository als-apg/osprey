# Data

This deployment's data tree. `osprey build` copies it into the build zone, and
everything in it is yours to replace.

```
data/
├── demo_machine.ttl          # Knowledge-graph corpus (services.graphdb.ttl_path)
└── logbook_seed/             # Demo logbook (ariel.demo_narrative)
    └── <scenario>/
        ├── logbook.json      # The scenario's logbook entries
        └── plots/            # Pictures those entries attach
```

`logbook_seed/` holds the logbook narrative of every control-assistant demo
scenario, in the scenario-bundle logbook format. `osprey up` seeds it into an
empty logbook, pictures included; `osprey ariel quickstart` does the same and
then adds embeddings. To use your own logbook instead, point `ariel.ingestion`
at it and remove `ariel.demo_narrative` from `profile.yml`.
