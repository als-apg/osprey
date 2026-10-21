# NARAD seed schemas

The facility schema (`src/osprey/facility/schema/core.yaml` and `vocabulary.yaml`)
starts from the NARAD LinkML schemas of the `narad-uitf` repository. The four files
here are copied byte for byte from its `schemas/` directory at commit `02f64d5`
(`git show 02f64d5:schemas/<file>`) and are never edited.

`../loosenings.py` reads them to write `src/osprey/facility/schema/loosenings.yaml`,
which says what became of every slot they require. `tests/facility/test_schema_loosenings.py`
pins the digests below.

| file | sha256 |
|---|---|
| `canonical_ingest.yaml` | `2efa863505615c555449d575343534463eda7ed23fe296760eeeed78776e9a91` |
| `facility_bindings.yaml` | `252208465d48ef582b031fc61be2371754eab6dec10582c933e42b54e53c6849` |
| `concept_vocabulary.yaml` | `48b1dc6cf7b96721706e9b0bdfc286a6224e59f6fa123d2275ffda86dee14b64` |
| `shared_semantics.yaml` | `f1836e33402f1ffb63d7518ef966374beb2581780cfbf7f3a3b57710717f6629` |

To move to a newer NARAD commit: copy the four files from that commit, update the
commit and the digests here and in the test, and run `../loosenings.py`.
