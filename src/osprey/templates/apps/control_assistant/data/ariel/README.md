# Logbook Vocabulary

Operators search in shorthand. Logbook entries are written in words. A search
for `t/s the bpm offset` will not find the entry that says "troubleshot the
beam position monitor offset" — stemming clips endings, it does not know that
`bpm` and "beam position monitor" are the same thing.

`vocabulary.yml` is where you tell ARIEL that they are. Before a search runs,
the query is rewritten using the concepts in that file, so the shorthand
spelling reaches the prose spelling. It is plain dictionary matching — no
model, no network call, microseconds — and every rewrite is reported back to
the operator and to the OSPREY agent, so a hit that looks surprising always
arrives with the reason it appeared.

Expansion applies to all three search modes: keyword, semantic and hybrid.

The file shipped here is an **example**, not your vocabulary. Its twenty
concepts are the ones most storage-ring facilities share, so it is useful on
day one — but it is meant to be edited down to what your control room
actually says. The orbit and ring group (storage ring, beam based alignment,
orbit response matrix, fast/slow orbit feedback) is storage-ring-specific —
a linac or FEL should delete it first.

## The format

One top-level key, `concepts`, holding a non-empty list. Each entry has three
required keys and no others:

```yaml
concepts:
  - canonical: beam position monitor   # the words logbook prose uses
    kind: acronym                      # acronym | shorthand
    forms: [bpm, bpms]                 # what operators type instead

  - canonical: troubleshoot
    kind: shorthand
    forms: [ts, t/s]
```

- **`canonical`** — write it exactly as it appears in entry text. It is what a
  matched form is rewritten into, so if your entries say "beam position
  monitor", that is the canonical. Canonicals must be unique.
- **`kind`** — `acronym` for a genuine initialism, `shorthand` for a clipped or
  slang spelling of an ordinary word. It selects a direction gate (below).
- **`forms`** — one or more spellings operators type. A form may not repeat its
  own canonical.

Anything else is refused rather than ignored: an unknown key, a missing field,
an unknown kind, a duplicate canonical, an empty list. The check reports every
problem in one pass rather than one per run.

## How text is matched

The file and the query go through the same rule before anything is compared:

- lowercase
- `-` becomes a space, so `Beam-Position Monitor` and `beam position monitor`
  are one phrase
- `/` is **kept**, so shorthand like `t/s` and `i/o` stays a single token
- runs of whitespace collapse to one

Nothing else is stripped, and no stemming happens here — PostgreSQL's stemmer
runs afterwards, on the text this produces. Multi-word forms are matched as
phrases, longest first.

## The two direction gates

Matching a form and adding its canonical is **always on**: type `bpm` and the
search also finds "beam position monitor". The reverse direction — spelling the
canonical out and *also* searching the short form — is what `kind` gates, via
two switches in `config.yml`:

| Setting | Applies to | Default |
|---------|-----------|---------|
| `ariel.vocabulary.canonical_to_acronym` | `kind: acronym` concepts | `true` |
| `ariel.vocabulary.canonical_to_shorthand` | `kind: shorthand` concepts | `false` |

The defaults are deliberately asymmetric. An acronym means one thing, so a
search for "beam position monitor" should also reach the entries that wrote
"BPM" — free recall. An ordinary word is not so lucky: expanding "calibration"
into `cal` pulls in every entry that happened to abbreviate something else that
way, and costs more precision than it buys back in recall. Turn it on only if
your logbook is written in shorthand and searched in prose.

## Forms that mean two things

Binding the same form to two concepts is legal. A query containing it expands
to **both** canonicals, which widens the search rather than picking a winner:

```yaml
  - canonical: troubleshoot
    kind: shorthand
    forms: [ts]

  - canonical: timing system
    kind: acronym
    forms: [ts]
```

`vocab-check` warns about it, listing every canonical the form reaches, so the
widening is a decision you made rather than something you discover from odd
results. The example file ships no ambiguous forms.

## Check every edit

```bash
osprey ariel vocab-check data/ariel/vocabulary.yml
```

It needs no database. Exit code 0 means the file loads; 1 means it lists the
errors. Warnings never fail the check — they cover the things that are legal
but worth knowing: an ambiguous form, a form that can never fire because a
longer one always wins, and a word PostgreSQL's English text search discards
outright (so expanding to it would match nothing).

Run it before you deploy. The file is read **once**, when `config.yml` is
parsed, so a broken file is a startup failure and a loud one: the ARIEL panel
comes up in configuration-invalid mode with the errors and the fix on screen
and the search form disabled, search returns an error naming
`ariel.vocabulary.path`, and the MCP server refuses to start. Browsing,
entries, status and publishing keep working throughout. Recovery is to fix the
file, or set `enabled: false`, and restart — and if the vocabulary is baked
into a deployed image, fix it here in the source and rebuild.

## Turning it off

Whole-deployment, in `config.yml`:

```yaml
ariel:
  vocabulary:
    enabled: false
```

Nothing is read at all, so a stale `path` left behind is inert rather than a
startup failure.

Per search, without touching the config: every search surface takes an
`expand_query` argument. Leave it unset and it follows
`ariel.vocabulary.expand_by_default`; pass `false` and that one search runs on
exactly the words you typed, with no expansions reported. It is the switch to
reach for when you want to see what a query finds on its own — and the OSPREY
agent can use it the same way when a rewrite looks like it widened a search too
far.
