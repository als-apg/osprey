# Claim taxonomy and verification strategies

A page is not one claim; it is dozens of small ones, each true or false on its own. The
first job on any page is to shred the prose into **atomic, individually falsifiable
claims**, then pick the right way to verify each. This is what lets one skill check a
how-to page, a reference table, and an architecture overview without a recipe per page.

## What counts as an atomic claim

A claim is atomic if a single check proves it true or false. Split compound sentences:
"the `execute` tool takes `code` and `description`, defaults `execution_mode` to
`readonly`, and truncates output to 500 characters" is three claims, each checked
differently.

Extract claims from every part of the page: prose, parameter tables, `code-block`
snippets in any language, admonitions, and the steps of a procedure. A snippet is a claim
too; a documented command or config block asserts "this works as written". A procedure
asserts its order.

## Claim types and how to verify each

| Type | Example | Verification | Run? |
|---|---|---|---|
| **symbol-exists** | "registered on the `python` server", "`LimitsValidator`" | Grep and read the source; confirm the module, class, function, or tool exists where stated. | static |
| **signature** | "`save_artifact(obj, title="Untitled", ...)`" | Read the `def`; compare parameter names, order, and defaults exactly. | static |
| **default-value** | "`execution_mode` defaults to `readonly`", "timeout 600 s" | Find where the default is defined (function default, model field, config schema, template). Compare. | static |
| **return-field** | "returns `artifact_ids`, `gallery_url`, …" | Read the response builder; confirm each field is produced. | static |
| **config-key** | "`services.jupyter.containers.read.port_host`" | Two parts: the key parses, and the code reads it. Grep the loader and accessor for the path; a key nothing reads is dead documentation. `scripts/config_key_manifest.yml` is the authoritative list of live and deleted keys. | static + probe |
| **cli-command** | "`osprey deploy --profile …`" | Run it, or `--help` when it would change state, with the checkout's interpreter; confirm the verb and every flag are real. The CLI root is a lazy group: enumerate through `list_commands` and `get_command`, not by grepping decorators. | **run** |
| **runtime-behaviour** | "readonly blocks writes", "output truncated to 500 chars", "falls back to a local subprocess" | Build a minimal probe on an existing test fixture and capture the output. | **run** |
| **import / API surface** | "user code calls `read_channel()` from `osprey.runtime`" | Import it; confirm the callables exist and are the intended public surface. | **run** |
| **generated-project** | "the generated `config.yml` ships `X` under `Y`", "the README tells you to …" | `osprey init` into a temp directory, or render the template in-process, and read the output. Never build a real project. | **run** |
| **procedure** | "1. do A; 2. do B; 3. observe C" | Check each step as its own claim, then check the order: is B still possible before A, is C still the observable result. | static + run |
| **install / setup** | "included in the default install", "no extra setup for local mode" | Check `pyproject.toml` extras and the install docs' guard workflow. | static |

Cross-references, labels, and markup are not claims for this skill; the Sphinx build with
warnings as errors owns them.

## Per section, where the truth lives

Derive it live every run: grep the page's most specific symbols and let the source tell
you. As orientation only:

| Section | Kind of claim that dominates | First place to look |
|---|---|---|
| `getting-started/` | commands, generated project, first-run behaviour | the CLI root, `src/osprey/templates/` |
| `how-to/` | procedures, config keys, runtime behaviour | the subsystem the page names, its tests |
| `reference/` | keys, flags, defaults, tables | the loader, the click commands, the manifest |
| `architecture/` | symbols, module boundaries, data flow | the package layout, `__init__` exports |
| `contributing/` | scripts, CI job names, check tiers | `scripts/`, `.github/workflows/`, `pyproject.toml` |

## Reuse the test harness before writing a probe

The fixtures under `tests/` already construct executors, registries, connectors, and
configs in-process. A probe built on an existing fixture is more trustworthy than one you
hand-roll. For any subsystem:

```bash
find tests -ipath "*<subsystem>*" \( -name "conftest.py" -o -name "test_*.py" \)
```

A runtime claim is best verified by a tiny pytest-style script that imports the same
fixtures and asserts the documented behaviour, run with the checkout's own interpreter.

## Building a probe when no fixture fits

One claim per probe. Set up the smallest context the claim needs, exercise exactly the
documented path, assert the documented outcome, and print the actual outcome either way.
Probes live in the scratch directory, never under `tests/`. Capture stdout, stderr, and
the exit code; the captured output is the evidence quoted in the report.

## When a probe cannot run

A claim that needs a container runtime, a live control system, a kernel, or a credential
is recorded under **Not run** with the exact blocker. Before giving up, do the static part:
container mode cannot be exercised without a runtime, but the client code and the config
keys can still be confirmed, and only the runtime portion is listed as not run. Partial
verification beats a blank; a guessed outcome is worse than either.

## Coverage accounting

Every lane's first line is `checked: <n> pages, <m> claims, <k> run, <j> not run`. A page
that passed with most of its runtime claims not run is a different signal from one fully
exercised, and the maintainer must be able to tell the two apart.
