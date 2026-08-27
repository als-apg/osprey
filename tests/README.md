# Writing tests for OSPREY

The unit suite runs in parallel: four worker processes, each importing test files
and running them in an order nobody controls. A test that changes something
process-global — an environment variable, the logging setup, the working
directory, a cached module-level value — can therefore break a test in a file it
has never heard of, on a run where the two happened to land on the same worker.

This page is the working agreement that keeps that from happening. Read the
section you need; the last section is a checklist for anyone adding a batch of
new tests.

---

## 1. Running the suite

```bash
# Parallel — what CI runs, and what your change has to pass
pytest tests/ --ignore=tests/e2e -n 4 --dist loadgroup

# Serial — for debugging one failure, or reading full output in order
pytest tests/ --ignore=tests/e2e
```

Both must pass. Parallel is the real gate; serial is the one that gives you
readable output when something is wrong.

The three helper scripts already use the parallel invocation, so you normally
just run one of them:

| Script | What it is for |
|---|---|
| `scripts/quick_check.sh` | Fast loop while working; skips `@pytest.mark.slow` |
| `scripts/premerge_check.sh` | Quiet pass/fail before opening a PR |
| `scripts/ci_check.sh` | Full CI mirror, with coverage |

**E2E tests are separate.** Run them with `pytest tests/e2e/`, never with
`-m e2e`.

**CI caps every unit test at 600 seconds** (`--timeout=600
--timeout-method=signal`, on the lane's pytest line only — an ini default would
apply to the e2e lanes too, which pass no `--timeout`). Ten minutes clears a
cold container image pull, so no file needs an exemption; what it ends is a
genuine hang, which used to run out the 40-minute step cap and cancel the job
with a truncated log and no name on the test. A timed-out test fails with
`Failed: Timeout (...) from pytest-timeout`, **not** `AssertionError` — so a
`@pytest.mark.flaky(only_rerun=["AssertionError"])` marker will not rerun it.
The two `flaky` markers in the unit suite rerun all failures and are unaffected.

### What `--dist loadgroup` does here

`tests/conftest.py` installs a small scheduler (`FileOrGroupScheduling`) that
changes how work is handed to workers:

- **A file with no `xdist_group` marker** goes to one worker as a unit. Every
  test in the file runs on the same worker, in file order. This is the default
  and it is what you want: module-level state and ordered test pairs keep
  working.
- **A file marked `@pytest.mark.xdist_group("name")`** joins that named group,
  and the whole group goes to one worker. Use this only when *separate files*
  must not run at the same time as each other.

There is exactly **one** sanctioned group today: `"docker"`, shared by the ten
files that start real containers (the MongoDB connector tests and the
Postgres-backed ARIEL tests). One group means one worker, one testcontainers
session, one reaper — see section 6.

**Adding a group is a last resort, and every use carries a comment saying why.**
Copy the shape from `tests/connectors/test_mongodb_archiver_connector.py`:

```python
# xdist_group("docker"): the session ``mongodb_container`` fixture starts a real
# container, and this file shares the group with the Postgres-backed ARIEL tests so
# every container start lands on one worker. ...
pytestmark = pytest.mark.xdist_group("docker")
```

A group is the right tool when two files contend for one *external* resource (a
database, a port, a daemon). It is the wrong tool for leaky state inside the
process — fix that with a fixture instead (section 2), because a group only
serializes the files you remembered to mark.

### Reading a parallel failure

- Node IDs get a `@group` suffix under `loadgroup`
  (`test_connect_success@docker`). That is cosmetic.
- `-x` and `--maxfail` are approximate under xdist: workers finish the test
  they are already running, so you may see a few results after the first
  failure.
- If a failure looks placement-dependent, re-run the failing file alone and
  serially. If it passes alone but fails in the suite, something leaked — start
  at section 2.

---

## 2. Isolation fixtures: the house style

When your tests change global state, undo it in an autouse fixture. The house
style has three parts:

1. **`autouse=True`** — isolation is not opt-in. A fixture you have to remember
   to request is a fixture someone will forget.
2. **Reset before *and* after `yield`** — resetting only afterwards trusts every
   other test to have been well behaved. Resetting on the way in too means your
   test is correct even when it runs second.
3. **Name the leak in the docstring** — say what global state exists, and what
   breaks without the guard. The next person needs to know whether they may
   delete it.

`tests/deployment/conftest.py` is the template:

```python
@pytest.fixture(autouse=True)
def reset_runtime_cache():
    """Isolate every test from the process-wide docker-vs-podman memo.

    ``runtime_helper`` detects the container runtime once per process and
    memoizes the result. Many tests here (and under ``web_terminals``) fake
    ``runtime_helper.subprocess.run``, so the first one to run would otherwise
    pin its fake answer — or the host's real runtime — for every test after it.
    """
    _reset_runtime_cache()
    yield
    _reset_runtime_cache()
```

Use `monkeypatch` or `unittest.mock` for patching. There is no `pytest-mock` in
this repo.

**Reset through a named seam in `src/`, not by reaching into globals from the
test.** If the seam you need does not exist, add it next to the state it resets.
The ones that exist today:

- `osprey.deployment.runtime_helper.reset_runtime_cache()`
- `osprey.deployment.compose_generator._reset_wheel_build_cache()` — defined in
  `wheel_build.py` and re-exported by `compose_generator`; import it from either,
  but `wheel_build.py` is where it lives if you need to read it.
- `osprey.connectors.factory.isolated_connector_registries(clear=True)` — a
  context manager that snapshots and restores the connector registries. Use it
  instead of `ConnectorFactory._registry.clear()`, which destroys registrations
  other files depend on.
- `osprey.health.offload.reset_abandoned_state()`

### What the root conftest already guards

`tests/conftest.py` runs these around **every** test in the suite. You do not
need to repeat them, but you should know they exist so you can tell "my test
leaked" from "my test was leaked on".

| Guard | Protects against |
|---|---|
| `restore_environ` | Any write to `os.environ` — including raw writes that bypass `monkeypatch`. Snapshots and restores the whole environment. Also clears `OSPREY_CONFIG` and `CONFIG_FILE` on the way in, so a developer's exported shell variable can't change a run. |
| `reset_state_between_tests` | The registry and the config caches. |
| `restore_cwd` | A leaked `os.chdir` — usually into a deleted `tmp_path`. The compose generator resolves paths relative to the cwd, so a stray cwd silently changes what later tests render. |
| `restore_root_logging` | Root log level, the OSPREY `RichHandler`, and the third-party logger levels `configure_logging()` raises. See section 3. |
| `reset_health_offload_state` | The health runner's cumulative abandoned-thread counter. Left set, in-process `osprey health` calls take an `os._exit` branch that kills the whole xdist worker. |

`restore_environ` is declared first in the file on purpose: same-scope autouse
fixtures set up in declaration order, so its snapshot window encloses
`monkeypatch`'s own undo rather than the other way round.

`restore_root_logging` compares against a baseline captured when `conftest.py`
was imported — not a per-test snapshot. A per-test snapshot cannot see far
enough back, because pytest sets up higher-scoped fixtures *before*
function-scoped ones, so a module-scoped fixture that configures logging would
already be baked into the snapshot and preserved instead of undone.

### Per-area conftests

- **`tests/cli/conftest.py`** — CLI tests drive Click commands in-process through
  `CliRunner`, so whatever a command does to the process happens inside pytest.
  `_guard_os_exit` turns an in-process `os._exit` into an ordinary assertion
  failure instead of a dead worker. `isolated_home` (request it explicitly)
  points both `Path.home()` and `$HOME` at a tmp directory — use it for anything
  that writes under the home directory.
- **`tests/hooks/conftest.py`** — hook subprocesses get a *curated* environment
  built from an allowlist, not a copy of `os.environ`. A hook run therefore
  cannot inherit config state that an earlier test left in the process. If a
  hook starts reading a new variable, add it to `_HOOK_VARS`.

### Marker vocabulary

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`.
`--strict-markers` is listed in `addopts`, but on the pinned pytest it only bites
when you pass it on the command line yourself: in a normal run an undeclared
marker is a `PytestUnknownMarkWarning`, not an error. So declare your marker, and
run `pytest --strict-markers …` when you want a typo to fail collection rather
than scroll past in the warnings summary. Four groups matter day to day:

- **`requires_<resource>`** (`requires_ollama`, `requires_als_apg`, …) — declares
  an external dependency. `pytest_collection_modifyitems` in `tests/conftest.py`
  turns it into a real skip when the resource is missing. To add a resource,
  append a zero-argument, side-effect-free predicate to `_RESOURCE_CHECKS` there
  and declare the marker in `pyproject.toml`.
- **`slow`** — deselected by `scripts/quick_check.sh` (`-m "not slow"`). Mark a
  test slow if it would make the fast loop annoying to run; it still runs in CI.
- **`xdist_group`** — the worker pin from section 1. Self-registered by xdist, so
  it needs no declaration.
- **`unit` / `integration`** — descriptive labels for what a test talks to.

Async tests need no marker: `asyncio_mode = "auto"`, so an `async def` test just
runs.

---

## 3. Logging and `caplog`

Entry points call `configure_logging()` once at startup. Libraries and imports
never call it — importing OSPREY leaves logging exactly as it found it.

`configure_logging()` is **strictly additive**. It sets the root level, adds the
OSPREY `RichHandler` only if no `RichHandler` is present, and raises six
third-party loggers to WARNING. It never removes a handler it did not install.
That is what keeps `caplog` alive: `caplog`'s capture handler survives a test
that reaches an entry point.

**Logs go to stderr, always** (`Console(stderr=True)`). stdout is reserved for
program output, which is what makes MCP stdio JSON-RPC and `--json` output safe.

What this buys you when writing tests:

- **`caplog` works regardless of test order or worker**, including in tests that
  invoke a CLI entry point in-process.
- **Assert stdout purity on stdout only.** Use `result.stdout`, never click's
  `result.output` — in click 8.2+ `output` folds stderr in, so a stray log byte
  on stdout would be invisible. `TestJsonStdoutPurity` in
  `tests/cli/test_query_cmd.py` is the worked example: it asserts that the whole
  of stdout is the JSON payload, not merely that it contains it. See also
  `tests/cli/test_health_cmd.py` and `tests/mcp_server/test_stdout_purity.py`.
- **stdout purity is conventional, not guaranteed by construction.** Because
  `configure_logging()` is additive, a host application's own stdout handler is
  deliberately left in place. That is the trade that keeps `caplog` working. It
  means the guarantee is "OSPREY does not write logs to stdout", not "nothing
  can".
- **Never assert a third-party logger's level against a value captured in
  another test.** `import litellm` raises the `httpx` logger to WARNING as an
  import side effect. Six OSPREY modules import it: three inside functions, so
  the level changes the first time that code path runs, and three — the LiteLLM
  provider adapter, the in-context channel-finder query tool, and the benchmark
  harness — at module level, so merely importing one of those modules is enough.
  The eager three are the stronger hazard, because collection alone can trigger
  them. Either way the value one test observes is not a valid expectation for
  the next, and WARNING is indistinguishable from what `configure_logging()`
  itself sets. Compare against `tests.conftest._PRISTINE_LOGGING` instead; that
  is the guard's actual contract. `tests/utils/test_logging_guard.py` shows the
  pattern.

---

## 4. Patching rules

### Naming the module under test does not scope the patch

This is the rule that catches people out, so it is worth stating flatly:

```python
patch("subprocess.run")                        # obviously global
patch("osprey.cli.chat_cmd.subprocess.run")   # equally global — just looks scoped
```

After a module does `import subprocess`, `module.subprocess` **is** the stdlib
module object. Both lines above mutate the same attribute on the same object.
While either patch is active, the fake is visible to every thread and every
background server in the worker.

The general rule: *"patch where it's used"* buys you scoping **only** for
`from x import y` bindings, where the importing module holds its own reference.
For a plain `import x`, the module attribute is the shared module object and the
patch is global no matter how you spell the target. This is equally true for
`os`, `shutil`, `socket`, and `subprocess`.

### The scoped-subprocess helper

`tests/cli/_scoped_subprocess.py` does the scoping properly, by rebinding the
importing module's own `subprocess` *name* to a stand-in. Its docstring is the
reference:

> Patch `subprocess` for one module without touching the stdlib module.
>
> A patch target that names the module under test is *not* automatically scoped
> to it. After a module does `import subprocess`, `<module>.subprocess` **is**
> the stdlib module object, so both of these mutate `sys.modules["subprocess"]`:
>
> ```python
> patch("subprocess.run")                       # obviously global
> patch("osprey.cli.chat_cmd.subprocess.run")   # equally global, looks scoped
> ```
>
> While either is active the fake is visible to every daemon thread and
> background server in the worker, which is the leak this helper exists to
> avoid. What does scope the fake is rebinding the *importing module's own*
> `subprocess` name to a stand-in object, which is what `patch_subprocess()`
> does. The stand-in carries the real exception classes across, so
> `except subprocess.TimeoutExpired` in the code under test still catches.
>
> Both spawning entry points are covered. `run` is always a fresh `MagicMock`;
> `Popen` becomes one only when the caller asks, via `popen=`. That default is
> deliberate: a globally-patched `Popen` swallows the `git init` the exemplar
> repo fixture runs — and anything else spawning in this worker — so a test
> that needs a fake child process must get it scoped, but a test that only
> cares about `run` must keep the real `Popen` working underneath it.
>
> This requires the module under test to import `subprocess` at module level. A
> function-local `import subprocess` re-reads `sys.modules` on every call and
> cannot be intercepted this way at all.
>
> Usage — as a decorator, where the injected argument is the stand-in:
>
> ```python
> @patch_subprocess("osprey.cli.chat_cmd", return_value=Mock(returncode=0))
> def test_launch(self, fake_subprocess, ...):
>     ...
>     fake_subprocess.run.assert_called_once()
> ```
>
> or as a context manager:
>
> ```python
> with patch_subprocess("osprey.deployment.container_lifecycle", side_effect=[a, b]) as fake:
>     ...
> ```
>
> To fake a detached child, pass `popen=` — either a ready-made callable, or
> `popen=True` for a plain `MagicMock` whose `return_value` you configure:
>
> ```python
> with patch_subprocess("osprey.cli.web_cmd", popen=True) as fake:
>     fake.Popen.return_value.pid = 4321
>     ...
>     assert fake.Popen.call_args.args[0] == [...]
> ```

### Known debt

`tests/deployment/` still has roughly 80 module-qualified patch sites spelled
like this:

```python
monkeypatch.setattr(runtime_helper.subprocess, "run", _fake_run)
```

By the rule above these are unscoped — `runtime_helper.subprocess` is the stdlib
module. They are recorded here as debt, not as an example. Do not copy them, and
if you touch one of those tests, move it to `patch_subprocess()`.

A handful of sites go one step further and patch the stdlib module itself:

```python
monkeypatch.setattr(subprocess, "run", no_git)          # tests/cli/test_init_verb.py
monkeypatch.setattr("subprocess.run", ...)              # tests/cli/test_build_zero_arg.py
```

Same debt, wider blast radius: for the duration of that test EVERY importer of
`subprocess` gets the fake, so a second, unrelated subprocess call inside the
test is silently answered by it instead of failing. `monkeypatch` still restores
the attribute afterwards, so the leak does not cross test boundaries — but
within the test there is nothing to tell you the fake answered more than you
aimed it at. Also `patch_subprocess()` when you touch one.

---

## 5. Import-time mutations

A test module must not change process state while it is being imported.

Import-time mutations are the one kind of pollution a fixture cannot undo: they
run during collection, in whichever worker happens to import the file, before
any fixture of any scope can snapshot what they overwrote.

`tests/infrastructure/test_import_time_audit.py` enforces this. It parses every
file under `tests/` with the `ast` module and collects import-time writes to
`os.environ`, `sys.path`, `sys.modules`, and sockets. It follows calls to
functions defined in the same module (calling one at import time runs its body
then), stops at function boundaries, and skips `if __name__ == "__main__"`
blocks.

The audit asserts **exact equality with the whitelist, in both directions**. A
new mutation fails it, and so does removing a whitelisted one — so the list
cannot rot into a record of things that used to be true.

The whitelist today is small, and each entry is genuinely impossible to move
into a fixture: the two `tests/va` files (libca latches `EPICS_CA_*` at
C-library init), `tests/documentation/test_workflow_autodoc.py` (`sys.path` for
a top-level import of a non-installed extension), and three `sys.modules`
registrations for scripts loaded by path.

If you truly cannot avoid one, add the file to `WHITELIST` **and** put an inline
comment at the mutation itself starting `# import-time required because` — a
second test checks that the comment is there. Say what forces it to run at
import time. "It was easier" is not a reason; move it into a fixture instead.

---

## 6. Ports and containers

### Ports

Take ports from the shared helpers in `src/osprey/interfaces/_serving.py`:

```python
from osprey.interfaces._serving import free_port, wait_for_port

port = free_port()          # an unused TCP port on 127.0.0.1
...
wait_for_port(port)         # block until the server accepts, or raise
```

Never hardcode a port number. Four workers running the same hardcoded port is a
guaranteed intermittent failure. Some older files still carry a private
`_free_port` copy; those are untouched debt — new and modified tests use the
shared helpers.

### Containers

Container-starting files share `xdist_group("docker")` so that only one worker
ever has a testcontainers session open. Each xdist worker is a separate process
with its own session and its own ryuk reaper, and two concurrent container
starts race the Docker daemon's port mapper.

Every container fixture must stop its own containers in teardown. That is a real
requirement, not a nicety — it is what makes the reaper optional (see below).

**Running the parallel suite locally on macOS** needs two things. Both were
found the hard way during the soak — the 20-run stability exercise that
qualified this invocation, described at the end of this page — where each one
produced red runs that had nothing to do with the test code.

1. **No crash-looping container on the Docker daemon.** A container in a restart
   loop churns Docker Desktop's port mapper and will intermittently break *any*
   testcontainers start. Check before a long run:

   ```bash
   docker ps --filter status=restarting
   ```

   The offender during the soak had a restart count of 245 and belonged to an
   unrelated project — it does not have to be anything you started.

2. **`TESTCONTAINERS_RYUK_DISABLED=1`.** Ryuk's own startup races Docker
   Desktop's port forwarder under multi-worker CPU load, and the pressure grows
   with the worker count. Ryuk only provides crash-cleanup insurance — it reaps
   containers if the test process dies without running teardown — and every
   fixture here stops its own containers. This was verified, not assumed: ten
   container starts with ryuk off left zero testcontainers residue.

   **CI keeps ryuk enabled.** Linux daemons have no Docker Desktop port
   forwarder in the path and do not show the race. This is a local-macOS
   recommendation only; do not put it in CI config.

### Reading a red `docker` group

Because the ten files share one group, a single transient container hiccup takes
out the entire group. The failure list will be long and it will be misleading:

- The **first** failure is the cause — typically
  `ConnectionError: Port mapping for container <id> and port 8080 is not available`.
- The trailing `409 Conflict: container name ... already in use` errors are
  cascade. testcontainers left a half-started reaper behind, and every later
  container request in the same process re-derived the same name and collided
  with the corpse.

Read past the 409s. They are a symptom of the first failure, not a second bug.

---

## 7. Adding a batch of new tests

Checklist for a new test area:

- [ ] **New test filenames are unique across all of `tests/`.** Most test
      directories have no `__init__.py`, so a file's module name is just its
      basename, and two files anywhere in `tests/` sharing one collide — pytest
      refuses to collect the second. Check with `find tests -name "<basename>.py"`;
      prefer a name qualified by the module under test. The two gates disagree
      about this failure, which is what makes it easy to ship: serially it aborts
      the whole run (`Interrupted: 1 error during collection`), but under `-n 4`
      it degrades — exit 1, and the colliding file's tests simply never run,
      which later reads as a coverage shortfall rather than a collection failure.
- [ ] **Reuse the shared fixtures and helpers.** `free_port()` / `wait_for_port()`
      for ports, the existing reset seams for global state, `patch_subprocess()`
      for subprocess fakes, `isolated_home` for anything touching `$HOME`.
- [ ] **New global state gets a public reset seam in `src/`**, plus an autouse
      fixture in the area's `conftest.py` in the house style — reset before and
      after `yield`, leak named in the docstring.
- [ ] **No import-time side effects.** If you think you need one, you almost
      certainly do not; if you genuinely do, whitelist it *and* justify it
      inline (section 5).
- [ ] **No new `xdist_group`** unless separate files contend for one external
      resource, and then only with a comment saying which resource and why.
- [ ] **Almost no flaky markers.** `@pytest.mark.flaky` reruns are routine in
      `tests/e2e`. In the unit suite they are a near-absolute no: a
      non-deterministic unit test is usually a leak that has not been found yet,
      and a rerun hides it. The bar for an exception is that the
      nondeterminism is genuinely outside the process and cannot be waited on.
      Two tests clear it today —
      `tests/interfaces/artifacts/test_store_watcher.py` reruns two
      filesystem-watcher tests because the OS occasionally never delivers the
      inotify/FSEvents event at all, which no amount of condition-based waiting
      can fix. Both carry a comment saying exactly that. If you cannot write
      that comment for your test, you have a leak, not a flake.
- [ ] **Assert stdout on `result.stdout`**, never on click's `result.output`.
- [ ] **Both gates pass**: serially, and under `-n 4 --dist loadgroup`. Passing
      one is not evidence for the other.

### If a new test flakes under `-n 4`

1. Run the file alone, serially. If it passes, the problem is shared state, not
   your assertions.
2. Run the file alongside its neighbours (`pytest tests/<area> -n 4 --dist loadgroup`)
   to find the partner.
3. Ask what is global: an environment variable, a module-level cache, the
   logging setup, the cwd, a port, a container, a singleton.
4. Fix it with a reset seam plus an autouse fixture. Reach for `xdist_group` only
   when the contended thing is genuinely outside the process.

Watch for one specific defect shape that keeps recurring: **a cache that
populates itself lazily, guarded by a presence check**. The guard asks "is this
initialized?" but what it can actually see is "has anything touched this?" — so
a partially populated cache reads as fully initialized and never converges. If a
reset seems to work in one order but not another, look for that pattern.

---

## Where this came from

The parallel invocation was validated by a 20-run soak: five consecutive green
runs at each of `-n 2`, `3`, `4`, and `6`, plus a serial confirmation run, all
with identical pass counts (`11169 passed, 39 skipped, 1 xfailed`). Wall-clock
scaled from about 15 minutes serial to about 4 minutes at `-n 4`. Every red run
during that soak traced to either shared global state in the suite or one of the
two container-environment prerequisites in section 6 — which is why both are
written down here rather than left as folklore.
