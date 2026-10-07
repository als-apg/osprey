# Live Channel Access test venue

The gate runs two suites. `tests/va/test_apply_fault.py` boots the virtual
accelerator's entrypoint over a stub view in a spawned process and asserts its
scenario faults from the far side of a real Channel Access wire, with a real
`pyepics` client in the pytest process. `tests/va/test_lume_pva_seam.py`
serves composites through the model runner on Channel Access and PVAccess and
asserts what a `pyepics` and a `p4p` client read.

That needs `pcaspy`, and `pcaspy` publishes manylinux **x86_64** wheels only —
no aarch64 wheel at any interpreter, and the macOS arm64 wheels it does publish
are unloadable as shipped. So on a developer's Mac the live classes skip.

They skip honestly, via `pytest.importorskip` with a reason. But a skipped live
suite proves nothing, and `pytest` exits 0 either way. This directory is the
venue where they are not allowed to skip.

## Run it

```bash
scripts/va/live_ca/run_live_ca.sh
```

Expected tail:

The shape of the tail, with each child's own pytest progress bar and warnings
summary elided between the header and its summary line:

```
--- tests/va/test_apply_fault.py ---
    [... test_apply_fault's own pytest output ...]
15 passed, 1 warning in 54.04s
  tests/va/test_apply_fault.py: passed=15 skipped=0 failed=0 errors=0 exit=0
--- tests/va/test_lume_pva_seam.py ---
    [... test_lume_pva_seam's own pytest output ...]
14 passed, 3 warnings in 207.69s (0:03:27)
  tests/va/test_lume_pva_seam.py: passed=14 skipped=0 failed=0 errors=0 exit=0

========================================================================
live Channel Access gate (--pva)
  passed=29 skipped=0 failed=0 errors=0 pytest_exit=0
  VERDICT: PASS -- 29 live Channel Access test(s) ran, none skipped.
========================================================================
```

Each child's full output really is printed — nothing is suppressed at runtime;
the elisions above are only to keep this block readable.

Exit status is the gate's, so this is usable directly as a check. First run
builds the image (a few minutes on an arm64 Mac, where linux/amd64 is
emulated); later runs reuse it, and the suites' own time is most of the run.

That one command covers both transports. `serving/runner.py` imports
`lume_pva_apg` and `p4p` alongside `pcaspy`, and all three arrive with the
`virtual-accelerator` extra, so both suites run here with nothing extra to
install, mount or set. A composite loads its physics engines by entry-point
name; the image carries the root project's entry points as metadata generated
from the staged `pyproject.toml`, so the engines resolve without the project
being installed.

## What makes it trustworthy

**It installs what CI installs.** The image runs
`uv sync --frozen --extra dev --extra virtual-accelerator` against the repo's
own `pyproject.toml` and `uv.lock` — the same command CI's unit-test job runs,
on the same platform CI runs it. There is no hand-maintained package list here
to drift out of step with the extras.

**A skip is a failure.** `gate.py` inspects the terminal reporter's own outcome
counts, then fails unless pytest exited 0, **nothing skipped**, and something
passed — applied to every module and to the total, so one module contributing
nothing cannot ride to green on the others' passes. It is not a text match on
the summary line, so a formatting or verbosity change cannot defeat it.

**Each module gets its own process.** The live modules pick a Channel Access
port at import with `os.environ.setdefault("EPICS_CA_SERVER_PORT", ...)`, so in
a single combined pytest run the first module imported wins the port and the
second stands its `pcaspy` server up on a port the process-wide libca client
has already latched onto the first. That was observed as every test in the
second module erroring at fixture setup with "the Channel Access server never
became reachable" — intermittently, twice in ~27 combined runs, while the same
module alone in a fresh process went 20 for 20. One module per process is what
makes that `setdefault` mean what it looks like it means. Counts still come
from pytest's own reporter: each child is `gate.py` re-entered with
`--run-module`, printing one machine-readable line the parent sums, and a child
that prints nothing is counted as an error rather than as zeroes.

With `pcaspy` made unimportable inside the container, pytest skips the live
classes and exits **0**, and the gate turns that into exit **1**. That vacuous
green is the exact failure this directory exists to prevent.

`run_live_ca.sh` always passes `--pva`, and the import precondition below
rejects a missing `pcaspy` *before* pytest starts — so a control run through it
exits 1 from the precondition, not from the skip check. To exercise the skip
check itself, run `gate.py` directly WITHOUT `--pva`, which is the mode the
precondition does not apply to:

```bash
docker run --rm --platform linux/amd64 -v "$PWD:/work:ro" <image> \
    python -u scripts/va/live_ca/gate.py
```

with `pcaspy` made unimportable. `pcaspy` is the only module
`tests/va/test_apply_fault.py` guards with `importorskip`, so it is also the
only one whose absence produces a skip there rather than an error.

**The seam suite is required whole.** `tests/va/test_lume_pva_seam.py`
importorskips `pcaspy`, `p4p` and `lume_pva_apg` at module level, so a host
lacking any of them skips the module whole. The gate runs in `--pva` mode,
which requires all three to import *before* pytest starts, so an image that
lacked one exits 1 naming it.

**The tag is content-addressed.** The image is tagged with a digest of
`pyproject.toml`, `uv.lock` and the `Containerfile`, so bumping the lume-pva-apg
pin or editing a build step produces a new tag rather than silently reusing a
stale image built under the same name.

## It claims no host port

The servers and their clients share the container's own network namespace. Nothing is published — there is no `-p` on the `docker run`,
by design, not by omission. This is why the venue never collides with a virtual
accelerator already serving on 5064. The suites also pick their own ephemeral
loopback port at import time, so two runs at once do not interfere either.

Keep that property when editing `run_live_ca.sh`.

## Where else the live suites run

CI's `ubuntu-latest` unit-test lanes are x86_64 and install the
`virtual-accelerator` extra, so `pcaspy` is present there. The unit lane's
sweep ignores both suites by name, and one cell runs `gate.py --pva` in a step
of its own, one process per module, under the same skip and pass checks. The
`macos-latest` lanes are arm64; the marker on `lume-pva-apg` excludes them, the same
way it excludes a developer's Mac. This directory's image is the local venue
for proving the contract before pushing.
