# Live Channel Access test venue

`tests/va/test_record_factory.py` and `tests/va/test_apply_fault.py` assert the
virtual accelerator's serving contract from the far side of a real Channel
Access wire. Everything runs in one process: a real `pcaspy` CA server hosting
the real serving database, the real write path deciding what each write means,
and a real `pyepics` client driving it from the pytest thread.

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

The shape of the tail, with each child's own pytest progress bar, warnings
summary and per-run summary line elided between the header and the count line:

```
--- tests/va/test_record_factory.py ---
    [... test_record_factory's own pytest output: 35 passed in 9.36s ...]
  tests/va/test_record_factory.py: passed=35 skipped=0 failed=0 errors=0 exit=0
--- tests/va/test_apply_fault.py ---
    [... test_apply_fault's own pytest output: 18 passed in 5.02s ...]
  tests/va/test_apply_fault.py: passed=18 skipped=0 failed=0 errors=0 exit=0
--- tests/va/test_facility_seam.py ---
    [... test_facility_seam's own pytest output: 4 passed in 9.48s ...]
  tests/va/test_facility_seam.py: passed=4 skipped=0 failed=0 errors=0 exit=0

========================================================================
live Channel Access gate (--pva)
  passed=57 skipped=0 failed=0 errors=0 pytest_exit=0
  VERDICT: PASS -- 57 live Channel Access test(s) ran, none skipped.
========================================================================
```

Each child's full output really is printed — nothing is suppressed at runtime;
the elisions above are only to keep this block readable.

Exit status is the gate's, so this is usable directly as a check. First run
builds the image (a few minutes on an arm64 Mac, where linux/amd64 is
emulated); later runs reuse it and take about ten seconds.

That one command covers both transports. `tests/va/test_facility_seam.py` has a
**served-boot** branch that needs more than Channel Access — `serving/runner.py`
imports `lume_pva_apg` and `p4p` alongside `pcaspy` — and all three now arrive
with the `virtual-accelerator` extra, so the seam suite runs here alongside the
CA suites with nothing extra to install, mount or set.

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

Verified against a negative control: with `pcaspy` made unimportable inside the
container, pytest reports `9 passed, 44 skipped` and exits **0**, and the gate
turns that into exit **1**. That vacuous green is the exact failure this
directory exists to prevent.

Reproducing that control today takes one extra step, because `run_live_ca.sh`
always passes `--pva` and the import precondition below now rejects a missing
`pcaspy` *before* pytest starts — so you get exit 1 from the precondition, not
from the skip check, and the skip check itself goes unexercised. To exercise
it, run `gate.py` directly WITHOUT `--pva`, which is the mode the precondition
does not apply to:

```bash
docker run --rm --platform linux/amd64 -v "$PWD:/work:ro" <image> \
    python -u scripts/va/live_ca/gate.py
```

with `pcaspy` made unimportable. `pcaspy` is the only module the live suites
guard with `importorskip`, so it is also the only one whose absence produces a
skip rather than an error — breaking anything else gives you a red run for a
different reason and tells you nothing about the skip check.

**For the seam suite, a skip check is not enough.** `test_facility_seam.py`'s
boot test passes on either of two outcomes — the boot served, or it stopped on
a missing server extension — and both are legitimate, so nothing skips either
way. Run it with the serving stack *absent* and it reports `4 passed, 0
skipped`, exit 0, having never touched the served path. So the gate runs in
`--pva` mode, which additionally requires `pcaspy`, `p4p` and `lume_pva_apg` to
import *before* pytest starts. That makes the fallback branch unreachable,
which is what makes the green mean "the served path ran"; an image that somehow
lacked one of the three exits 1 rather than certifying the fallback.

**The tag is content-addressed.** The image is tagged with a digest of
`pyproject.toml`, `uv.lock` and the `Containerfile`, so bumping the pcaspy
floor or editing a build step produces a new tag rather than silently reusing a
stale image built under the same name.

## It claims no host port

The CA server and its client share one process inside the container's own
network namespace. Nothing is published — there is no `-p` on the `docker run`,
by design, not by omission. This is why the venue never collides with a virtual
accelerator already serving on 5064. The suites also pick their own ephemeral
loopback port at import time, so two runs at once do not interfere either.

Keep that property when editing `run_live_ca.sh`.

## Where else the live suites run

CI's `ubuntu-latest` unit-test lanes are x86_64 and install the
`virtual-accelerator` extra, so `pcaspy` is present there and the live suites
run as part of the ordinary `pytest tests/` job. The `macos-latest` lanes are
arm64; the marker on `pcaspy` excludes them and the suites skip there, the same
way they skip on a developer's Mac.

The gate in this directory does not run on CI. It is the local instrument for
proving the contract before pushing, and the reference for what a genuine
Channel Access green looks like.
