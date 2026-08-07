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

```
53 passed, 1 warning in 10.09s
========================================================================
live Channel Access gate
  passed=53 skipped=0 failed=0 errors=0 pytest_exit=0
  VERDICT: PASS -- 53 live Channel Access test(s) ran, none skipped.
========================================================================
```

Exit status is the gate's, so this is usable directly as a check. First run
builds the image (a few minutes on an arm64 Mac, where linux/amd64 is
emulated); later runs reuse it and take about ten seconds.

### Composed CA + PVA mode

`tests/va/test_facility_seam.py` has a **served-boot** branch that needs more
than Channel Access: `serving/runner.py` imports `lume_pva_apg` and `p4p`
alongside `pcaspy`. `lume-pva-apg` is not published yet, so point the runner at
a checkout of it:

```bash
OSPREY_LUME_PVA_PATH=/path/to/lume-pva scripts/va/live_ca/run_live_ca.sh
```

That builds a thin PVA layer on the base image (adding `p4p`), bind-mounts the
fork at `/fork`, and runs the seam suite alongside the CA suites — **57 passed,
0 skipped**. Leave the variable unset and you get CA-only, with a printed note
saying the served branch was not exercised.

The path is opt-in rather than probed. A guessed sibling directory that missed
would fall back to CA-only while looking like it had run everything — the same
class of quiet false green this directory exists to prevent.

Once task 6.2 publishes the fork and 6.3 pins it, the mount, the `p4p` install
and `Containerfile.pva` all collapse back into the base image.

## What makes it trustworthy

**It installs what CI installs.** The image runs
`uv sync --frozen --extra dev --extra virtual-accelerator` against the repo's
own `pyproject.toml` and `uv.lock` — the same command CI's unit-test job runs,
on the same platform CI runs it. There is no hand-maintained package list here
to drift out of step with the extras.

**A skip is a failure.** `gate.py` runs the suites through `pytest.main` and
inspects the terminal reporter's own outcome counts, then fails unless pytest
exited 0, **nothing skipped**, and something passed. It is not a text match on
the summary line, so a formatting or verbosity change cannot defeat it.

Verified against a negative control: with `pcaspy` made unimportable inside the
container, pytest reports `9 passed, 44 skipped` and exits **0**, and the gate
turns that into exit **1**. That vacuous green is the exact failure this
directory exists to prevent.

**In `--pva` mode, a skip check is not enough.** `test_facility_seam.py`'s boot
test passes on either of two outcomes — the boot served, or it stopped on a
missing server extension — and both are legitimate, so nothing skips either
way. Run the seam suite in the PVA image with the fork *absent* and it reports
`4 passed, 0 skipped`, exit 0, having never touched the served path. So `--pva`
additionally requires `pcaspy`, `p4p` and `lume_pva_apg` to import *before*
pytest starts. That makes the fallback branch unreachable, which is what makes
the green mean "the served path ran". Without the fork mounted, the gate exits
1 rather than running.

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
