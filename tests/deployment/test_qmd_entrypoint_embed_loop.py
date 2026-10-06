"""The qmd entrypoint's embedding pass, run for real against a stub ``qmd``.

``qmd embed`` runs inside a session with a fixed time limit. When the limit
hits it skips the remaining batches, prints ``Done``, and exits 0, so a single
call can leave most of a large corpus without vectors while reporting success.
The entrypoint therefore repeats the call until ``qmd status`` reports nothing
pending, or until a round embeds nothing (chunks that fail permanently would
otherwise loop forever), and says how many documents are left.

The test sources the shipped entrypoint (minus its final ``main`` call) under
``sh`` and puts a stub ``qmd`` on PATH whose ``embed`` clears a fixed number of
pending documents per call, the way a session that runs out of time does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the image has")

#: A stub qmd. ``embed`` embeds at most $STUB_PER_ROUND pending documents, never
#: going below $STUB_FLOOR (documents whose chunks fail on every retry), and logs
#: each call; ``status`` prints the lines the real CLI prints when piped.
_STUB = """#!/bin/sh
state="$STUB_STATE"
pending=$(cat "$state")
case "$1" in
    embed)
        echo embed >> "$STUB_CALLS"
        left=$((pending - STUB_PER_ROUND))
        [ "$left" -ge "$STUB_FLOOR" ] || left=$STUB_FLOOR
        echo "$left" > "$state"
        echo "Session expired - skipping remaining document batches"
        echo "Done! Embedded some chunks"
        ;;
    status)
        echo "Documents"
        echo "  Total:    100 files indexed"
        echo "  Vectors:  42 embedded"
        if [ "$pending" -gt 0 ]; then
            echo "  Pending:  $pending need embedding (run 'qmd embed')"
        fi
        ;;
esac
exit 0
"""


def _run_embed_pass(tmp_path: Path, *, pending: int, per_round: int, floor: int = 0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "qmd"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    state = tmp_path / "pending"
    state.write_text(f"{pending}\n")
    calls = tmp_path / "calls"
    calls.write_text("")

    lines = ENTRYPOINT.read_text().splitlines()
    assert lines[-1] == 'main "$@"', "the entrypoint no longer ends with its main call"
    library = tmp_path / "entrypoint-lib.sh"
    library.write_text("\n".join(lines[:-1]) + "\n")

    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "OSPREY_QMD_PORT": "1",
        "STUB_STATE": str(state),
        "STUB_CALLS": str(calls),
        "STUB_PER_ROUND": str(per_round),
        "STUB_FLOOR": str(floor),
    }
    script = f'. "{library}"; embed_until_done; echo "PENDING=$EMBED_PENDING"'
    result = subprocess.run(
        ["sh", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    return result, len(calls.read_text().split())


def test_embedding_repeats_until_nothing_is_pending(tmp_path):
    # 9,984 documents with ~1,258 embedded per session in the field; scaled down.
    result, rounds = _run_embed_pass(tmp_path, pending=10, per_round=3)

    assert result.returncode == 0, result.stdout + result.stderr
    assert rounds == 4
    assert "PENDING=0" in result.stdout


def test_a_round_that_embeds_nothing_ends_the_loop_and_reports_what_is_left(tmp_path):
    result, rounds = _run_embed_pass(tmp_path, pending=10, per_round=3, floor=2)

    assert result.returncode == 0, result.stdout + result.stderr
    # 10 -> 7 -> 4 -> 2 -> 2: the fourth round made no progress.
    assert rounds == 4
    assert "PENDING=2" in result.stdout
    assert "2 document(s) still have no vectors" in result.stdout


def test_embedding_runs_once_even_when_status_reports_nothing_pending(tmp_path):
    # The status parser fails towards zero; a zero read before any embedding must
    # not skip the embedding itself.
    result, rounds = _run_embed_pass(tmp_path, pending=0, per_round=3)

    assert result.returncode == 0, result.stdout + result.stderr
    assert rounds == 1
    assert "PENDING=0" in result.stdout
