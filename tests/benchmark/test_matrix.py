"""Unit tests for the declarative matrix launcher (scripts/benchmark/matrix.py).

Pure-logic only — no pytest-e2e, no network, no subprocess. Exercises the
decision layer that replaced the bash ``case``/lane logic: credential
resolution, grid expansion (seed-major + resume), route derivation, and the
per-cell environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import scripts/benchmark/matrix.py as a module (scripts/ is not a package).
# Register in sys.modules before exec so @dataclass can resolve cls.__module__.
_MATRIX_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "matrix.py"
_spec = importlib.util.spec_from_file_location("benchmark_matrix", _MATRIX_PATH)
matrix = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_matrix"] = matrix
_spec.loader.exec_module(matrix)


def _cfg() -> matrix.MatrixConfig:
    """A config mirroring the shipped matrix.yaml shape (cborg/als-apg/ds4)."""
    return matrix.MatrixConfig(
        providers={
            "cborg": {
                "base_url": "https://api.cborg.lbl.gov/v1",
                "key_env": "CBORG_API_KEY",
                "protocol": "openai",
                "judge": {"via": "cborg", "model": "google/claude-haiku-4-5"},
            },
            "als-apg": {
                "base_url": "https://llm.gianlucamartino.com",
                "key_env": "ALS_APG_API_KEY",
                "protocol": "anthropic",
                "judge": {"via": "als-apg", "model": "claude-haiku-4-5-20251001"},
            },
            "ds4": {
                "base_url": "http://127.0.0.1:8000/v1",
                "keyless": True,
                "protocol": "openai",
                "judge": {"via": "cborg", "model": "google/claude-haiku-4-5"},
            },
        },
        models=[
            {"id": "gpt-oss-20b", "provider": "cborg"},
            {"id": "google/qwen-3", "provider": "cborg"},
            {"id": "claude-sonnet-4-6", "provider": "als-apg", "seeds": 1, "budget_scale": 3},
            {"id": "deepseek-v4-flash", "provider": "ds4"},
        ],
        defaults={"seeds": 3, "budget_scale": 1, "timeout_s": 1800},
    )


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("CBORG_API_KEY", "cb-key")
    monkeypatch.setenv("ALS_APG_API_KEY", "apg-key")


# --- credential resolution --------------------------------------------------


def test_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv("CBORG_API_KEY", "cb-key")
    assert matrix.resolve_key({"key_env": "CBORG_API_KEY"}) == "cb-key"


def test_resolve_key_keyless_is_empty():
    assert matrix.resolve_key({"keyless": True, "key_env": "CBORG_API_KEY"}) == ""


def test_resolve_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MYKEY", raising=False)
    kf = tmp_path / "k"
    kf.write_text("file-secret\n")
    assert matrix.resolve_key({"key_env": "MYKEY", "key_file": str(kf)}) == "file-secret"


def test_resolve_key_env_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MYKEY", "env-secret")
    kf = tmp_path / "k"
    kf.write_text("file-secret\n")
    assert matrix.resolve_key({"key_env": "MYKEY", "key_file": str(kf)}) == "env-secret"


def test_resolve_key_missing_is_empty(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert matrix.resolve_key({"key_env": "NOPE"}) == ""


# --- route derivation -------------------------------------------------------


def test_route_openai_is_proxy(keys):
    cell = matrix.build_cell(_cfg(), {"id": "gpt-oss-20b", "provider": "cborg"}, 1)
    assert cell.route == "proxy"


def test_route_anthropic_is_direct(keys):
    cell = matrix.build_cell(_cfg(), {"id": "claude-sonnet-4-6", "provider": "als-apg"}, 1)
    assert cell.route == "direct"


def test_per_model_protocol_override_beats_provider(keys):
    # CBORG serves claude-* on the Anthropic route under the OpenAI-native
    # cborg provider — the per-model override must win.
    cell = matrix.build_cell(
        _cfg(), {"id": "claude-x", "provider": "cborg", "protocol": "anthropic"}, 1
    )
    assert cell.protocol == "anthropic" and cell.route == "direct"


def test_safe_name_slashes(keys):
    cell = matrix.build_cell(_cfg(), {"id": "google/qwen-3", "provider": "cborg"}, 2)
    assert cell.safe == "google__qwen-3"
    assert cell.summary_json == "google__qwen-3__seed2.json"


# --- grid expansion: seed-major, per-model seeds, resume, filter ------------


def test_expand_grid_seed_major_and_per_model_seeds(keys):
    cells = matrix.expand_grid(_cfg())
    # cborg(3 seeds)x3 models? -> 3 cborg models? no: 2 cborg + 1 als-apg(1) + 1 ds4(3)
    # seed1: all 4 ; seed2: cborg+ds4 (3) ; seed3: cborg+ds4 (3) => 10 cells
    assert len(cells) == 10
    # seed-major: every cell at seed 1 precedes any cell at seed 2
    seeds_in_order = [c.seed for c in cells]
    assert seeds_in_order == sorted(seeds_in_order)
    # als-apg ref (seeds:1) appears only at seed 1
    alsapg = [c for c in cells if c.provider == "als-apg"]
    assert [c.seed for c in alsapg] == [1]


def test_only_filter(keys):
    cells = matrix.expand_grid(_cfg(), only="deepseek")
    assert {c.model for c in cells} == {"deepseek-v4-flash"}
    assert [c.seed for c in cells] == [1, 2, 3]


def test_provider_filter_isolates_lanes(keys):
    # the subject lane and reference lane run at different parallelism via
    # matrix_restart.sh's --provider filter.
    cborg = matrix.expand_grid(_cfg(), providers={"cborg"})
    assert {c.provider for c in cborg} == {"cborg"}
    refs = matrix.expand_grid(_cfg(), providers={"als-apg"})
    assert {c.model for c in refs} == {"claude-sonnet-4-6"}
    assert [c.seed for c in refs] == [1]


def test_resume_skips_existing_summary(tmp_path, keys):
    cfg = _cfg()
    results = tmp_path / "results"
    results.mkdir()
    # mark one cell done
    done = matrix.build_cell(cfg, {"id": "gpt-oss-20b", "provider": "cborg"}, 1)
    (results / done.summary_json).write_text("{}")
    todo = [c for c in matrix.expand_grid(cfg) if not (results / c.summary_json).exists()]
    assert done.summary_json not in {c.summary_json for c in todo}
    assert len(todo) == 9


def test_matrix_log_markers_pair_like_dashboard(keys):
    # Regression guard: matrix_dashboard_live.sh pairs START/END by stripping a
    # trailing HH:MM:SS off START and "rc=..." off END. A START line whose tail
    # is NOT a time (e.g. "route=proxy") never strips, so cells look perpetually
    # "running" and the run never finalizes. Replicate the dashboard's reduction.
    import re

    cell = matrix.build_cell(_cfg(), {"id": "google/qwen-3", "provider": "cborg"}, 2)
    start = matrix.start_marker(cell, "14:32:01")
    end = matrix.end_marker(cell, 0, "14:35:10")

    # dashboard: s/.*START //;s/ [0-9:]*$//
    start_key = re.sub(r" [0-9:]*$", "", start.split("START ", 1)[1])
    # dashboard: s/.*END   //;s/ rc=.*//
    end_key = re.sub(r" rc=.*", "", end.split("END   ", 1)[1])

    assert start_key == end_key == "google/qwen-3 seed2"


# --- the per-cell environment (the replaced ``case`` logic) -----------------


def test_cell_env_cborg_open_model(keys):
    cell = matrix.build_cell(_cfg(), {"id": "gpt-oss-20b", "provider": "cborg"}, 1)
    env = matrix.cell_env(cell)
    assert env["OSPREY_E2E_FORCE_PROVIDER"] == "cborg"
    assert env["OSPREY_E2E_FORCE_MODEL"] == "gpt-oss-20b"
    assert env["OSPREY_E2E_PROXY_UPSTREAM"] == "https://api.cborg.lbl.gov/v1"
    assert env["OSPREY_E2E_PROXY_KEY"] == "cb-key"
    assert env["ALS_APG_BASE_URL"] == "https://api.cborg.lbl.gov/v1"
    assert env["OSPREY_E2E_JUDGE_MODEL"] == "google/claude-haiku-4-5"
    assert env["ALS_APG_API_KEY"] == "cb-key"
    assert env["CBORG_API_KEY"] == "cb-key"
    assert env["OSPREY_E2E_BUDGET_SCALE"] == "1"


def test_cell_env_alsapg_reference_is_direct(keys):
    cell = matrix.build_cell(
        _cfg(), {"id": "claude-sonnet-4-6", "provider": "als-apg", "budget_scale": 3}, 1
    )
    env = matrix.cell_env(cell)
    # direct route -> proxy vars explicitly blanked (conftest treats "" as "no
    # proxy"); blanking overrides any stale ambient value run_cell copies down.
    assert env["OSPREY_E2E_PROXY_UPSTREAM"] == ""
    assert env["OSPREY_E2E_PROXY_KEY"] == ""
    assert env["ALS_APG_BASE_URL"] == "https://llm.gianlucamartino.com"
    assert env["OSPREY_E2E_JUDGE_MODEL"] == "claude-haiku-4-5-20251001"
    assert env["ALS_APG_API_KEY"] == "apg-key"  # judge key == als-apg routing key
    assert "CBORG_API_KEY" not in env
    assert env["OSPREY_E2E_BUDGET_SCALE"] == "3"


def test_cell_env_ds4_local_keyless_judge_stays_remote(keys):
    cell = matrix.build_cell(_cfg(), {"id": "deepseek-v4-flash", "provider": "ds4"}, 1)
    env = matrix.cell_env(cell)
    assert env["OSPREY_E2E_FORCE_PROVIDER"] == "ds4"
    # local OpenAI server -> proxy route at the loopback upstream, keyless
    assert env["OSPREY_E2E_PROXY_UPSTREAM"] == "http://127.0.0.1:8000/v1"
    assert env["OSPREY_E2E_PROXY_KEY"] == ""
    # judge is NEVER the model under test: stays on a reachable remote (cborg)
    assert env["ALS_APG_BASE_URL"] == "https://api.cborg.lbl.gov/v1"
    assert env["OSPREY_E2E_JUDGE_MODEL"] == "google/claude-haiku-4-5"
    assert env["ALS_APG_API_KEY"] == "cb-key"
    assert env["CBORG_API_KEY"] == "cb-key"
    assert env["OSPREY_E2E_BUDGET_SCALE"] == "1"  # local inference is free


def test_cell_env_non_cborg_sut_exposes_its_own_key_var(monkeypatch):
    """A non-cborg open-model provider (e.g. Stanford AI Playground) must get its
    key exposed under its OWN declared var — not CBORG_API_KEY. This is the
    regression guard for the old `if cell.provider == "cborg"` hardcode: adding a
    provider is a matrix.yaml edit, with no launcher change.
    """
    monkeypatch.setenv("STANFORD_API_KEY", "stan-key")
    monkeypatch.setenv("CBORG_API_KEY", "cb-key")
    cfg = matrix.MatrixConfig(
        providers={
            "stanford": {
                "base_url": "https://aiplayground.stanford.edu/v1",
                "key_env": "STANFORD_API_KEY",
                "protocol": "openai",  # OpenAI-compatible -> proxy route
                # judge stays on a reachable remote, never the model under test
                "judge": {"via": "cborg", "model": "google/claude-haiku-4-5"},
            },
            "cborg": {
                "base_url": "https://api.cborg.lbl.gov/v1",
                "key_env": "CBORG_API_KEY",
                "protocol": "openai",
            },
        },
        models=[{"id": "llama-3-70b", "provider": "stanford"}],
        defaults={},
    )
    cell = matrix.build_cell(cfg, cfg.models[0], 1)
    env = matrix.cell_env(cell)
    # SUT key under STANFORD_API_KEY (what spec.auth_secret_env reads), and the
    # proxy carries it too — all without naming "stanford" anywhere in matrix.py.
    assert env["STANFORD_API_KEY"] == "stan-key"
    assert env["OSPREY_E2E_PROXY_KEY"] == "stan-key"
    # judge runs via cborg, so CBORG_API_KEY is the *judge* key (not the SUT's)
    assert env["CBORG_API_KEY"] == "cb-key"
    assert cell.sut_key_env == "STANFORD_API_KEY"


def test_provider_key_env_derives_by_convention_when_undeclared():
    # No key_env declared -> OSPREY's {NAME}_API_KEY convention (matches
    # provider_registry.PROVIDER_API_KEYS / claude_code_resolver).
    assert matrix.provider_key_env("stanford", {}) == "STANFORD_API_KEY"
    assert matrix.provider_key_env("als-apg", {}) == "ALS_APG_API_KEY"
    # keyless local servers expose nothing
    assert matrix.provider_key_env("ds4", {"keyless": True}) is None
    # declared key_env wins over the convention
    assert matrix.provider_key_env("x", {"key_env": "MY_KEY"}) == "MY_KEY"


# --- errors & misc ----------------------------------------------------------


def test_unknown_provider_raises(keys):
    with pytest.raises(KeyError):
        matrix.build_cell(_cfg(), {"id": "x", "provider": "nope"}, 1)


def test_fmt_scale_drops_trailing_zero():
    assert matrix._fmt_scale(1) == "1"
    assert matrix._fmt_scale(6.0) == "6"
    assert matrix._fmt_scale(1.5) == "1.5"


def test_load_config_round_trip(tmp_path):
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "cborg": {
                        "base_url": "https://api.cborg.lbl.gov/v1",
                        "protocol": "openai",
                        "judge": {"via": "cborg", "model": "g/h"},
                    }
                },
                "models": [{"id": "gpt-oss-20b", "provider": "cborg"}],
                "defaults": {"seeds": 2},
            }
        )
    )
    cfg = matrix.load_config(p)
    assert cfg.defaults["seeds"] == 2
    assert [m["id"] for m in cfg.models] == ["gpt-oss-20b"]


def test_load_config_top_level_judge_becomes_default(tmp_path):
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "providers": {"ds4": {"base_url": "http://x/v1", "keyless": True}},
                "judge": {"via": "ds4", "model": "g/h"},
                "models": [{"id": "deepseek", "provider": "ds4"}],
            }
        )
    )
    cfg = matrix.load_config(p)
    cell = matrix.build_cell(cfg, cfg.models[0], 1)
    assert cell.judge_model == "g/h"


# --- orchestration (drive): resume + matrix.log markers ---------------------


def test_drive_writes_markers_and_resumes(tmp_path, keys, monkeypatch):
    """drive() runs each cell once, emits the dashboard markers, and on a second
    pass skips every already-summarized cell. The per-cell worker is stubbed so
    no real pytest-e2e is spawned."""
    cfg = _cfg()
    results = tmp_path / "results"

    calls: list[str] = []

    def fake_run_cell(cell, results_dir, log_path, extra_env):
        calls.append(cell.summary_json)
        # mimic the worker: drop a summary json so resume can see it
        (results_dir / cell.summary_json).write_text("{}")
        matrix._log(log_path, matrix.start_marker(cell, "00:00:00"))
        matrix._log(log_path, matrix.end_marker(cell, 0, "00:00:01"))
        return 0

    monkeypatch.setattr(matrix, "run_cell", fake_run_cell)

    rc = matrix.drive(
        cfg, results_dir=results, parallel=2, only=None, providers={"als-apg"}, extra_env={}
    )
    assert rc == 0
    # als-apg lane = 1 cell (claude-sonnet, seeds:1)
    assert calls == ["claude-sonnet-4-6__seed1.json"]
    log = (results / "matrix.log").read_text()
    assert ">> START claude-sonnet-4-6 seed1" in log
    assert ">> END   claude-sonnet-4-6 seed1 rc=0" in log
    assert "MATRIX_EXIT_0" in log

    # second pass: everything already summarized -> no new worker calls
    calls.clear()
    matrix.drive(
        cfg, results_dir=results, parallel=2, only=None, providers={"als-apg"}, extra_env={}
    )
    assert calls == []
