#!/usr/bin/env python3
"""Declarative driver for the OSPREY model-capability e2e matrix (issue #259).

This replaces the bash lane-logic (``matrix_driver.sh`` + the ``case`` statements
in ``run_e2e_for_model.sh``/``run_e2e_alsapg.sh``) with a single declarative
config (``matrix.yaml``) plus a thin orchestrator. Each model row names a
``provider`` and a model ``id``; the launcher resolves credentials, derives the
route (proxy vs direct) from the model's wire ``protocol``, wires the LLM judge,
and spawns one isolated worker process per (model, seed) cell.

    scripts/benchmark/matrix.py                      # run the full matrix
    scripts/benchmark/matrix.py --config m.yaml      # alternate config
    scripts/benchmark/matrix.py --only deepseek-v4   # substring-filter models
    scripts/benchmark/matrix.py --dry-run            # print the plan, run nothing

Design seam: this module owns every *decision* (which provider, which key, proxy
or direct, judge endpoint, budget scale, grid order, resume); the worker
``run_e2e_for_model.sh`` owns *execution* (per-cell ARIEL DB isolation, the
``pytest tests/e2e/`` invocation, and JUnit→JSON summarization). The functions
below are pure and import-safe so they can be unit-tested without spawning a run
(see tests/benchmark/test_matrix.py).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_CONFIG = HERE / "matrix.yaml"
WORKER = HERE / "run_e2e_for_model.sh"


@dataclass(frozen=True)
class Cell:
    """One (model, seed) unit of work."""

    model: str
    provider: str
    seed: int
    protocol: str  # "openai" -> proxy route; "anthropic" -> direct
    budget_scale: float
    # judge: the grader endpoint for this cell (always a reachable, capable
    # model — NEVER the model under test).
    judge_base_url: str
    judge_model: str
    judge_key: str
    # routing
    base_url: str  # the model-under-test provider's base URL
    key: str  # the model-under-test provider's key ("" when keyless/local)
    # The env var each provider exposes its key under — derived from the provider
    # config, NEVER a hardcoded provider name. The built project's provider
    # adapter and the e2e harness (sdk_helpers.spec.auth_secret_env) read the key
    # from *this* var (CBORG_API_KEY, STANFORD_API_KEY, ...). None when keyless.
    sut_key_env: str | None = None
    judge_key_env: str | None = None

    @property
    def route(self) -> str:
        """proxy for OpenAI-protocol models, direct for Anthropic-protocol."""
        return "direct" if self.protocol == "anthropic" else "proxy"

    @property
    def safe(self) -> str:
        """Filesystem-safe model id (mirrors the bash ``${m//\\//__}``)."""
        return self.model.replace("/", "__")

    @property
    def summary_json(self) -> str:
        return f"{self.safe}__seed{self.seed}.json"


@dataclass
class MatrixConfig:
    providers: dict[str, dict[str, Any]]
    models: list[dict[str, Any]]
    defaults: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def resolve_key(provider_conf: dict[str, Any]) -> str:
    """Resolve a provider's API key from env or a key file.

    A provider may declare ``keyless: true`` (local servers like ds4/ollama that
    ignore auth), ``key_env`` (read from the environment), and/or ``key_file``
    (read from a ``~``-expanded path; trailing whitespace stripped). ``key_env``
    wins when both are present and the env var is set. Keyless providers — and
    providers whose key cannot be found — resolve to ``""`` (the worker sends an
    empty bearer, which local servers ignore and the gate-shims tolerate).
    """
    if provider_conf.get("keyless"):
        return ""
    key_env = provider_conf.get("key_env")
    if key_env and os.environ.get(key_env):
        return os.environ[key_env]
    key_file = provider_conf.get("key_file")
    if key_file:
        path = Path(os.path.expanduser(key_file))
        if path.is_file():
            return path.read_text().strip()
    return ""


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def provider_key_env(name: str, pconf: dict[str, Any]) -> str | None:
    """The env var a provider exposes its API key under.

    Prefer the provider's declared ``key_env`` (matrix.yaml); otherwise derive it
    by OSPREY's own convention (``models.provider_registry.PROVIDER_API_KEYS`` /
    ``build.claude_code_resolver``: ``{NAME}_API_KEY`` with ``-`` → ``_``). Keyless
    providers (local servers) return ``None``. This is the single seam that keeps
    the launcher from hardcoding any provider name — add a provider, get the
    right key var for free.
    """
    if pconf.get("keyless"):
        return None
    declared = pconf.get("key_env")
    if declared:
        return declared
    return f"{name.upper().replace('-', '_')}_API_KEY"


def _model_field(model: dict[str, Any], defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    """Per-model value, falling back to ``defaults`` then a hard fallback."""
    if key in model:
        return model[key]
    if key in defaults:
        return defaults[key]
    return fallback


def build_cell(cfg: MatrixConfig, model: dict[str, Any], seed: int) -> Cell:
    """Resolve one model row + seed into a fully-wired :class:`Cell`."""
    provider = model["provider"]
    pconf = cfg.providers.get(provider)
    if pconf is None:
        raise KeyError(f"model '{model.get('id')}' references unknown provider '{provider}'")

    # Effective wire protocol: a per-model override wins over the provider's
    # native protocol. This is REQUIRED for CBORG, which serves claude-* on the
    # Anthropic route (direct) and everything else on the OpenAI route (proxy)
    # under one provider name.
    protocol = model.get("protocol", pconf.get("protocol", "openai"))

    # Judge: each provider declares which grader to use (``judge.via`` names a
    # provider whose base_url + key back the judge; ``judge.model`` is a model id
    # valid at that endpoint). Local/open SUTs point their judge at a reachable
    # remote (e.g. cborg); als-apg SUTs judge via als-apg (same key, consistent).
    judge = pconf.get("judge") or cfg.defaults.get("judge")
    if not judge:
        raise KeyError(
            f"provider '{provider}' has no judge and config has no default judge; "
            "the scenario tests need a reachable LLM grader"
        )
    judge_via = judge.get("via", provider)
    judge_pconf = cfg.providers.get(judge_via)
    if judge_pconf is None:
        raise KeyError(f"judge.via='{judge_via}' is not a known provider")

    return Cell(
        model=model["id"],
        provider=provider,
        seed=seed,
        protocol=protocol,
        budget_scale=float(_model_field(model, cfg.defaults, "budget_scale", 1)),
        judge_base_url=judge_pconf["base_url"],
        judge_model=judge["model"],
        judge_key=resolve_key(judge_pconf),
        base_url=pconf.get("base_url", ""),
        key=resolve_key(pconf),
        sut_key_env=provider_key_env(provider, pconf),
        judge_key_env=provider_key_env(judge_via, judge_pconf),
    )


def expand_grid(
    cfg: MatrixConfig,
    only: str | None = None,
    providers: set[str] | None = None,
) -> list[Cell]:
    """Expand the config into a seed-major list of cells.

    Seed-major (every model at seed 1, then seed 2, ...) so a complete
    cross-model comparison lands after the first pass rather than after all seeds
    of the first model. ``only`` substring-filters the model ids; ``providers``
    restricts to a set of provider names (used to run the subject lane and the
    reference lane at different parallelism).
    """
    models = cfg.models
    if only:
        models = [m for m in models if only in m["id"]]
    if providers:
        models = [m for m in models if m["provider"] in providers]

    # collect the union of seeds across all (each model may set its own count)
    per_model_seeds = {
        m["id"]: list(range(1, int(_model_field(m, cfg.defaults, "seeds", 3)) + 1)) for m in models
    }
    max_seed = max((s for seeds in per_model_seeds.values() for s in seeds), default=0)

    cells: list[Cell] = []
    for seed in range(1, max_seed + 1):
        for m in models:
            if seed in per_model_seeds[m["id"]]:
                cells.append(build_cell(cfg, m, seed))
    return cells


# ---------------------------------------------------------------------------
# Per-cell environment (the old ``case`` logic, now data-driven)
# ---------------------------------------------------------------------------


def cell_env(cell: Cell) -> dict[str, str]:
    """Compute the worker env for one cell.

    This is the single place that used to be the ``case "$MODEL"`` /
    ``MATRIX_PROVIDER`` branching. Everything the worker needs to route the
    model-under-test, wire the judge, and satisfy the collection gates is set
    here; the worker recomputes nothing.
    """
    env: dict[str, str] = {}

    # --- model-under-test routing -----------------------------------------
    env["OSPREY_E2E_FORCE_PROVIDER"] = cell.provider
    env["OSPREY_E2E_FORCE_MODEL"] = cell.model
    if cell.route == "proxy":
        # open / OpenAI-protocol model: the session fixture starts the
        # translation proxy and points it at this upstream.
        env["OSPREY_E2E_PROXY_UPSTREAM"] = cell.base_url
        env["OSPREY_E2E_PROXY_KEY"] = cell.key  # "" for keyless local servers
    else:
        # Direct route: explicitly blank the proxy vars so a stale value in the
        # ambient env (run_cell copies os.environ) can't make conftest spin up a
        # proxy for an Anthropic-direct cell. conftest treats "" as "no proxy".
        env["OSPREY_E2E_PROXY_UPSTREAM"] = ""
        env["OSPREY_E2E_PROXY_KEY"] = ""

    # --- LLM judge (tests/e2e/judge.py reads these three) -----------------
    env["ALS_APG_BASE_URL"] = cell.judge_base_url
    env["OSPREY_E2E_JUDGE_MODEL"] = cell.judge_model
    # ALS_APG_API_KEY = judge key, and ALSO the routing key when the SUT provider
    # is als-apg (single env var, consistent because such cells judge via als-apg).
    env["ALS_APG_API_KEY"] = cell.judge_key or cell.key
    # Collection-gate shims (requires_anthropic / requires_als_apg) — any
    # non-empty value; real routing is overridden above.
    env["ANTHROPIC_API_KEY"] = cell.key or cell.judge_key or "EMPTY"

    # --- per-provider key exposure (data-driven; no hardcoded provider) ----
    # The built project's provider adapter and the e2e harness's
    # spec.auth_secret_env read the SUT/judge key from the *provider's own* env
    # var (CBORG_API_KEY, STANFORD_API_KEY, ...). Expose each under the name the
    # provider config declares (provider_key_env), so adding a provider — Stanford,
    # Argo, a local vLLM — stays a matrix.yaml edit with no launcher change.
    if cell.sut_key_env and cell.key:
        env[cell.sut_key_env] = cell.key
    if cell.judge_key_env and cell.judge_key:
        env[cell.judge_key_env] = cell.judge_key

    # --- budget scale (price normalizer; local models stay at 1) ----------
    env["OSPREY_E2E_BUDGET_SCALE"] = _fmt_scale(cell.budget_scale)

    return env


def _fmt_scale(value: float) -> str:
    """Render a budget scale without a trailing ``.0`` for whole numbers."""
    return str(int(value)) if float(value).is_integer() else str(value)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: Path) -> MatrixConfig:
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    providers = raw.get("providers") or {}
    models = raw.get("models") or []
    defaults = raw.get("defaults") or {}
    # a top-level judge becomes the default judge for providers that omit one
    if "judge" in raw and "judge" not in defaults:
        defaults["judge"] = raw["judge"]
    if not providers:
        raise ValueError(f"{path}: no 'providers' defined")
    if not models:
        raise ValueError(f"{path}: no 'models' defined")
    return MatrixConfig(providers=providers, models=models, defaults=defaults)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


_log_lock = threading.Lock()


def _log(log_path: Path, line: str) -> None:
    # Thread-safe: concurrent cells (and the two lanes matrix_restart.sh launches)
    # append to one matrix.log. Keep the console print and file append together.
    stamp = time.strftime("%H:%M:%S")
    with _log_lock:
        with open(log_path, "a") as fh:
            fh.write(f"{line}\n")
        print(f"[{stamp}] {line}", flush=True)


def run_cell(cell: Cell, results_dir: Path, log_path: Path, extra_env: dict[str, str]) -> int:
    """Spawn the worker for one cell, streaming its output to a per-cell log.

    The ``>> START``/``>> END`` lines MUST keep the legacy shape — a trailing
    ``HH:MM:SS`` on START and ``rc=<n> HH:MM:SS`` on END — because
    matrix_dashboard_live.sh pairs them by stripping exactly those suffixes to
    decide when the run is complete.
    """
    env = os.environ.copy()
    env.update(extra_env)
    env.update(cell_env(cell))
    run_log = results_dir / f"{cell.safe}__seed{cell.seed}.run.log"
    _log(log_path, start_marker(cell, time.strftime("%H:%M:%S")))
    with open(run_log, "w") as out:
        rc = subprocess.call(
            ["bash", str(WORKER), cell.model, str(cell.seed)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    _log(log_path, end_marker(cell, rc, time.strftime("%H:%M:%S")))
    return rc


def start_marker(cell: Cell, ts: str) -> str:
    """``>> START <model> seed<n> <HH:MM:SS>`` — dashboard strips the trailing time."""
    return f">> START {cell.model} seed{cell.seed} {ts}"


def end_marker(cell: Cell, rc: int, ts: str) -> str:
    """``>> END   <model> seed<n> rc=<n> <HH:MM:SS>`` — dashboard strips ``rc=...``."""
    return f">> END   {cell.model} seed{cell.seed} rc={rc} {ts}"


def drive(
    cfg: MatrixConfig,
    *,
    results_dir: Path,
    parallel: int,
    only: str | None,
    providers: set[str] | None,
    extra_env: dict[str, str],
) -> int:
    """Expand the grid and run all not-yet-done cells with bounded parallelism.

    Resumable: cells whose summary json already exists are skipped (mirrors the
    bash driver). Writes the same ``results/matrix.log`` markers the live
    dashboard watches (``>> START/END``, ``MATRIX_EXIT_0``).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "matrix.log"
    cells = expand_grid(cfg, only=only, providers=providers)

    todo, skipped = [], 0
    for cell in cells:
        if (results_dir / cell.summary_json).exists():
            skipped += 1
        else:
            todo.append(cell)

    _log(
        log_path,
        f"=== matrix start models={sorted({c.model for c in cells})} "
        f"cells={len(cells)} todo={len(todo)} skipped={skipped} parallel={parallel} ===",
    )

    rcs: dict[str, int] = {}
    lock = threading.Lock()

    def _work(cell: Cell) -> None:
        rc = run_cell(cell, results_dir, log_path, extra_env)
        with lock:
            rcs[f"{cell.safe}__seed{cell.seed}"] = rc

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        list(pool.map(_work, todo))

    _log(log_path, f"=== matrix complete cells_run={len(todo)} ===")
    _log(log_path, "MATRIX_EXIT_0")
    # non-zero only if a worker itself failed to launch (pytest failures are a
    # per-test signal captured in the json, not a driver error).
    return 0


def plan_lines(cfg: MatrixConfig, only: str | None, providers: set[str] | None = None) -> list[str]:
    """Human-readable plan (for --dry-run); no side effects."""
    lines = []
    for cell in expand_grid(cfg, only=only, providers=providers):
        keymark = "keyless" if not cell.key else "key:set"
        lines.append(
            f"{cell.model:<32} provider={cell.provider:<8} seed={cell.seed} "
            f"route={cell.route:<6} protocol={cell.protocol:<9} "
            f"budget_scale={_fmt_scale(cell.budget_scale)} "
            f"judge={cell.judge_model}@{cell.judge_base_url} sut={keymark}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("MATRIX_PARALLEL", "4")))
    ap.add_argument("--only", default=None, help="substring-filter the model ids")
    ap.add_argument(
        "--provider",
        default=None,
        help="comma-separated provider filter (e.g. 'cborg' or 'als-apg')",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    providers = {p.strip() for p in args.provider.split(",")} if args.provider else None

    if args.dry_run:
        for line in plan_lines(cfg, args.only, providers):
            print(line)
        return 0

    extra_env: dict[str, str] = {}
    if "timeout_s" in cfg.defaults:
        extra_env["E2E_TEST_TIMEOUT"] = str(cfg.defaults["timeout_s"])

    return drive(
        cfg,
        results_dir=args.results_dir,
        parallel=args.parallel,
        only=args.only,
        providers=providers,
        extra_env=extra_env,
    )


if __name__ == "__main__":
    sys.exit(main())
