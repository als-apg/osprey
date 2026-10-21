"""Every bundled-service key the build writes or a template reads is a manifest row.

A bundled service's ``services.<name>`` block is closed against the config-key
manifest's rows (``build_profile_load.reject_unknown_closed_config_keys``), so
a key with no row is a key an operator cannot author. Two kinds of key never
reach the guard's preset-and-framework union: the ones the service injectors
write into ``services.<name>`` during a deploying build, and the ones a compose
template or a Python reader takes that no preset states. These tests pin both
halves, and pin that the guard itself now renders the injected shape, so the
gap cannot reopen unseen.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from osprey.cli.build_profile_load import bundled_service_names, manifest_rows, service_family

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "check_config_keys.py"
SOURCE_ROOT = REPO_ROOT / "src" / "osprey"
SERVICE_TEMPLATES = SOURCE_ROOT / "templates" / "services"

#: Spellings the measured grep finds that are not config keys, each with the
#: reason. Nothing else may be missing a row.
NOT_KEYS = {
    # The compose generator writes it into the lane's block at render time
    # from the control-system posture; a profile spelling it would be
    # overwritten, so it is not an operator key.
    "services.bluesky.writes_enabled",
    # host_ports names it only to say it does not exist (the graph store
    # publishes `port_host` and `http_port_host`).
    "services.graphdb.port",
}


def _load_guard():
    spec = importlib.util.spec_from_file_location("check_config_keys", GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard_module():
    return _load_guard()


@pytest.fixture(scope="module")
def guard(guard_module):
    return guard_module.ConfigKeyGuard(
        REPO_ROOT, guard_module.load_manifest(guard_module.DEFAULT_MANIFEST)
    )


def _is_row(path: str, rows) -> bool:
    data_maps = [
        key for key, spec in rows.items() if isinstance(spec, dict) and spec.get("data-map")
    ]
    return path in rows or any(path.startswith(f"{prefix}.") for prefix in data_maps)


def test_every_key_an_injector_writes_is_a_manifest_row(guard, guard_module) -> None:
    """Each root preset rendered through the injectors: every added path has a row."""
    rows = manifest_rows()
    missing: dict[str, list[str]] = {}
    injected_any = False
    for rel in guard.preset_rels:
        name = guard_module.preset_id(rel)
        added = guard.injected_paths(name)
        injected_any = injected_any or any(path.startswith("services.") for path in added)
        unmapped = sorted(path for path in added if not _is_row(guard.family_path(path), rows))
        if unmapped:
            missing[name] = unmapped

    assert injected_any, (
        "no preset's deploying render injected a service — the probe proves nothing"
    )
    assert not missing, f"injector-written keys with no manifest row: {missing}"


def _template_reads() -> dict[str, set[str]]:
    """``services.<bundled>.<key>`` spellings across src/osprey, by family key.

    Two spellings: the dotted path itself (not preceded by a word character or
    a dot, so ``osprey.services.archive.run`` — a module path — is not one),
    and a compose template's alias of its block
    (``{% set svc = services.archive ... %}`` or ``services[lane_key]``), whose
    ``svc.<key>`` reads are the block's keys.
    """
    names = sorted(bundled_service_names(), key=len, reverse=True)
    alternatives = "|".join(names)
    dotted = re.compile(rf"(?<![\w.])services\.({alternatives})\.([A-Za-z_]\w*)")
    alias = re.compile(r"set (\w+) = services(?:\.(\w+)|\[\w+\])")
    hits: dict[str, set[str]] = defaultdict(set)
    for path in SOURCE_ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name == "config_key_manifest.yml":
            continue
        text = path.read_text(errors="ignore")
        rel = str(path.relative_to(REPO_ROOT))
        for match in dotted.finditer(text):
            hits[f"services.{service_family(match.group(1))}.{match.group(2)}"].add(rel)
        if path.name == "docker-compose.yml.j2" and path.parent.parent == SERVICE_TEMPLATES:
            for match in alias.finditer(text):
                service = match.group(2) or path.parent.name
                for read in re.finditer(rf"\b{match.group(1)}\.(\w+)", text):
                    hits[f"services.{service_family(service)}.{read.group(1)}"].add(rel)
    return hits


def test_every_bundled_service_key_a_template_reads_is_a_manifest_row() -> None:
    rows = manifest_rows()
    reads = _template_reads()
    assert "services.bluesky.port" in reads, "the grep found nothing it must find"

    missing = {
        key: sorted(sites)
        for key, sites in reads.items()
        if key not in NOT_KEYS and not _is_row(key, rows)
    }
    assert not missing, f"bundled-service keys read in src/osprey with no manifest row: {missing}"


def test_the_not_keys_list_is_still_measured() -> None:
    """An exclusion the grep no longer finds is stale and goes."""
    assert NOT_KEYS <= set(_template_reads())


def test_the_guard_renders_the_injected_shape(guard_module) -> None:
    """A row removed from the manifest is ``unmapped-key`` through the injected render."""
    manifest = guard_module.load_manifest(guard_module.DEFAULT_MANIFEST)
    removed = "services.dispatch_worker.worker_count"
    assert removed in manifest["keys"]
    del manifest["keys"][removed]

    doctored = guard_module.ConfigKeyGuard(REPO_ROOT, manifest)
    doctored.check_unmapped_keys()

    failures = [f for f in doctored.result.failures if f.mode == "unmapped-key"]
    assert [f.detail.split(" ", 1)[0] for f in failures] == [removed]
    assert "control-assistant+injectors" in failures[0].detail


def test_a_lane_instance_key_is_read_against_its_familys_rows(guard_module) -> None:
    assert guard_module.ConfigKeyGuard.family_path("services.bluesky_va.port") == (
        "services.bluesky.port"
    )
    assert guard_module.ConfigKeyGuard.family_path("services.live_standin.port") == (
        "services.live_standin.port"
    )
    assert guard_module.ConfigKeyGuard.family_path("web.panels.bluesky.url") == (
        "web.panels.bluesky.url"
    )
