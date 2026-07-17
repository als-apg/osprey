"""Tests for the web-terminal ``seed`` step (osprey.deployment.web_terminals.seeding).

The container runtime is entirely mocked: ``seeding.subprocess.run`` is patched to
record every emitted argv (and its stdin ``input``) instead of touching a real
docker/podman daemon, and ``seeding.get_runtime_command``/``seeding.runtime_env``
are pinned to fixed values. The ``docker/web-terminal-context/`` overlay tree is
built under ``tmp_path``. No real container is ever created, execed into, or
removed by these tests.
"""

from __future__ import annotations

import io
import subprocess
import tarfile

import pytest
import yaml

from osprey.deployment.web_terminals import seeding

_FACILITY_PREFIX = "dls"


def _config(users, *, facility_prefix=_FACILITY_PREFIX, registry=None, web_terminals_extra=None):
    """Minimal-but-complete facility config exercising every field seed_user_containers reads.

    ``registry`` and ``web_terminals_extra`` (merged into ``modules.web_terminals``, e.g.
    ``personas``/``default_persona``/``image_source``) let persona-resolution tests build on
    top of this without duplicating the whole config shape.
    """
    web_terminals: dict = {
        "enabled": True,
        "users": users,
    }
    if web_terminals_extra:
        web_terminals.update(web_terminals_extra)
    config = {
        "project_name": "demo-project",
        "facility": {"name": "Demo Light Source", "prefix": facility_prefix, "timezone": "UTC"},
        "modules": {"web_terminals": web_terminals},
    }
    if registry is not None:
        config["registry"] = registry
    return config


def _write_config(tmp_path, config):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class _ReadySet(set):
    """A plain ``set`` of "ready" container names, plus a ``.failing`` sibling set.

    Subclassing (rather than returning a 4-tuple) keeps every existing
    ``ready.add(container)`` call site working unchanged for tests that only
    care about the not-ready/ready distinction; tests exercising the
    systemic-failure path additionally reach into ``ready.failing``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.failing: set[str] = set()
        self.owner: str = "1000:1000"


_FAKE_STDERR = b"boom: chown: unknown user dispatch"


@pytest.fixture
def fake_runtime(monkeypatch):
    """Patch subprocess.run + get_runtime_command/runtime_env; return recorded calls.

    ``ready`` is a mutable ``_ReadySet`` of container names ``inspect`` should
    report as existing; tests populate it before calling
    seed_user_containers/seed_web_terminals to control which users' containers
    are "ready". ``ready.failing`` additionally marks containers whose exec
    calls should fail (``check=True`` raises ``CalledProcessError`` with
    ``_FAKE_STDERR``, mimicking a real non-zero exec exit). Returns
    ``(calls, inputs, ready)``: ``calls`` records every emitted argv in order,
    ``inputs`` records the matching ``input=`` payload (``None`` for the
    ``inspect`` calls, which have none).
    """
    calls: list[list[str]] = []
    inputs: list[bytes | None] = []
    ready = _ReadySet()

    def _fake_run(argv, capture_output=True, text=False, env=None, check=False, input=None):
        calls.append(list(argv))
        inputs.append(input)
        if argv[1] == "inspect":
            name = argv[-1]
            rc = 0 if name in ready else 1
            return subprocess.CompletedProcess(argv, returncode=rc, stdout="", stderr="")
        if argv[1] == "exec" and "id -u" in argv[-1]:
            # Owner query: [runtime, "exec", container, "sh", "-c", <id script>],
            # deliberately WITHOUT -u 0 so it reports the image's configured user.
            container = argv[2]
            if container in ready.failing:
                if check:
                    raise subprocess.CalledProcessError(
                        1, argv, output="", stderr=_FAKE_STDERR.decode()
                    )
                return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="")
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout=f"{ready.owner}\n", stderr=""
            )
        container = argv[5] if len(argv) > 5 else None
        if container in ready.failing:
            if check:
                raise subprocess.CalledProcessError(1, argv, output=b"", stderr=_FAKE_STDERR)
            return subprocess.CompletedProcess(argv, returncode=1, stdout=b"", stderr=_FAKE_STDERR)
        return subprocess.CompletedProcess(argv, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(seeding.subprocess, "run", _fake_run)
    monkeypatch.setattr(seeding, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(seeding, "runtime_env", lambda config, base_env=None: {"FAKE": "env"})
    return calls, inputs, ready


def _write_base_md(tmp_path, content="# base context\n"):
    context_dir = tmp_path / "docker" / "web-terminal-context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "base.md").write_text(content, encoding="utf-8")
    return context_dir


def _claude_md_calls(calls):
    """Filter recorded argvs down to the CLAUDE.md exec calls (script mentions cat >).

    argv layout: [runtime, "exec", "-u", "0", "-i", container, "sh", "-c", script].
    """
    return [c for c in calls if len(c) >= 9 and "cat >" in c[8]]


def _skills_calls(calls):
    """Filter recorded argvs down to the skills-reconcile exec calls.

    argv layout: [runtime, "exec", "-u", "0", "-i", container, "sh", "-c", script,
    "sh", names, project_skills_dir].
    """
    return [c for c in calls if len(c) >= 9 and "tar -xf -" in c[8]]


# =============================================================================
# CLAUDE.md
# =============================================================================


def test_claude_md_exec_content_and_target(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path, "BASE\n")
    overlay = tmp_path / "docker" / "web-terminal-context" / "alice"
    overlay.mkdir(parents=True)
    (overlay / "extra.md").write_text("EXTRA\n", encoding="utf-8")

    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    md_calls = _claude_md_calls(calls)
    assert len(md_calls) == 1
    argv = md_calls[0]
    assert argv[0] == "docker"
    assert argv[1:6] == ["exec", "-u", "0", "-i", container]
    assert argv[6] == "sh"
    idx = calls.index(argv)
    assert inputs[idx] == b"BASE\nEXTRA\n"


def test_legacy_flat_extra_md_fallback(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    context_dir = _write_base_md(tmp_path, "BASE\n")
    (context_dir / "alice.md").write_text("LEGACY EXTRA\n", encoding="utf-8")

    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    md_calls = _claude_md_calls(calls)
    assert len(md_calls) == 1
    idx = calls.index(md_calls[0])
    assert inputs[idx] == b"BASE\nLEGACY EXTRA\n"


def test_missing_extra_md_seeds_base_only(tmp_path, monkeypatch, fake_runtime):
    """Neither <user>/extra.md nor the legacy flat <user>.md exists — base.md alone is seeded."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path, "BASE ONLY\n")

    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    md_calls = _claude_md_calls(calls)
    idx = calls.index(md_calls[0])
    assert inputs[idx] == b"BASE ONLY\n"


# =============================================================================
# skills sentinel reconcile
# =============================================================================


def test_skills_reconcile_carries_names_and_target_and_sentinel_phases(
    tmp_path, monkeypatch, fake_runtime
):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    skills_dir = tmp_path / "docker" / "web-terminal-context" / "alice" / "skills" / "myskill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("hello", encoding="utf-8")

    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    skills_calls = _skills_calls(calls)
    assert len(skills_calls) == 1
    argv = skills_calls[0]
    assert argv[1:6] == ["exec", "-u", "0", "-i", container]
    script = argv[8]
    # $0, $1 (names), $2 (target)
    assert argv[9] == "sh"
    assert argv[10] == "myskill"
    assert argv[11] == f"/app/{_FACILITY_PREFIX}-assistant/.claude/skills"

    # C3 guarantee: the three-phase sentinel dance is intact in the emitted script.
    # Phase 1 — drop deploy-managed dirs no longer shipped (gated on the sentinel file).
    assert ".deploy-managed" in script
    assert 'rm -rf -- "$d"' in script
    # Phase 2 — drop + re-extract every currently-shipped skill.
    assert 'rm -rf -- "$name"' in script
    assert "tar -xf -" in script
    # Phase 3 — re-stamp the sentinel on each shipped skill.
    assert 'touch "$name/.deploy-managed"' in script
    # Phase 1 must run before phase 2/3 clears anything currently shipped, and
    # never touches a dir lacking the sentinel (user-installed skills survive).
    assert script.index(".deploy-managed") < script.index('rm -rf -- "$name"')
    # Ownership handoff to the queried runtime user ($3), never a fixed username.
    assert 'chown -R "$owner" "$target"' in script
    assert argv[12] == "1000:1000"

    idx = calls.index(argv)
    assert inputs[idx] is not None and len(inputs[idx]) > 0  # non-empty tar stream


def test_no_catalog_config_targets_hardcoded_default_dir(tmp_path, monkeypatch, fake_runtime):
    """Zero-migration: a config with no personas catalog resolves to today's exact hardcoded
    skills path (`resolve_personas` guarantees this default), so pre-existing rosters are
    unaffected by the switch to persona-derived paths."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    skills_calls = _skills_calls(calls)
    assert len(skills_calls) == 1
    assert skills_calls[0][11] == f"/app/{_FACILITY_PREFIX}-assistant/.claude/skills"


def test_non_default_persona_drives_skills_target_from_its_own_project(
    tmp_path, monkeypatch, fake_runtime
):
    """A non-default persona's `container_project_dir` (its own `/app/<project>`, not the
    facility-prefix default) drives the per-user skills target."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    config = _config(
        [{"name": "alice", "index": 0, "persona": "beamline-ops"}],
        registry={"url": "registry.example.org"},
        web_terminals_extra={
            "personas": {"beamline-ops": {"project": "beamline-ops-app"}},
        },
    )

    seeding.seed_user_containers(config)

    skills_calls = _skills_calls(calls)
    assert len(skills_calls) == 1
    assert skills_calls[0][11] == "/app/beamline-ops-app/.claude/skills"


def test_default_persona_skills_target_follows_its_project(tmp_path, monkeypatch, fake_runtime):
    """The default persona's skills target follows its own catalog project uniformly,
    like every other persona — `/app/<persona.project>/.claude/skills` with no
    facility-prefix special case. Uses a project (`ops-app`) that does not coincide
    with the pre-persona `/app/<facility_prefix>-assistant` path to prove it."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    config = _config(
        [{"name": "alice", "index": 0}],
        registry={"url": "registry.example.org"},
        web_terminals_extra={
            "default_persona": "ops",
            "personas": {"ops": {"project": "ops-app"}},
        },
    )

    seeding.seed_user_containers(config)

    skills_calls = _skills_calls(calls)
    assert len(skills_calls) == 1
    assert skills_calls[0][11] == "/app/ops-app/.claude/skills"


def test_unresolvable_persona_raises_before_touching_runtime(tmp_path, monkeypatch, fake_runtime):
    """An unresolvable persona reference is a misconfiguration, not a per-user issue — it must
    raise before any container is even inspected, same as the missing-base.md case."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    config = _config([{"name": "alice", "index": 0, "persona": "missing"}])

    with pytest.raises(ValueError, match="missing"):
        seeding.seed_user_containers(config)

    assert calls == []  # aborted before any runtime call, not even the inspect check


def test_no_skills_overlay_still_reconciles_with_empty_tar(tmp_path, monkeypatch, fake_runtime):
    """No skills/ overlay at all — the reconcile still runs (to clean up stale managed skills)."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)

    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    skills_calls = _skills_calls(calls)
    assert len(skills_calls) == 1
    argv = skills_calls[0]
    assert argv[10] == ""  # no skill names
    idx = calls.index(argv)
    # A valid tar stream with no member entries (still carries end-of-archive
    # padding, so it isn't literally b"").
    with tarfile.open(fileobj=io.BytesIO(inputs[idx])) as tf:
        assert tf.getmembers() == []


# =============================================================================
# tolerance / hard-error semantics
# =============================================================================


def test_container_not_ready_is_skipped_others_still_seeded(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)

    ready.add(f"{_FACILITY_PREFIX}-web-bob")  # alice not ready, bob is

    seeding.seed_user_containers(_config(["alice", "bob"]))  # must not raise

    md_calls = _claude_md_calls(calls)
    seeded_containers = {c[5] for c in md_calls}
    assert seeded_containers == {f"{_FACILITY_PREFIX}-web-bob"}


def test_missing_base_md_raises_before_touching_runtime(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    # base.md intentionally not written.

    with pytest.raises(RuntimeError, match="base.md"):
        seeding.seed_user_containers(_config(["alice"]))

    assert calls == []  # aborted before any runtime call, not even the inspect check


def test_disabled_web_terminals_is_a_noop(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config["modules"]["web_terminals"]["enabled"] = False

    seeding.seed_user_containers(config)

    assert calls == []


def test_empty_roster_is_a_noop(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)

    seeding.seed_user_containers(_config([]))

    assert calls == []


# =============================================================================
# users normalization (object form) + seed_web_terminals wrapper
# =============================================================================


def test_object_form_users_are_seeded_by_name(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    container = f"{_FACILITY_PREFIX}-web-bob"
    ready.add(container)

    seeding.seed_user_containers(_config([{"name": "bob", "index": 3}]))

    md_calls = _claude_md_calls(calls)
    assert len(md_calls) == 1
    assert md_calls[0][5] == container


def test_seed_web_terminals_loads_config_and_delegates(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    config_path = _write_config(tmp_path, _config(["alice"]))

    seeding.seed_web_terminals(config_path)

    md_calls = _claude_md_calls(calls)
    assert len(md_calls) == 1
    assert md_calls[0][5] == container


# =============================================================================
# systemic-failure surfacing
# =============================================================================


def test_all_ready_containers_failing_raises_systemic_error(
    tmp_path, monkeypatch, fake_runtime, caplog
):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")
    ready.failing.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.failing.add(f"{_FACILITY_PREFIX}-web-bob")

    with caplog.at_level("WARNING", logger="deployment.web_terminals.seeding"):
        with pytest.raises(RuntimeError, match="Seeding failed for all 2 ready"):
            seeding.seed_user_containers(_config(["alice", "bob"]))

    # Each per-user warning surfaces the container's stderr, not just "exit status 1".
    assert caplog.text.count(_FAKE_STDERR.decode()) == 2


def test_one_of_two_ready_failing_does_not_raise(tmp_path, monkeypatch, fake_runtime, caplog):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")
    ready.failing.add(f"{_FACILITY_PREFIX}-web-alice")  # bob still succeeds

    with caplog.at_level("INFO", logger="deployment.web_terminals.seeding"):
        seeding.seed_user_containers(_config(["alice", "bob"]))  # must not raise

    # alice's CLAUDE.md exec was attempted (and is recorded regardless of
    # outcome) but failed before completing; bob's succeeded end-to-end.
    assert "seeded bob" in caplog.text
    assert "seeded alice" not in caplog.text
    assert _FAKE_STDERR.decode() in caplog.text


def test_all_containers_not_ready_does_not_raise(tmp_path, monkeypatch, fake_runtime):
    """Zero *ready* (attempted) containers is not a systemic failure — just an empty run."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    # Neither alice nor bob is in `ready` — both skipped as not-ready.

    seeding.seed_user_containers(_config(["alice", "bob"]))  # must not raise

    assert _claude_md_calls(calls) == []


# =============================================================================
# optional single-user targeting
# =============================================================================


def test_seed_web_terminals_with_user_seeds_only_that_user(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")

    config_path = _write_config(tmp_path, _config(["alice", "bob"]))

    seeding.seed_web_terminals(config_path, "alice")

    md_calls = _claude_md_calls(calls)
    assert {c[5] for c in md_calls} == {f"{_FACILITY_PREFIX}-web-alice"}


def test_seed_web_terminals_unknown_user_raises_value_error(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)

    config_path = _write_config(tmp_path, _config(["alice"]))

    with pytest.raises(ValueError, match="carol.*not present"):
        seeding.seed_web_terminals(config_path, "carol")

    assert calls == []  # nothing touched — not even the ready check


def test_seed_web_terminals_no_user_seeds_all(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")

    config_path = _write_config(tmp_path, _config(["alice", "bob"]))

    seeding.seed_web_terminals(config_path, None)

    md_calls = _claude_md_calls(calls)
    assert {c[5] for c in md_calls} == {
        f"{_FACILITY_PREFIX}-web-alice",
        f"{_FACILITY_PREFIX}-web-bob",
    }


# =============================================================================
# Seed ownership follows the container's runtime user
# =============================================================================


def _owner_query_calls(calls):
    """Filter recorded argvs down to the runtime-user queries (id -u based)."""
    return [c for c in calls if c[1] == "exec" and "id -u" in c[-1]]


def test_seed_chowns_to_container_runtime_user(tmp_path, monkeypatch, fake_runtime):
    """The chown owner is queried per container, not hardcoded to any username.

    The persona images create their own runtime user (uid:gid), so the seed
    scripts must receive the queried ``uid:gid`` as an argument and chown to
    that — a fixed username like ``dispatch`` breaks on any image that names
    its user differently.
    """
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.owner = "1234:5678"

    seeding.seed_user_containers(_config(["alice"]))

    owner_queries = _owner_query_calls(calls)
    assert len(owner_queries) == 1
    # The query must run as the image's configured user — no -u override.
    assert "-u" not in owner_queries[0]

    (md_call,) = _claude_md_calls(calls)
    assert md_call[-2:] == ["sh", "1234:5678"]
    assert '"$owner"' in md_call[8]

    (skills_call,) = _skills_calls(calls)
    assert skills_call[-1] == "1234:5678"
    assert '"$owner"' in skills_call[8]


def test_seed_scripts_never_hardcode_a_username(tmp_path, monkeypatch, fake_runtime):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    seeding.seed_user_containers(_config(["alice"]))

    for call in calls:
        for arg in call:
            assert "dispatch:dispatch" not in arg


def test_seed_owner_query_garbage_fails_that_user_only(tmp_path, monkeypatch, fake_runtime):
    """A non-uid:gid owner answer (e.g. an image printing a banner) must not reach chown."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")
    ready.owner = "welcome to the container\n1000:1000"

    with pytest.raises(RuntimeError, match="Seeding failed for all 2"):
        seeding.seed_user_containers(_config(["alice", "bob"]))

    assert _claude_md_calls(calls) == []  # chown never attempted with garbage
