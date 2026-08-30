"""Tests for the web-terminal ``seed`` step (osprey.deployment.web_terminals.seeding).

The container runtime is entirely mocked: ``seeding.subprocess.run`` is patched to
record every emitted argv (and its stdin ``input``) instead of touching a real
docker/podman daemon, and ``seeding.get_runtime_command``/``seeding.runtime_env``
are pinned to fixed values. The overlay tree is built under ``tmp_path`` in the
zone a real deployment keeps it in, ``build/docker/web-terminal-context/`` (see
``_context_root``). No real container is ever created, execed into, or removed
by these tests.
"""

from __future__ import annotations

import io
import os
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
        self.runtime_uid: str | None = None


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
    # Captured BEFORE the patch below: the owner branch runs the real emitted
    # script through a real shell when ``ready.runtime_uid`` is set, and
    # ``subprocess.run`` is by then this very fake.
    real_run = subprocess.run

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
            if ready.runtime_uid is not None:
                # The image declares OSPREY_RUNTIME_UID. Rather than stubbing
                # what the container "would" answer, run the REAL emitted
                # script through a real shell with that variable in the env, so
                # what is under test is the script's own precedence.
                real = real_run(
                    ["sh", "-c", argv[-1]],
                    capture_output=True,
                    text=True,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "OSPREY_RUNTIME_UID": ready.runtime_uid,
                    },
                )
                return subprocess.CompletedProcess(
                    argv, returncode=real.returncode, stdout=real.stdout, stderr=real.stderr
                )
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
    monkeypatch.setattr(seeding, "runtime_env", lambda config, base_env=None, **kw: {"FAKE": "env"})
    return calls, inputs, ready


def _context_root(tmp_path):
    """Where a built deployment repo keeps the overlay tree seeding reads.

    Inside the OUTPUT zone, not at the repo root: the tree is build output —
    ``base.md`` is installed by the framework template and each roster user's
    directory is copied in below it by the profile's ``web-terminal-context``
    convention, both into ``build/``. ``tmp_path`` stands in for the repo root
    (every test chdirs into it).
    """
    return tmp_path / "build" / "docker" / "web-terminal-context"


def _write_base_md(tmp_path, content="# base context\n"):
    context_dir = _context_root(tmp_path)
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
    overlay = _context_root(tmp_path) / "alice"
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


def test_claude_md_seed_hands_the_whole_volume_to_the_runtime_user(
    tmp_path, monkeypatch, fake_runtime
):
    """The volume chown is recursive and precedes the write (#785).

    A claude-config volume that outlived a root-running image keeps root-owned
    ``projects/`` / ``session-env/`` / ``sessions/`` under an osprey-owned top
    directory; a non-recursive chown leaves every SessionStart hook failing
    with EACCES and no transcript persisting. The seed must hand back the whole
    tree, owned by the uid:gid it queried, before it writes CLAUDE.md.
    """
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path, "BASE\n")
    container = f"{_FACILITY_PREFIX}-web-alice"
    ready.add(container)

    seeding.seed_user_containers(_config(["alice"]))

    (argv,) = _claude_md_calls(calls)
    script = argv[8]
    assert 'chown -R "$owner" /data/claude-config\n' in script
    assert script.index('chown -R "$owner" /data/claude-config') < script.index("cat > ")
    # $0, then $1 = the queried owner the recursive chown applies.
    assert argv[9:11] == ["sh", "1000:1000"]


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
# CLAUDE.md — per-persona base-prepend opt-out (seed_base)
# =============================================================================


def _write_extra_md(tmp_path, user, content):
    overlay = _context_root(tmp_path) / user
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "extra.md").write_text(content, encoding="utf-8")


def _optout_config(users, personas, **kw):
    """A config whose personas may set `seed_base`, reusing the base `_config` shape."""
    return _config(users, web_terminals_extra={"personas": personas}, **kw)


def test_seed_base_true_default_is_byte_identical(tmp_path, monkeypatch, fake_runtime):
    """A persona that keeps seed_base (default true) seeds the exact historical
    `base_content + extra_content` concatenation — byte for byte."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path, "BASE\n")
    _write_extra_md(tmp_path, "alice", "EXTRA\n")
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    config = _optout_config(
        [{"name": "alice", "index": 0, "persona": "gui"}],
        {"gui": {"project": "gui-app"}},  # no seed_base key → default true
    )
    seeding.seed_user_containers(config)

    (md_call,) = _claude_md_calls(calls)
    idx = calls.index(md_call)
    assert inputs[idx] == b"BASE\nEXTRA\n"


def test_seed_base_false_seeds_extra_alone(tmp_path, monkeypatch, fake_runtime):
    """seed_base: false → the user's CLAUDE.md is its extra.md alone, no base prepend."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path, "BASE\n")
    _write_extra_md(tmp_path, "alice", "EXTRA ONLY\n")
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    config = _optout_config(
        [{"name": "alice", "index": 0, "persona": "standalone"}],
        {"standalone": {"project": "standalone-app", "seed_base": False}},
    )
    seeding.seed_user_containers(config)

    (md_call,) = _claude_md_calls(calls)
    idx = calls.index(md_call)
    assert inputs[idx] == b"EXTRA ONLY\n"


def test_seed_base_false_tolerates_missing_base_md(tmp_path, monkeypatch, fake_runtime):
    """When every seeded user opts out, a missing base.md is not an error — the
    seed runs from each user's extra.md alone."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    # base.md deliberately NOT written; only the per-user overlay dir exists.
    _write_extra_md(tmp_path, "alice", "EXTRA ONLY\n")
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    config = _optout_config(
        [{"name": "alice", "index": 0, "persona": "standalone"}],
        {"standalone": {"project": "standalone-app", "seed_base": False}},
    )
    seeding.seed_user_containers(config)  # must not raise despite missing base.md

    (md_call,) = _claude_md_calls(calls)
    idx = calls.index(md_call)
    assert inputs[idx] == b"EXTRA ONLY\n"


def test_mixed_roster_base_user_and_optout_user(tmp_path, monkeypatch, fake_runtime):
    """A roster mixing a seed_base=true user and a seed_base=false user: the
    former gets base+extra, the latter gets extra alone, in one run."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path, "BASE\n")
    _write_extra_md(tmp_path, "alice", "ALICE EXTRA\n")
    _write_extra_md(tmp_path, "bob", "BOB EXTRA\n")
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")

    config = _optout_config(
        [
            {"name": "alice", "index": 0, "persona": "gui"},
            {"name": "bob", "index": 1, "persona": "standalone"},
        ],
        {
            "gui": {"project": "gui-app"},  # default seed_base true
            "standalone": {"project": "standalone-app", "seed_base": False},
        },
    )
    seeding.seed_user_containers(config)

    payload_by_container = {c[5]: inputs[calls.index(c)] for c in _claude_md_calls(calls)}
    assert payload_by_container[f"{_FACILITY_PREFIX}-web-alice"] == b"BASE\nALICE EXTRA\n"
    assert payload_by_container[f"{_FACILITY_PREFIX}-web-bob"] == b"BOB EXTRA\n"


def test_mixed_roster_missing_base_md_still_raises(tmp_path, monkeypatch, fake_runtime):
    """If even one seeded user keeps seed_base=true, a missing base.md still aborts
    the whole seed up front — before any container is touched."""
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    # base.md deliberately NOT written.
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.add(f"{_FACILITY_PREFIX}-web-bob")

    config = _optout_config(
        [
            {"name": "alice", "index": 0, "persona": "gui"},  # keeps base
            {"name": "bob", "index": 1, "persona": "standalone"},  # opts out
        ],
        {
            "gui": {"project": "gui-app"},
            "standalone": {"project": "standalone-app", "seed_base": False},
        },
    )
    with pytest.raises(RuntimeError, match="base.md"):
        seeding.seed_user_containers(config)

    assert calls == []  # aborted before any runtime call


# =============================================================================
# skills sentinel reconcile
# =============================================================================


def test_skills_reconcile_carries_names_and_target_and_sentinel_phases(
    tmp_path, monkeypatch, fake_runtime
):
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    skills_dir = _context_root(tmp_path) / "alice" / "skills" / "myskill"
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
    assert argv[11] == f"/app/{_FACILITY_PREFIX}-assistant/build/.claude/skills"

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
    # The render zone the target now lives in is root-owned, so the reconcile
    # chowns to root — never to the container's runtime user, who could
    # otherwise rewrite the skills the next session loads. The queried owner is
    # still handed over as $3 (asserted below), for call-shape parity with the
    # CLAUDE.md seed, which does chown to it.
    assert 'chown -R 0:0 "$target"' in script
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
    assert skills_calls[0][11] == f"/app/{_FACILITY_PREFIX}-assistant/build/.claude/skills"


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
    assert skills_calls[0][11] == "/app/beamline-ops-app/build/.claude/skills"


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
    assert skills_calls[0][11] == "/app/ops-app/build/.claude/skills"


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

    # DEBUG, not INFO: the per-user "seeded <user>" line is debug-grade now
    # (disposition row 18) -- the default view gets the loop's count instead.
    with caplog.at_level("DEBUG", logger="deployment.web_terminals.seeding"):
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
    scripts must receive the queried ``uid:gid`` as an argument — a fixed
    username like ``dispatch`` breaks on any image that names its user
    differently. The CLAUDE.md seed chowns to it; the skills reconcile, whose
    target sits in the root-owned render zone, chowns to root instead.
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
    # Skills land in the root-owned render zone, so this one chowns to root
    # regardless of the queried owner.
    assert 'chown -R 0:0 "$target"' in skills_call[8]


def _run_owner_query(**env):
    """Run the REAL emitted owner-query script through a real shell.

    ``seeding._OWNER_QUERY_SH`` is the one piece of this module that is shell,
    not Python: mocking it away would test a stub. ``env`` replaces the
    process environment wholesale (``PATH`` is added so ``id`` resolves), which
    is what lets a test say "this image declares OSPREY_RUNTIME_UID" and
    "this one does not" without touching a container.
    """
    result = subprocess.run(
        ["sh", "-c", seeding._OWNER_QUERY_SH],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_owner_query_prefers_the_images_declared_runtime_uid():
    """OSPREY_RUNTIME_UID wins over `id`.

    `id` reports whoever the exec happens to run as, which is not necessarily
    the user the image declares its processes run as. When the image says so
    outright, that answer is authoritative.
    """
    assert _run_owner_query(OSPREY_RUNTIME_UID="4242:4243") == "4242:4243"


def test_owner_query_completes_a_bare_uid_with_the_current_group():
    """A bare uid is accepted, the group coming from `id -g`."""
    gid = subprocess.run(["id", "-g"], capture_output=True, text=True, check=True).stdout.strip()
    assert _run_owner_query(OSPREY_RUNTIME_UID="4242") == f"4242:{gid}"


def test_owner_query_falls_back_to_id_when_the_image_declares_nothing():
    """Images predating OSPREY_RUNTIME_UID keep working off `id -u`/`id -g`."""
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
    gid = subprocess.run(["id", "-g"], capture_output=True, text=True, check=True).stdout.strip()
    assert _run_owner_query() == f"{uid}:{gid}"
    # An image that exports it EMPTY must fall back too, not chown to ":gid".
    assert _run_owner_query(OSPREY_RUNTIME_UID="") == f"{uid}:{gid}"


def test_runtime_uid_in_the_container_becomes_the_seed_owner(tmp_path, monkeypatch, fake_runtime):
    """End to end: the container's OSPREY_RUNTIME_UID is what CLAUDE.md is chowned to.

    The fixture answers the owner query by running the real script with that
    variable set, so this pins the whole path — emitted script, container env,
    argument handoff — rather than just the shell snippet.
    """
    calls, inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    _write_base_md(tmp_path)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")
    ready.owner = "1000:1000"  # what `id` would have said — must NOT win
    ready.runtime_uid = "7000:7001"

    seeding.seed_user_containers(_config(["alice"]))

    (md_call,) = _claude_md_calls(calls)
    assert md_call[-2:] == ["sh", "7000:7001"]
    (skills_call,) = _skills_calls(calls)
    assert skills_call[-1] == "7000:7001"


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


# =============================================================================
# where the overlay tree is read from
# =============================================================================


def test_overlay_at_the_repo_root_is_not_read(tmp_path, monkeypatch, fake_runtime):
    """A tree at ``<repo>/docker/web-terminal-context`` is not consulted.

    That path is what a cwd-relative join produces, and in a deployment repo it
    is not where the build puts the overlay — ``build/`` is. A ``base.md``
    sitting there must not satisfy the requirement, or the seed would run off
    whatever predated the build.
    """
    _calls, _inputs, ready = fake_runtime
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "docker" / "web-terminal-context"
    stale.mkdir(parents=True)
    (stale / "base.md").write_text("STALE BASE\n", encoding="utf-8")
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    with pytest.raises(RuntimeError, match="base.md not found"):
        seeding.seed_user_containers(_config(["alice"]))


def test_overlay_is_found_from_the_config_not_the_working_directory(
    tmp_path, monkeypatch, fake_runtime
):
    """``osprey users seed`` hands over a config path; the overlay follows it.

    Running from anywhere else must still read the repo that config belongs to,
    the same way the compose invocation pins ``--project-directory``.
    """
    calls, inputs, ready = fake_runtime
    repo = tmp_path / "repo"
    _write_base_md(repo, "REPO BASE\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    ready.add(f"{_FACILITY_PREFIX}-web-alice")

    config_path = repo / "build" / "config.yml"
    config_path.write_text(yaml.safe_dump(_config(["alice"])), encoding="utf-8")
    seeding.seed_web_terminals(str(config_path))

    md_calls = _claude_md_calls(calls)
    assert len(md_calls) == 1
    assert inputs[calls.index(md_calls[0])] == b"REPO BASE\n"
