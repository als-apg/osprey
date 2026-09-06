"""The posture floor: the six keys a build refuses to guess at.

Two halves, and they check different things.

The **unit** half exercises :func:`~osprey.cli.build_posture_check.missing_posture_errors`
directly over hand-written rendered configs. It is where the conditional rules
live: which keys a controls-less standalone is asked for, which a profile that
never selected the approval hook is asked for, and what counts as "stated".

The **CLI** half runs the real ``osprey init`` + ``osprey build``, because the
floor is only worth anything if it is wired into the gate a build actually
passes through. Each of the four presets is built once and cached for the
session; the refusals then take one built hello-world repo, delete a single
required line from its ``profile.yml``, and rebuild — so the assertion is on the
message the CLI prints, not on a function called in isolation.

Running it::

    uv run --extra dev pytest tests/cli/test_posture_floor.py -q

The CLI half is marked ``slow``. Each build is seconds, not minutes: the
presets that carry a virtual accelerator do the most work, and every cell is
built at most once per session.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_posture_check import (
    APPROVAL_HOOK,
    PROFILE_FILENAME,
    REQUIRED_ALWAYS,
    REQUIRED_WITH_APPROVAL_HOOK,
    REQUIRED_WITH_CONTROLS,
    missing_posture_errors,
)

#: The presets the feature converts, and the ones the floor has to leave alone.
PRESETS: tuple[str, ...] = (
    "hello-world",
    "control-assistant",
    "ariel-standalone",
    "channel-finder-standalone",
)

#: Deployment directory name for every built cell. Becomes ``project_name``.
PROJECT_NAME = "osprey-posture"

#: The complete key set the floor can require, in the order it reports them.
#: The config-key manifest's ``required`` column is a superset of this set;
#: these six are the ones the build gate itself enforces, and a change here
#: without a matching change there would leave the two disagreeing about what
#: a deployment must state.
FLOOR_KEYS: tuple[str, ...] = tuple(
    key for key, _ in (*REQUIRED_WITH_CONTROLS, *REQUIRED_WITH_APPROVAL_HOOK, *REQUIRED_ALWAYS)
)


# ─────────────────────────────────────────────────────────────────────────────
# Rendered-config fakes for the unit half
# ─────────────────────────────────────────────────────────────────────────────


def _rendered(**overrides: Any) -> dict[str, Any]:
    """A rendered config that states every floor key, with *overrides* applied.

    The baseline says yes to everything, so a test that removes one key is
    testing exactly that key. ``claude_code.servers`` is left unset, which is
    the shape of a render that never overrode the registry: the controls server
    is on by its own default.

    Args:
        overrides: Top-level sections to replace wholesale.

    Returns:
        A fresh mapping in the shape ``yaml.safe_load`` produces.
    """
    document: dict[str, Any] = {
        "control_system": {"type": "mock"},
        "archiver": {"type": "mock_archiver"},
        "approval": {"enabled": True, "default_policy": "always"},
        "claude_code": {"telemetry": {"enabled": True}},
        "hooks": {"debug": False},
    }
    document.update(overrides)
    return document


def _named_keys(errors: list[str]) -> list[str]:
    """The dotted key each refusal line opens with.

    Args:
        errors: Lines from :func:`missing_posture_errors`.

    Returns:
        One key per line, in the order the lines came.
    """
    return [line.split(" ", 1)[0] for line in errors]


# ─────────────────────────────────────────────────────────────────────────────
# The floor's key set
# ─────────────────────────────────────────────────────────────────────────────


class TestFloorKeySet:
    """What the floor can ask for, and nothing else."""

    def test_the_floor_is_exactly_six_keys(self) -> None:
        """The set is pinned: these six are the ones the build gate enforces,

        out of the larger ``required`` set the manifest marks.
        """
        assert FLOOR_KEYS == (
            "control_system.type",
            "archiver.type",
            "approval.enabled",
            "approval.default_policy",
            "claude_code.telemetry.enabled",
            "hooks.debug",
        )

    def test_a_render_that_states_everything_is_not_refused(self) -> None:
        """The floor is a gate, so a complete render passes it silently."""
        assert missing_posture_errors(_rendered(), [APPROVAL_HOOK]) == []

    def test_every_refusal_names_its_key_and_the_file_to_edit(self) -> None:
        """A line an operator can act on: the dotted key, then ``profile.yml``."""
        errors = missing_posture_errors({}, [APPROVAL_HOOK])
        assert _named_keys(errors) == list(FLOOR_KEYS)
        for key, line in zip(FLOOR_KEYS, errors, strict=True):
            assert line.startswith(f"{key} ")
            assert PROFILE_FILENAME in line
            assert f"config: {key}:" in line


# ─────────────────────────────────────────────────────────────────────────────
# What makes a key required
# ─────────────────────────────────────────────────────────────────────────────


class TestControlsGate:
    """``control_system.type`` and ``archiver.type`` follow the controls server."""

    def test_controls_on_by_registry_default_requires_both(self) -> None:
        """A render that says nothing about servers still runs the controls one."""
        errors = missing_posture_errors(_rendered(control_system={}, archiver={}), [APPROVAL_HOOK])
        assert _named_keys(errors) == ["control_system.type", "archiver.type"]

    def test_controls_switched_off_requires_neither(self) -> None:
        """The two standalone presets' shape: no controls server, no type to state."""
        rendered = _rendered(
            control_system={},
            archiver={},
            claude_code={
                "telemetry": {"enabled": True},
                "servers": {"controls": {"enabled": False}},
            },
        )
        assert missing_posture_errors(rendered, [APPROVAL_HOOK]) == []

    def test_controls_switched_on_explicitly_requires_both(self) -> None:
        """An explicit ``enabled: true`` is read the same as the default."""
        rendered = _rendered(
            control_system={},
            archiver={},
            claude_code={
                "telemetry": {"enabled": True},
                "servers": {"controls": {"enabled": True}},
            },
        )
        assert _named_keys(missing_posture_errors(rendered, [])) == [
            "control_system.type",
            "archiver.type",
        ]


class TestApprovalHookGate:
    """The approval pair follows the profile's hook selection, not the render."""

    def test_hook_selected_requires_both_leaves(self) -> None:
        assert _named_keys(missing_posture_errors(_rendered(approval={}), [APPROVAL_HOOK])) == [
            "approval.enabled",
            "approval.default_policy",
        ]

    def test_hook_not_selected_requires_neither(self) -> None:
        """A profile with no approval hook has no approval posture to state."""
        assert missing_posture_errors(_rendered(approval={}), ["hook-log", "limits"]) == []

    def test_one_leaf_stated_still_names_the_other(self) -> None:
        """Half an answer is not one: the missing leaf is named on its own."""
        rendered = _rendered(approval={"enabled": True})
        assert _named_keys(missing_posture_errors(rendered, [APPROVAL_HOOK])) == [
            "approval.default_policy"
        ]


class TestUnconditionalKeys:
    """Telemetry and hook debugging are asked of every deployment."""

    def test_required_with_no_controls_and_no_approval_hook(self) -> None:
        rendered = {"claude_code": {"servers": {"controls": {"enabled": False}}}}
        assert _named_keys(missing_posture_errors(rendered, [])) == [
            "claude_code.telemetry.enabled",
            "hooks.debug",
        ]

    def test_false_is_an_answer(self) -> None:
        """The floor asks to be told, not to be told ``true``."""
        rendered = _rendered(
            claude_code={
                "telemetry": {"enabled": False},
                "servers": {"controls": {"enabled": False}},
            },
            hooks={"debug": False},
        )
        assert missing_posture_errors(rendered, [APPROVAL_HOOK]) == []


class TestWhatCountsAsStated:
    """Silence has more than one spelling, and they all read as silence."""

    @pytest.mark.parametrize(
        ("hooks_section", "why"),
        [
            ({}, "the block exists but carries no leaf"),
            ({"debug": None}, "the leaf exists but carries no value"),
            (None, "the block itself is null"),
            ("yes", "the block is not a mapping at all"),
        ],
    )
    def test_hooks_debug_unstated(self, hooks_section: Any, why: str) -> None:
        errors = missing_posture_errors(_rendered(hooks=hooks_section), [])
        assert _named_keys(errors) == ["hooks.debug"], why

    def test_a_nested_key_is_read_through_its_parents(self) -> None:
        """``claude_code.telemetry.enabled`` needs every segment, not just the leaf."""
        rendered = _rendered(claude_code={"telemetry": {}})
        assert _named_keys(missing_posture_errors(rendered, [])) == [
            "claude_code.telemetry.enabled"
        ]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_string_leaf_is_unstated(self, blank: str) -> None:
        """An emptied-but-quoted value is the same silence as a missing one."""
        rendered = _rendered(control_system={"type": blank})
        assert "control_system.type" in _named_keys(missing_posture_errors(rendered, []))


# ─────────────────────────────────────────────────────────────────────────────
# The CLI half
# ─────────────────────────────────────────────────────────────────────────────


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one ``osprey`` command under the running interpreter.

    Invoked as ``[sys.executable, "-m", "osprey", …]`` rather than a bare
    ``osprey``: PATH may resolve to a different install whose presets diverge
    from the tree under test.

    Args:
        args: Arguments after ``osprey``.
        cwd: Working directory for the command.

    Returns:
        The completed process, output captured, a non-zero exit not raised.
    """
    return subprocess.run(
        [sys.executable, "-m", "osprey", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    """Everything a CLI run wrote, for an assertion or a failure message."""
    return f"{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="session")
def built_preset(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build a preset once per session and hand back its repo.

    Args:
        tmp_path_factory: Pytest's session-scoped directory factory.

    Returns:
        A callable taking a preset name and returning the built repo directory.
        Each preset is initialised and built at most once; a failure of either
        verb fails the calling test with the CLI's own output.
    """
    built: dict[str, Path] = {}

    def _build(preset: str) -> Path:
        if preset in built:
            return built[preset]
        workspace = tmp_path_factory.mktemp(f"posture-{preset}")
        init = _run_cli(["init", PROJECT_NAME, "--preset", preset, "--no-git"], workspace)
        assert init.returncode == 0, f"osprey init {preset} failed:\n{_output(init)}"
        repo = workspace / PROJECT_NAME
        build = _run_cli(["build"], repo)
        assert build.returncode == 0, f"osprey build ({preset}) failed:\n{_output(build)}"
        built[preset] = repo
        return repo

    return _build


@pytest.mark.slow
@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_builds_green(preset: str, built_preset: Any) -> None:
    """Every shipped preset already states the posture its surfaces read.

    The floor is a gate over keys the presets carry, so this is the assertion
    that adding it regressed nothing: ``osprey init`` then ``osprey build``,
    end to end, with no ``--set`` and no edit. The fixture fails with the CLI's
    own output if either verb refuses.
    """
    repo = built_preset(preset)
    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text())
    assert missing_posture_errors(rendered, _selected_hooks(repo)) == []


def _selected_hooks(repo: Path) -> list[str]:
    """The ``hooks:`` list a built repo's profile selects.

    Read from the emitted profile rather than the preset, because the profile
    is what the build resolves; a persona's selection is merged into it before
    the gate sees it.

    Args:
        repo: A built deployment repo.

    Returns:
        The hook names, or an empty list when the profile selects none.
    """
    profile = yaml.safe_load((repo / PROFILE_FILENAME).read_text()) or {}
    return list(profile.get("hooks") or [])


@pytest.mark.slow
def test_standalone_without_controls_is_not_asked_for_a_control_system_type(
    built_preset: Any,
) -> None:
    """The ARIEL standalone builds green while stating no ``control_system.type``.

    It switches the controls server off, so the pair the server would make
    required is not a posture it has. This is the half of the rule that a floor
    written as an unconditional list would get wrong: the build would refuse a
    preset that ships as intended.
    """
    repo = built_preset("ariel-standalone")
    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text())
    assert "type" not in (rendered.get("control_system") or {})
    assert missing_posture_errors(rendered, _selected_hooks(repo)) == []


@pytest.mark.slow
@pytest.mark.parametrize("key", FLOOR_KEYS)
def test_removing_a_required_key_is_refused_naming_it(
    key: str, built_preset: Any, tmp_path: Path
) -> None:
    """A built repo, one line deleted from its profile, refuses to rebuild.

    hello-world is the base because it is the one preset whose surfaces read
    all six: it runs the controls server and selects the approval hook. The
    edit is the deletion of exactly one dotted ``config:`` line, which is what
    an operator trimming a profile does, and the rebuild has to name that key
    and the file it belongs in.
    """
    source = built_preset("hello-world")
    repo = tmp_path / PROJECT_NAME
    shutil.copytree(source, repo)

    profile = repo / PROFILE_FILENAME
    lines = profile.read_text().splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(f"  {key}:")]
    assert len(kept) == len(lines) - 1, f"expected exactly one `  {key}:` line in {profile}"
    profile.write_text("".join(kept))

    result = _run_cli(["build"], repo)
    output = _output(result)
    assert result.returncode != 0, f"build should have refused a profile with no {key}:\n{output}"
    assert f"{key} is not stated" in output
    assert PROFILE_FILENAME in output
