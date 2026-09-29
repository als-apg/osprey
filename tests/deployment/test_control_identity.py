"""Tests for the stdlib-only control-identity leaf module."""

from __future__ import annotations

import ast
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.deployment import control_identity
from osprey.deployment.control_identity import (
    BASE_IMAGE_ACCOUNTS,
    CANONICAL_NAME,
    CONTROL_IDENTITY_CONTAINER_PATH,
    IDENTITY_RE,
    RESERVED_RE,
    append_group_members,
    apply,
    main,
    rewrite,
    validate_identity,
)


class TestValidateLeafModule:
    """The module is copied into containers and run by path: stdlib only."""

    def test_validate_module_imports_only_stdlib(self) -> None:
        path = Path(control_identity.__file__ or "")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [name for name in imported if name.split(".")[0] == "osprey"]

    def test_validate_module_runs_by_path_without_the_package(self) -> None:
        """Run as a script from outside the tree; the guard must not exit 0 silently."""
        path = Path(control_identity.__file__ or "")
        result = subprocess.run(
            [sys.executable, "-I", str(path)],
            capture_output=True,
            text=True,
            cwd=path.parent.parent,
        )
        assert "Traceback" not in result.stderr, result.stderr
        assert result.returncode != 0

    def test_validate_constants(self) -> None:
        assert CONTROL_IDENTITY_CONTAINER_PATH == "/opt/osprey/control_identity.py"
        assert CANONICAL_NAME == {1000: "osprey", 0: "root"}
        assert IDENTITY_RE.pattern == r"^[a-z_][a-z0-9_-]{0,31}\Z"
        assert RESERVED_RE.pattern == r"^(root|osprey(-[a-z0-9_-]*)?)\Z"

    def test_validate_base_image_accounts_exclude_osprey_owned_names(self) -> None:
        assert "root" not in BASE_IMAGE_ACCOUNTS
        assert "osprey" not in BASE_IMAGE_ACCOUNTS
        # A sample of Debian base-passwd accounts present in both slim images.
        assert {"daemon", "bin", "nobody", "www-data", "backup", "list", "proxy", "_apt"} <= (
            BASE_IMAGE_ACCOUNTS
        )
        # Every entry is itself a syntactically valid login name.
        assert all(IDENTITY_RE.fullmatch(name) for name in BASE_IMAGE_ACCOUNTS)


class TestValidateIdentity:
    @pytest.mark.parametrize(
        "name",
        ["alice", "carol", "_svc", "a", "a" * 32, "carol-2", "alice_b", "x9"],
    )
    def test_validate_accepts_plain_names(self, name: str) -> None:
        validate_identity(name)
        validate_identity(name, allow_service=True)

    @pytest.mark.parametrize("value", [None, 1000, b"alice", ["alice"], {"name": "alice"}])
    def test_validate_refuses_non_string(self, value: object) -> None:
        with pytest.raises(ValueError, match="string"):
            validate_identity(value)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "Alice",
            "1alice",
            "-alice",
            "al ice",
            "alice.b",
            "alicé",
            "a" * 33,
            "alice$",
        ],
    )
    def test_validate_refuses_bad_charset(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_identity(name)
        with pytest.raises(ValueError):
            validate_identity(name, allow_service=True)

    @pytest.mark.parametrize(
        "name",
        ["alice:x", "alice\n", "alice\nroot", "alice\r", "a:0:0::/:/bin/sh"],
    )
    def test_validate_refuses_passwd_separators(self, name: str) -> None:
        """``fullmatch`` with ``\\Z`` — a trailing newline cannot slip past ``$``."""
        with pytest.raises(ValueError):
            validate_identity(name)
        with pytest.raises(ValueError):
            validate_identity(name, allow_service=True)

    @pytest.mark.parametrize("name", ["root", "osprey"])
    def test_validate_refuses_canonical_names_even_for_services(self, name: str) -> None:
        with pytest.raises(ValueError, match="reserved"):
            validate_identity(name)
        with pytest.raises(ValueError, match="reserved"):
            validate_identity(name, allow_service=True)

    @pytest.mark.parametrize(
        "name", ["osprey-", "osprey-bluesky", "osprey-bluesky-va", "osprey-x_1"]
    )
    def test_validate_reserves_service_names_for_osprey(self, name: str) -> None:
        with pytest.raises(ValueError, match="reserved"):
            validate_identity(name)
        validate_identity(name, allow_service=True)

    @pytest.mark.parametrize("name", ["ospreyx", "osprey_x", "rooter", "xroot", "my-osprey"])
    def test_validate_reserved_match_is_anchored(self, name: str) -> None:
        validate_identity(name)

    @pytest.mark.parametrize("name", sorted(BASE_IMAGE_ACCOUNTS))
    def test_validate_refuses_base_image_accounts(self, name: str) -> None:
        with pytest.raises(ValueError, match="base image"):
            validate_identity(name)
        with pytest.raises(ValueError, match="base image"):
            validate_identity(name, allow_service=True)

    def test_validate_returns_none(self) -> None:
        assert validate_identity("alice") is None


ROOT_LINE = "root:x:0:0:root:/root:/bin/bash"
DAEMON_LINE = "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin"
OSPREY_LINE = "osprey:x:1000:1000::/home/osprey:/bin/bash"
PASSWD = "\n".join([ROOT_LINE, DAEMON_LINE, OSPREY_LINE]) + "\n"


class TestRewrite:
    """``rewrite`` puts one identity line directly above the canonical line."""

    def test_rewrite_inserts_identity_above_osprey(self) -> None:
        out = rewrite(PASSWD, 1000, "alice")
        assert out.splitlines() == [
            ROOT_LINE,
            DAEMON_LINE,
            "alice:x:1000:1000::/home/osprey:/bin/bash",
            OSPREY_LINE,
        ]

    def test_rewrite_inserts_identity_above_root(self) -> None:
        out = rewrite(PASSWD, 0, "alice")
        assert out.splitlines() == [
            "alice:x:0:0:root:/root:/bin/bash",
            ROOT_LINE,
            DAEMON_LINE,
            OSPREY_LINE,
        ]

    def test_rewrite_replaces_only_the_name_field(self) -> None:
        out = rewrite(PASSWD, 1000, "alice")
        identity = out.splitlines()[2].split(":")
        assert identity[0] == "alice"
        assert identity[1:] == OSPREY_LINE.split(":")[1:]

    def test_rewrite_accepts_service_names(self) -> None:
        out = rewrite(PASSWD, 1000, "osprey-bluesky-va")
        assert "osprey-bluesky-va:x:1000:1000::/home/osprey:/bin/bash" in out.splitlines()

    def test_rewrite_is_idempotent(self) -> None:
        once = rewrite(PASSWD, 1000, "alice")
        assert rewrite(once, 1000, "alice") == once

    def test_rewrite_changed_identity_replaces_previous(self) -> None:
        out = rewrite(rewrite(PASSWD, 1000, "alice"), 1000, "carol")
        assert out == rewrite(PASSWD, 1000, "carol")
        assert not any(line.startswith("alice:") for line in out.splitlines())

    def test_rewrite_drops_every_other_line_with_the_uid(self) -> None:
        text = PASSWD + "alice:x:1000:1000::/home/alice:/bin/sh\n"
        out = rewrite(text, 1000, "carol")
        uid_lines = [line for line in out.splitlines() if line.split(":")[2] == "1000"]
        assert uid_lines == ["carol:x:1000:1000::/home/osprey:/bin/bash", OSPREY_LINE]

    def test_rewrite_leaves_other_uids_alone(self) -> None:
        out = rewrite(PASSWD, 1000, "alice")
        assert ROOT_LINE in out.splitlines()
        assert DAEMON_LINE in out.splitlines()
        assert rewrite(out, 0, "carol").splitlines()[2:] == [
            DAEMON_LINE,
            "alice:x:1000:1000::/home/osprey:/bin/bash",
            OSPREY_LINE,
        ]

    def test_rewrite_preserves_trailing_newline(self) -> None:
        assert rewrite(PASSWD, 1000, "alice").endswith("\n")
        assert not rewrite(PASSWD.rstrip("\n"), 1000, "alice").endswith("\n")

    def test_rewrite_keeps_blank_and_malformed_lines(self) -> None:
        text = ROOT_LINE + "\n\n# comment\n" + OSPREY_LINE + "\n"
        out = rewrite(text, 1000, "alice")
        assert out.splitlines() == [
            ROOT_LINE,
            "",
            "# comment",
            "alice:x:1000:1000::/home/osprey:/bin/bash",
            OSPREY_LINE,
        ]

    def test_rewrite_refuses_missing_canonical_line(self) -> None:
        text = ROOT_LINE + "\n" + DAEMON_LINE + "\n"
        with pytest.raises(ValueError, match="osprey"):
            rewrite(text, 1000, "alice")

    def test_rewrite_refuses_canonical_name_with_wrong_uid(self) -> None:
        text = ROOT_LINE + "\nosprey:x:1001:1001::/home/osprey:/bin/bash\n"
        with pytest.raises(ValueError):
            rewrite(text, 1000, "alice")

    def test_rewrite_refuses_unrewritable_uid(self) -> None:
        with pytest.raises(ValueError, match="uid"):
            rewrite(PASSWD, 1, "alice")

    def test_rewrite_refuses_collision_with_another_uid(self) -> None:
        text = PASSWD + "alice:x:1001:1001::/home/alice:/bin/sh\n"
        with pytest.raises(ValueError, match="alice"):
            rewrite(text, 1000, "alice")

    def test_rewrite_refuses_name_held_by_the_other_canonical_uid(self) -> None:
        once = rewrite(PASSWD, 0, "alice")
        with pytest.raises(ValueError, match="alice"):
            rewrite(once, 1000, "alice")

    @pytest.mark.parametrize("name", ["root", "osprey", "daemon", "Alice", "alice\n", "a:b"])
    def test_rewrite_refuses_invalid_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            rewrite(PASSWD, 1000, name)


GROUP_ROOT = "root:x:0:"
GROUP_OSPREY = "osprey:x:1000:"
GROUP_MOUNT = "osprey-mount-4242:x:4242:osprey"
GROUP_NAMED = "data:x:5000:bob,osprey,carol"
GROUP_OTHER = "users:x:100:bob"
GROUP = "\n".join([GROUP_ROOT, GROUP_OSPREY, GROUP_MOUNT, GROUP_NAMED, GROUP_OTHER]) + "\n"


class TestAppendGroupMembers:
    """``append_group_members`` adds the identity beside the canonical member."""

    def test_group_append_adds_name_where_canonical_is_a_member(self) -> None:
        out = append_group_members(GROUP, "osprey", "alice")
        assert out.splitlines() == [
            GROUP_ROOT,
            GROUP_OSPREY,
            "osprey-mount-4242:x:4242:osprey,alice",
            "data:x:5000:bob,osprey,carol,alice",
            GROUP_OTHER,
        ]

    def test_group_append_leaves_primary_group_untouched(self) -> None:
        out = append_group_members(GROUP, "osprey", "alice")
        assert GROUP_OSPREY in out.splitlines()

    def test_group_append_leaves_groups_without_canonical_untouched(self) -> None:
        out = append_group_members(GROUP, "osprey", "alice")
        assert GROUP_OTHER in out.splitlines()
        assert GROUP_ROOT in out.splitlines()

    def test_group_append_matches_whole_member_tokens_only(self) -> None:
        text = "g:x:6000:osprey-x,xosprey\n"
        assert append_group_members(text, "osprey", "alice") == text

    def test_group_append_is_idempotent(self) -> None:
        once = append_group_members(GROUP, "osprey", "alice")
        assert append_group_members(once, "osprey", "alice") == once

    def test_group_append_skips_name_already_a_member(self) -> None:
        text = "g:x:6000:alice,osprey\n"
        assert append_group_members(text, "osprey", "alice") == text

    def test_group_append_never_removes_a_previous_identity(self) -> None:
        once = append_group_members(GROUP, "osprey", "alice")
        out = append_group_members(once, "osprey", "carol")
        assert "osprey-mount-4242:x:4242:osprey,alice,carol" in out.splitlines()

    def test_group_append_for_root_canonical(self) -> None:
        text = "wheel:x:10:root\n"
        assert append_group_members(text, "root", "alice") == "wheel:x:10:root,alice\n"

    def test_group_append_preserves_trailing_newline(self) -> None:
        assert append_group_members(GROUP, "osprey", "alice").endswith("\n")
        stripped = GROUP.rstrip("\n")
        assert not append_group_members(stripped, "osprey", "alice").endswith("\n")

    def test_group_append_keeps_blank_and_malformed_lines(self) -> None:
        text = GROUP_MOUNT + "\n\n# comment\nbroken:osprey\n"
        out = append_group_members(text, "osprey", "alice")
        assert out.splitlines() == [
            "osprey-mount-4242:x:4242:osprey,alice",
            "",
            "# comment",
            "broken:osprey",
        ]

    def test_group_append_accepts_service_names(self) -> None:
        out = append_group_members(GROUP, "osprey", "osprey-bluesky-va")
        assert "osprey-mount-4242:x:4242:osprey,osprey-bluesky-va" in out.splitlines()

    def test_group_append_refuses_non_canonical_account(self) -> None:
        with pytest.raises(ValueError, match="canonical"):
            append_group_members(GROUP, "bob", "alice")

    @pytest.mark.parametrize("name", ["root", "osprey", "daemon", "Alice", "a,b", "a:b"])
    def test_group_append_refuses_invalid_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            append_group_members(GROUP, "osprey", name)


MODULE_PATH = Path(control_identity.__file__ or "")


def _write_targets(tmp_path: Path, passwd: str = PASSWD, group: str = GROUP) -> tuple[Path, Path]:
    etc = tmp_path / "etc"
    etc.mkdir()
    passwd_path = etc / "passwd"
    group_path = etc / "group"
    passwd_path.write_text(passwd, encoding="utf-8")
    group_path.write_text(group, encoding="utf-8")
    return passwd_path, group_path


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(MODULE_PATH), *args],
        capture_output=True,
        text=True,
        cwd=MODULE_PATH.parent.parent,
    )


def _apply_args(uid: int, name: str, passwd: Path, group: Path) -> list[str]:
    return [
        "apply",
        "--uid",
        str(uid),
        "--name",
        name,
        "--passwd",
        str(passwd),
        "--group",
        str(group),
    ]


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name not in {"passwd", "group"})


def _assert_one_line_failure(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Traceback" not in result.stderr, result.stderr
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert result.stderr.endswith("\n")


class TestApplyCli:
    """``python control_identity.py apply ...`` run by path, on temp files."""

    def test_apply_script_rewrites_both_files(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        result = _run_script(*_apply_args(1000, "alice", passwd, group))
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
        assert result.stderr == ""
        assert passwd.read_text(encoding="utf-8") == rewrite(PASSWD, 1000, "alice")
        assert group.read_text(encoding="utf-8") == append_group_members(GROUP, "osprey", "alice")

    def test_apply_script_writes_mode_0644(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        passwd.chmod(0o600)
        group.chmod(0o600)
        result = _run_script(*_apply_args(1000, "alice", passwd, group))
        assert result.returncode == 0, result.stderr
        assert stat.S_IMODE(passwd.stat().st_mode) == 0o644
        assert stat.S_IMODE(group.stat().st_mode) == 0o644

    def test_apply_script_leaves_no_temp_file_on_success(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        assert _run_script(*_apply_args(1000, "alice", passwd, group)).returncode == 0
        assert _leftovers(passwd.parent) == []

    def test_apply_script_is_idempotent(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        assert _run_script(*_apply_args(1000, "alice", passwd, group)).returncode == 0
        first = (passwd.read_text(encoding="utf-8"), group.read_text(encoding="utf-8"))
        assert _run_script(*_apply_args(1000, "alice", passwd, group)).returncode == 0
        assert (passwd.read_text(encoding="utf-8"), group.read_text(encoding="utf-8")) == first

    def test_apply_script_identity_resolves_first_for_uid(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        assert _run_script(*_apply_args(1000, "alice", passwd, group)).returncode == 0
        uid_lines = [
            line
            for line in passwd.read_text(encoding="utf-8").splitlines()
            if line.split(":")[2] == "1000"
        ]
        assert uid_lines == ["alice:x:1000:1000::/home/osprey:/bin/bash", OSPREY_LINE]

    def test_apply_script_service_name_for_root(self, tmp_path: Path) -> None:
        group_text = "root:x:0:\nwheel:x:10:root\n"
        passwd, group = _write_targets(tmp_path, group=group_text)
        result = _run_script(*_apply_args(0, "osprey-bluesky", passwd, group))
        assert result.returncode == 0, result.stderr
        assert passwd.read_text(encoding="utf-8").splitlines()[0] == (
            "osprey-bluesky:x:0:0:root:/root:/bin/bash"
        )
        assert "wheel:x:10:root,osprey-bluesky" in group.read_text(encoding="utf-8").splitlines()

    def test_apply_script_invalid_name_fails_and_leaves_files(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        result = _run_script(*_apply_args(1000, "Alice", passwd, group))
        _assert_one_line_failure(result)
        assert passwd.read_text(encoding="utf-8") == PASSWD
        assert group.read_text(encoding="utf-8") == GROUP
        assert _leftovers(passwd.parent) == []

    def test_apply_script_missing_canonical_line_fails(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path, passwd=ROOT_LINE + "\n")
        result = _run_script(*_apply_args(1000, "alice", passwd, group))
        _assert_one_line_failure(result)
        assert passwd.read_text(encoding="utf-8") == ROOT_LINE + "\n"
        assert _leftovers(passwd.parent) == []

    def test_apply_script_unknown_uid_fails(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        _assert_one_line_failure(_run_script(*_apply_args(1001, "alice", passwd, group)))
        assert passwd.read_text(encoding="utf-8") == PASSWD

    def test_apply_script_missing_group_file_changes_nothing(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        group.unlink()
        result = _run_script(*_apply_args(1000, "alice", passwd, group))
        _assert_one_line_failure(result)
        assert passwd.read_text(encoding="utf-8") == PASSWD
        assert _leftovers(passwd.parent) == []

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["apply"],
            ["apply", "--uid", "1000"],
            ["apply", "--name", "alice"],
            ["apply", "--uid", "x", "--name", "alice"],
            ["frobnicate", "--uid", "1000", "--name", "alice"],
            ["apply", "--uid", "1000", "--name", "alice", "--bogus"],
            ["apply", "--help"],
            ["--help"],
        ],
    )
    def test_apply_script_bad_arguments_fail_closed(self, argv: list[str]) -> None:
        _assert_one_line_failure(_run_script(*argv))


class TestApplyFunction:
    """In-process ``apply`` and ``main``, including forced write failures."""

    def test_apply_function_writes_files(self, tmp_path: Path) -> None:
        passwd, group = _write_targets(tmp_path)
        apply(1000, "alice", passwd=passwd, group=group)
        assert passwd.read_text(encoding="utf-8") == rewrite(PASSWD, 1000, "alice")
        assert group.read_text(encoding="utf-8") == append_group_members(GROUP, "osprey", "alice")

    def test_apply_function_replace_failure_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passwd, group = _write_targets(tmp_path)

        def boom(*_args: object) -> None:
            raise OSError("forced replace failure")

        monkeypatch.setattr(control_identity.os, "replace", boom)
        with pytest.raises(OSError, match="forced"):
            apply(1000, "alice", passwd=passwd, group=group)
        assert _leftovers(passwd.parent) == []
        assert passwd.read_text(encoding="utf-8") == PASSWD
        assert group.read_text(encoding="utf-8") == GROUP

    def test_apply_function_post_check_failure_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passwd, group = _write_targets(tmp_path)
        monkeypatch.setattr(control_identity, "rewrite", lambda text, uid, name: text)
        with pytest.raises(RuntimeError, match="post-check"):
            apply(1000, "alice", passwd=passwd, group=group)

    def test_apply_function_skips_nss_check_off_etc_passwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passwd, group = _write_targets(tmp_path)
        called: list[int] = []
        monkeypatch.setattr(control_identity, "_nss_name", lambda uid: called.append(uid) or "")
        apply(1000, "alice", passwd=passwd, group=group)
        assert called == []

    def test_main_failure_prints_one_stderr_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        passwd, group = _write_targets(tmp_path)
        assert main(_apply_args(1000, "root", passwd, group)) != 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(captured.err.splitlines()) == 1

    def test_main_success_is_silent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        passwd, group = _write_targets(tmp_path)
        assert main(_apply_args(1000, "alice", passwd, group)) == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_main_exports(self) -> None:
        assert "apply" in control_identity.__all__
        assert "main" in control_identity.__all__
        assert os.path.basename(str(MODULE_PATH)) == "control_identity.py"
