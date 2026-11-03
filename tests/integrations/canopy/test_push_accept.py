"""``push accept`` and ``push reject`` against fake containers.

The harness is ``test_push_show``'s: a fake ``docker`` on PATH that runs the
fixed scripts with this interpreter, a tmp deployment with a three-user roster
(carol is the admin, bob the pusher) and a two-screen library with a fresh
report. A checked stage is either written by a real ``show`` (the cases that
need CANOPY, which the gate installs and refuses to skip) or hand-built from
the module's own staging helpers, so every refusal that comes before the emit
container runs with the standard library alone.

A kill at a boundary is simulated by a ``BaseException`` raised from a seam
of the module: ``main`` catches only ``Refusal`` and ``OSError``, so the state
on disk is what a real kill would leave, and the two locks are released by
their context managers.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import test_push_show as harness
import yaml

from osprey.integrations.canopy import push
from osprey.integrations.canopy.library import (
    REPORT_JSON,
    REPORT_MD,
    judge,
    library_is_fresh,
    scan,
)
from osprey.integrations.canopy.push import container, refusals, verbs
from osprey.integrations.canopy.push import deployment as deployment_module
from osprey_connectors.config import load_project_config

KNOB, OTHER_RID, PNG, RID, USERS = (
    harness.KNOB,
    harness.OTHER_RID,
    harness.PNG,
    harness.RID,
    harness.USERS,
)
_bytes, _doc, _node = harness._bytes, harness._doc, harness._node
library_of, request_doc, run = (
    harness.library_of,
    harness.request_doc,
    harness.run,
)
simple_files, write_bundle, write_report = (
    harness.simple_files,
    harness.write_bundle,
    harness.write_report,
)
# The harness fixtures, bound by name so pytest finds them in this module.
deployment = harness.deployment
fake = harness.fake

GOLDENS = Path(__file__).parent / "fixtures" / "push_accept"
HOST_IDS = (501, 20)
NONCE = "0badf00d"
IMAGE = "proj-readonly:local"
PAYLOAD = "/app/proj-readonly/canopy_payload"
LIBRARY = "data/canopy/library"
LOG_FIELDS = {
    "rid",
    "volume_user",
    "target_dir",
    "reviewer_attested",
    "os_user",
    "sudo_user",
    "digest",
    "decision",
    "time",
    "files",
    "replaced",
}
NEW_KNOB = _doc(
    _node(
        "k",
        "action-button",
        {"actions": [{"type": "write_pv", "pvName": "SR:KNOB:NEW", "value": "1"}]},
    )
)


class Kill(BaseException):
    """A kill at a boundary: ``main`` does not catch it, so the state is left as it was."""


# --- helpers ------------------------------------------------------------------------------------


def _golden(name: str, text: str) -> None:
    path = GOLDENS / name
    if os.environ.get("PUSH_ACCEPT_GOLDENS") == "write":  # pragma: no cover - regeneration only
        path.write_text(text, encoding="utf-8")
    assert path.exists(), (
        f"{path} is absent; regenerate deliberately with PUSH_ACCEPT_GOLDENS=write"
    )
    assert path.read_text(encoding="utf-8") == text, f"update {path} deliberately"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def library_files(library: Path) -> dict[str, bytes]:
    """Every file below ``library`` (the reports included) by its relative path."""
    return {
        p.relative_to(library).as_posix(): p.read_bytes()
        for p in sorted(library.rglob("*"))
        if p.is_file()
    }


def raw_request(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def stage_bundle(
    deployment: Path, files: dict[str, bytes], request: dict | None = None, user: str = "bob"
) -> tuple[Path, verbs.Request]:
    """A checked stage at ``var/canopy-push/<user>/<rid>/`` written as ``show`` writes it, without a container."""
    doc = request_doc(files) if request is None else request
    req = verbs._validate_request(raw_request(doc), doc["rid"], dict(files))
    dep = deployment_module.Deployment(deployment, {}, library_of(deployment), "docker")
    fresh = verbs._stage(dep, user, req, "cafe")
    return verbs._promote_stage(fresh, doc["rid"]), req


def digest_for(deployment: Path, request: verbs.Request) -> str:
    return push.framed_digest(request, judge(scan(library_of(deployment))).digest)


def log_entries(deployment: Path) -> list[dict]:
    log = deployment / "var" / "canopy" / "push-log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def accepted_pairs(deployment: Path) -> list[tuple[str, str]]:
    return [
        (e["volume_user"], e["rid"]) for e in log_entries(deployment) if e["decision"] == "accepted"
    ]


def digest_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("digest: ")]
    assert len(lines) == 1, out
    return lines[0][len("digest: ") :]


def accept(deployment: Path, digest: str, *extra: str, capsys, reviewer: str = "carol", rid=RID):
    return run(
        deployment,
        "accept",
        f"bob/{rid}",
        "--reviewer",
        reviewer,
        "--digest",
        digest,
        *extra,
        capsys=capsys,
    )


def shown_digest(deployment: Path, capsys, rid: str = RID) -> str:
    code, out, err = run(deployment, "show", f"bob/{rid}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    return digest_line(out)


def replaced_dirs(deployment: Path, user: str = "bob") -> list[Path]:
    root = deployment / "var" / "canopy-push" / user / "replaced"
    return sorted(root.iterdir()) if root.is_dir() else []


def real_report_write(fake, library: Path) -> None:
    """``canopy_mcp.report --write`` run for real on the host, as the keep sequence says."""
    env = dict(os.environ)
    env["CANOPY_VIEWER_DIST"] = str(fake.payload / "viewer")
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "canopy_mcp.report",
            str(library),
            "--served-at",
            "library",
            "--write",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr


def real_report_check(fake, library: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CANOPY_VIEWER_DIST"] = str(fake.payload / "viewer")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "canopy_mcp.report",
            str(library),
            "--served-at",
            "library",
            "--check",
        ],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def stamps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Distinct replaced-directory stamps per accept, whatever the wall clock does."""
    made: list[str] = []

    def stamp() -> str:
        made.append(f"20261007T0000{len(made):02d}Z")
        return made[-1]

    monkeypatch.setattr(verbs, "_utc_stamp", stamp)
    return made


# --- argv, script and epilogue goldens --------------------------------------------------------------


def test_accept_emit_argv_golden() -> None:
    argv = push.container_argv(
        "docker",
        "accept-emit",
        nonce=NONCE,
        image=IMAGE,
        script=container.EMIT_SCRIPT,
        args=["/s/would-be"],
        stage="/srv/mca/var/canopy-push/bob/0123456789abcdef",
        payload_dir=PAYLOAD,
        host_ids=HOST_IDS,
    )
    _golden("accept-emit.json", json.dumps(argv, indent=2) + "\n")
    assert argv[argv.index("--user") + 1] == "501:20"
    assert argv[argv.index("--mount") + 1] == (
        "type=bind,src=/srv/mca/var/canopy-push/bob/0123456789abcdef,dst=/s,readonly"
    )
    assert argv.count("--mount") == 1
    assert argv[-5:] == [IMAGE, "-I", "-c", container.EMIT_SCRIPT, "/s/would-be"]


def test_the_emit_script_holds_no_host_path_and_no_git() -> None:
    assert "/Users" not in container.EMIT_SCRIPT and "/home" not in container.EMIT_SCRIPT
    assert "git" not in container.EMIT_SCRIPT
    assert "--emit" in container.EMIT_SCRIPT and "canopy_mcp.report" in container.EMIT_SCRIPT


def _config_document(image_source: str) -> dict:
    return {
        "project_name": "proj",
        "modules": {"web_terminals": harness.roster_document(image_source)},
    }


@pytest.mark.parametrize("branch", ["local", "registry"])
def test_the_epilogue_golden_per_image_source(tmp_path: Path, capsys, branch: str) -> None:
    """One stdout golden per image-source branch, built from a full build/config.yml document."""
    root = tmp_path / "mca"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text(
        yaml.safe_dump(_config_document(branch)), encoding="utf-8"
    )
    config = load_project_config(root / "build" / "config.yml")
    dep = deployment_module.Deployment(
        Path("/srv/mca"), config, Path("/srv/mca") / LIBRARY, "docker"
    )
    files = simple_files()
    request = verbs._validate_request(raw_request(request_doc(files)), RID, files)
    verbs._epilogue(dep, "bob", request, "carol", "f" * 64, has_repo=True)
    out = capsys.readouterr().out
    _golden(f"epilogue-{branch}.txt", out)
    assert f"git -C /srv/mca add -- {LIBRARY}" in out
    assert f"git -C /srv/mca status --ignored -- {LIBRARY}" in out
    message = shlex.quote(f"canopy: accept bob/{RID} (sr/rf), reviewer carol, digest {'f' * 64}")
    assert f"git -C /srv/mca commit -m {message} -- {LIBRARY}" in out
    assert "no repository of its own" not in out
    assert "uncommitted developer-avenue change" in out
    assert "--as-built" in out and "source drift" in out
    if branch == "local":
        assert "run osprey build, then osprey up" in out
        assert "pipeline" not in out
    else:
        assert (
            "push to the deployment remote, let the pipeline build and publish the images, then osprey up"
            in out
        )
        assert "run osprey build" not in out
    for path in ("screens/sr/rf/view.screen.json", "screens/sr/rf/img/logo.png", REPORT_JSON):
        assert f"  {path}" in out
    assert "\r" not in out


def test_the_epilogue_warns_first_when_the_deployment_has_no_repository(capsys) -> None:
    config = _config_document("local")
    dep = deployment_module.Deployment(
        Path("/srv/mca"), config, Path("/srv/mca") / LIBRARY, "docker"
    )
    files = simple_files()
    request = verbs._validate_request(raw_request(request_doc(files)), RID, files)
    verbs._epilogue(dep, "bob", request, "carol", "f" * 64, has_repo=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "this deployment has no repository of its own; put it under version control, "
        "or the replaced panels are kept 14 days only"
    )


# --- common rules ------------------------------------------------------------------------------------


def test_a_non_owner_accept_is_refused(fake, deployment, capsys, monkeypatch) -> None:
    fake.volume("bob")
    other = os.geteuid() + 1
    monkeypatch.setattr(deployment_module, "_effective_uid", lambda: other)
    code, _out, err = accept(deployment, "0" * 64, capsys=capsys)
    assert code == refusals.EXIT_NOT_OWNER
    assert "run as the deployment's owner" in err
    assert fake.runs("accept-emit") == [] and log_entries(deployment) == []


def test_a_uid_zero_accept_is_refused_before_anything_else(
    fake, deployment, capsys, monkeypatch
) -> None:
    fake.volume("bob")
    monkeypatch.setattr(deployment_module, "_effective_uid", lambda: 0)
    code, _out, err = accept(deployment, "0" * 64, capsys=capsys)
    assert code == refusals.EXIT_ROOT_UID
    assert "instead of using sudo" in err
    assert fake.calls() == [] and log_entries(deployment) == []


@pytest.mark.usefixtures("fake")
def test_a_uid_zero_reject_is_refused_too(deployment, capsys, monkeypatch) -> None:
    monkeypatch.setattr(deployment_module, "_effective_uid", lambda: 0)
    code = run(
        deployment, "reject", f"bob/{RID}", "--reviewer", "carol", "--reason", "no", capsys=capsys
    )[0]
    assert code == refusals.EXIT_ROOT_UID


@pytest.mark.parametrize("reviewer", ["bob", "alice", "mallory", "Carol", "carol\n", ""])
def test_the_reviewer_must_be_a_roster_user_whose_render_edits_the_deployment(
    fake, deployment, capsys, reviewer
) -> None:
    fake.volume("bob")
    files = simple_files()
    _stage, request = stage_bundle(deployment, files)
    before = library_files(library_of(deployment))
    code, _out, err = accept(
        deployment, digest_for(deployment, request), capsys=capsys, reviewer=reviewer
    )
    assert code == refusals.EXIT_REVIEWER
    assert "holds" in err and verbs.REVIEWER_PRIVILEGE in err
    code = run(
        deployment, "reject", f"bob/{RID}", "--reviewer", reviewer, "--reason", "x", capsys=capsys
    )[0]
    assert code == refusals.EXIT_REVIEWER
    assert library_files(library_of(deployment)) == before
    assert log_entries(deployment) == [] and fake.runs("accept-emit") == []


def _retier(deployment: Path, tiers: dict[str, bool], users: dict[str, str]) -> None:
    """Rewrite the fixture deployment's roster and renders for ``tiers``."""
    config_path = deployment / "build" / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    roster = harness.roster_document("local", tiers)
    roster["users"] = [
        {"name": name, "index": at, "tier": tier} for at, (name, tier) in enumerate(users.items())
    ]
    config["modules"]["web_terminals"] = roster
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    harness.write_renders(deployment, tiers)


def _reviewer_check(deployment: Path, reviewer: str) -> None:
    config = load_project_config(deployment / "build" / "config.yml")
    dep = deployment_module.Deployment(deployment, config, library_of(deployment), "docker")
    verbs._check_reviewer(dep, reviewer)


def test_a_renamed_admin_tier_may_review(deployment) -> None:
    """The check follows the capability, not the tier's name: a tier called
    ``maintainers`` whose render holds the setup tool reviews."""
    _retier(
        deployment,
        {"readonly": False, "maintainers": True},
        {"alice": "readonly", "bob": "readonly", "carol": "maintainers"},
    )
    _reviewer_check(deployment, "carol")
    with pytest.raises(push.Refusal) as refused:
        _reviewer_check(deployment, "bob")
    assert refused.value.code == refusals.EXIT_REVIEWER


def test_a_tier_named_admin_without_the_privilege_may_not_review(deployment) -> None:
    """A render that denies the setup tool is refused, whatever its tier is named."""
    _retier(
        deployment,
        {"readonly": False, "admin": False},
        {"alice": "readonly", "bob": "readonly", "carol": "admin"},
    )
    with pytest.raises(push.Refusal) as refused:
        _reviewer_check(deployment, "carol")
    assert refused.value.code == refusals.EXIT_REVIEWER


def test_an_unrendered_reviewer_render_holds_nothing(deployment) -> None:
    """No rendered config to read is no privilege: the check fails closed."""
    (deployment / "build" / "proj-admin" / "config.yml").unlink()
    with pytest.raises(push.Refusal) as refused:
        _reviewer_check(deployment, "carol")
    assert refused.value.code == refusals.EXIT_REVIEWER


def test_accept_is_a_verb_held_under_the_push_lock(fake, deployment, capsys) -> None:
    fake.volume("bob")
    lock = deployment / "var" / "canopy" / "push.lock"
    lock.parent.mkdir(parents=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        code, _out, err = accept(deployment, "0" * 64, capsys=capsys)
    finally:
        os.close(fd)
    assert code == refusals.EXIT_BUSY and "another push review holds" in err


def test_accept_without_a_checked_stage_says_run_show_first(fake, deployment, capsys) -> None:
    fake.volume("bob")
    code, _out, err = accept(deployment, "0" * 64, capsys=capsys)
    assert code == refusals.EXIT_NO_STAGE and "run show first" in err
    fresh = deployment / "var" / "canopy-push" / "bob" / f"{RID}.abcd"
    (fresh / "overlay").mkdir(parents=True)
    (fresh / "would-be").mkdir()
    (fresh / "request.json").write_bytes(b"{}")
    code, _out, _err = accept(deployment, "0" * 64, capsys=capsys)
    assert code == refusals.EXIT_NO_STAGE, "an unchecked <rid>.<nonce> stage is never read"
    assert log_entries(deployment) == []


def test_a_stale_library_refuses_accept_before_the_stage_is_read(fake, deployment, capsys) -> None:
    fake.volume("bob")
    _stage, request = stage_bundle(deployment, simple_files())
    digest = digest_for(deployment, request)
    library = library_of(deployment)
    (library / "screens" / "gtb" / "x.screen.json").write_bytes(_bytes(_doc(title="changed")))
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_STALE
    assert "changed: screens/gtb/x.screen.json" in err and "canopy_mcp.report" in err
    assert log_entries(deployment) == []


def test_a_digest_that_is_not_the_one_show_printed_is_refused(fake, deployment, capsys) -> None:
    fake.volume("bob")
    _stage, request = stage_bundle(deployment, simple_files())
    right = digest_for(deployment, request)
    before = library_files(library_of(deployment))
    for wrong in (
        "0" * 64,
        right[:-1] + ("0" if right[-1] != "0" else "1"),
        "short",
        right.upper(),
    ):
        code, _out, err = accept(deployment, wrong, capsys=capsys)
        assert code == refusals.EXIT_DIGEST_MISMATCH, wrong
        assert "digest" in err
    assert library_files(library_of(deployment)) == before
    assert log_entries(deployment) == [] and fake.runs("accept-emit") == []


def test_the_library_change_pair_digest_code_against_freshness_code(
    fake, deployment, capsys
) -> None:
    """A library change with a regenerated report: the digest code. Without one: the freshness code."""
    fake.volume("bob")
    _stage, request = stage_bundle(deployment, simple_files())
    digest = digest_for(deployment, request)
    library = library_of(deployment)
    (library / "screens" / "gtb" / "x.screen.json").write_bytes(_bytes(_doc(title="changed")))
    write_report(library)
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_DIGEST_MISMATCH
    assert "show" in err, "the refusal says to run show again"
    (library / "screens" / "gtb" / "x.screen.json").write_bytes(_bytes(_doc(title="again")))
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_STALE
    assert "changed: screens/gtb/x.screen.json" in err
    assert log_entries(deployment) == [] and fake.runs("accept-emit") == []


def test_a_collision_needs_replace(fake, deployment, capsys) -> None:
    fake.volume("bob")
    files = {"KnobPanel.screen.json": _bytes(NEW_KNOB)}
    doc = request_doc(files, source_id="x/KnobPanel", source_dir="x", target_dir="common")
    _stage, request = stage_bundle(deployment, files, doc)
    before = library_files(library_of(deployment))
    code, _out, err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_COLLISION
    assert "--replace" in err and "screens/common/KnobPanel.screen.json" in err
    assert library_files(library_of(deployment)) == before
    assert log_entries(deployment) == [] and fake.runs("accept-emit") == []
    assert replaced_dirs(deployment) == []


def test_identical_bytes_are_not_a_collision(fake, deployment, capsys, monkeypatch) -> None:
    """A destination whose existing bytes equal the overlay's passes step 4 without ``--replace``."""
    fake.volume("bob")
    files = {"KnobPanel.screen.json": _bytes(KNOB)}
    doc = request_doc(files, source_id="x/KnobPanel", source_dir="x", target_dir="common")
    _stage, request = stage_bundle(deployment, files, doc)
    reached: list[str] = []

    def stop(*_args, **_kwargs):
        reached.append("emit")
        raise push.Refusal(refusals.EXIT_CONTAINER_FAILED, "stopped by the test at the emit")

    monkeypatch.setattr(verbs, "_emit_report", stop)
    code, _out, _err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_CONTAINER_FAILED and reached == ["emit"]


@pytest.mark.parametrize(
    ("files", "overrides", "code", "spelled"),
    [
        (
            {"y.screen.json": _bytes(_doc())},
            {"source_id": "x/y", "source_dir": "x", "target_dir": "GTB"},
            refusals.EXIT_CASE_DIR,
            "'screens/gtb'",
        ),
        (
            {"knobpanel.screen.json": _bytes(KNOB)},
            {"source_id": "x/knobpanel", "source_dir": "x", "target_dir": "common"},
            refusals.EXIT_CASE_FILE,
            "'screens/common/KnobPanel.screen.json'",
        ),
    ],
)
def test_the_case_only_comparison_is_repeated_against_the_current_listing(
    fake, deployment, capsys, files, overrides, code, spelled
) -> None:
    """A hand-built stage ``show`` would have refused: ``accept`` refuses it on its own."""
    fake.volume("bob")
    _stage, request = stage_bundle(deployment, files, request_doc(files, **overrides))
    got, _out, err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert got == code
    assert spelled in err
    assert log_entries(deployment) == [] and fake.runs("accept-emit") == []


def test_a_stage_whose_overlay_leaves_the_target_directory_is_refused(
    fake, deployment, capsys
) -> None:
    fake.volume("bob")
    stage, request = stage_bundle(deployment, simple_files())
    stray = stage / "overlay" / "screens" / "other" / "x.png"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(PNG)
    code, _out, err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_BAD_STAGE
    assert "screens/other/x.png" in err
    assert log_entries(deployment) == []


def test_accept_holds_an_exclusive_lock_on_the_library_while_it_emits(
    fake, deployment, capsys, monkeypatch
) -> None:
    fake.volume("bob")
    _stage, request = stage_bundle(deployment, simple_files())
    library = library_of(deployment)
    seen: dict[str, bool] = {}

    def probe(*_args, **_kwargs):
        fd = os.open(library, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            seen["blocked"] = True
        else:
            seen["blocked"] = False
        finally:
            os.close(fd)
        raise push.Refusal(refusals.EXIT_CONTAINER_FAILED, "stopped by the test at the emit")

    monkeypatch.setattr(verbs, "_emit_report", probe)
    code, _out, _err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_CONTAINER_FAILED
    assert seen == {"blocked": True}, "a shared lock is refused while accept holds the library"
    fd = os.open(library, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_the_freshness_refusal_prints_both_recovery_sequences_and_the_logged_accepts(
    fake, deployment, capsys
) -> None:
    fake.volume("bob")
    library = library_of(deployment)
    landed = _bytes(_doc(title="landed"))
    (library / "screens" / "sr").mkdir()
    (library / "screens" / "sr" / "new.screen.json").write_bytes(landed)
    log = deployment / "var" / "canopy" / "push-log.jsonl"
    log.parent.mkdir(parents=True)
    entries = [
        {
            "rid": RID,
            "volume_user": "bob",
            "decision": "accepted",
            "files": [{"path": "screens/sr/new.screen.json", "sha256": _sha(landed)}],
        },
        {
            "rid": OTHER_RID,
            "volume_user": "bob",
            "decision": "accepted",
            "files": [{"path": "screens/sr/gone.screen.json", "sha256": "0" * 64}],
        },
        {"rid": "1" * 16, "volume_user": "bob", "decision": "rejected", "reason": "no"},
    ]
    log.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    for verb in ("show", "accept"):
        if verb == "show":
            code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
        else:
            code, _out, err = accept(deployment, "0" * 64, capsys=capsys)
        assert code == refusals.EXIT_STALE, verb
        assert "added: screens/sr/new.screen.json" in err
        assert "discard" in err and "keep" in err
        assert ".tmp" in err and "replaced" in err
        blanket = (
            f"git -C {deployment} restore --source=HEAD --staged --worktree -- {LIBRARY} && "
            f"git -C {deployment} clean -fd -- {LIBRARY}"
        )
        assert blanket in err
        lines = err.splitlines()
        at = next(i for i, line in enumerate(lines) if blanket in line)
        assert "discards every uncommitted library change" in lines[at + 1]
        assert f"bob/{RID}" in lines[at + 1] and f"bob/{OTHER_RID}" not in lines[at + 1]
        assert "canopy_mcp.report" in err and "show" in err
        first = next(i for i, line in enumerate(lines) if "discard:" in line)
        assert first < at, "the blanket is printed second"


def test_a_write_failure_after_the_log_append_has_its_own_code(
    fake, deployment, capsys, monkeypatch, stamps
) -> None:
    """The attestation is logged before any byte; a failing write then names the logged pair."""
    fake.volume("bob")
    _stage, request = stage_bundle(deployment, simple_files())
    library = library_of(deployment)
    judged = judge(scan(library))
    report = {
        "generator": {"schema": "canopy-library-report/1", "canopy_version": "0"},
        "files": {"digest": verbs._overlay_digest(judged, request), "list": []},
    }
    monkeypatch.setattr(verbs, "_emit_report", lambda *_a, **_k: (report, "# md\n"))

    def broken(_root, _rel, _data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(verbs, "_place_file", broken)
    before = library_files(library)
    code, _out, err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_WRITE_FAILED
    assert "No space left on device" in err and f"bob/{RID}" in err and "recovery" in err
    assert accepted_pairs(deployment) == [("bob", RID)]
    assert library_files(library) == before and library_is_fresh(library)
    [replaced] = replaced_dirs(deployment)
    assert replaced.name == f"{RID}.{stamps[0]}"
    assert log_entries(deployment)[0]["replaced_dir"] == str(replaced)


# --- reject ------------------------------------------------------------------------------------------


def test_reject_logs_the_reason_and_touches_nothing_else(
    fake, deployment, capsys, monkeypatch
) -> None:
    fake.volume("bob")
    stage, _request = stage_bundle(deployment, simple_files())
    replaced = (
        deployment / "var" / "canopy-push" / "bob" / "replaced" / f"{OTHER_RID}.20260101T000000Z"
    )
    replaced.mkdir(parents=True)
    then = time.time() - 15 * 24 * 3600
    os.utime(replaced, (then, then))
    monkeypatch.setenv("SUDO_USER", "")
    before = library_files(library_of(deployment))
    code, out, err = run(
        deployment,
        "reject",
        f"bob/{RID}",
        "--reviewer",
        "carol",
        "--reason",
        "wrong targeté",
        capsys=capsys,
    )
    assert code == refusals.EXIT_OK, err
    assert f"rejected bob/{RID}" in out and "é" not in out
    entries = log_entries(deployment)
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == LOG_FIELDS | {"reason"}
    assert entry["decision"] == "rejected" and entry["reason"] == "wrong targeté"
    assert entry["rid"] == RID and entry["volume_user"] == "bob"
    assert entry["reviewer_attested"] == "carol" and entry["digest"] is None
    assert entry["target_dir"] == "sr/rf" and entry["replaced"] == []
    assert {f["path"] for f in entry["files"]} == {
        "screens/sr/rf/view.screen.json",
        "screens/sr/rf/parts/inner.screen.json",
        "screens/sr/rf/img/logo.png",
    }
    assert isinstance(entry["os_user"], str) and entry["os_user"]
    assert entry["time"].endswith("Z")
    assert stage.is_dir(), "reject removes no stage; the next show prunes it"
    assert replaced.is_dir(), "reject prunes no replaced-bytes directory"
    assert library_files(library_of(deployment)) == before
    assert fake.runs("accept-emit") == []


def test_reject_without_a_stage_logs_what_it_knows(fake, deployment, capsys) -> None:
    fake.volume("bob")
    code, _out, err = run(
        deployment,
        "reject",
        f"bob/{RID}",
        "--reviewer",
        "carol",
        "--reason",
        "evicted",
        capsys=capsys,
    )
    assert code == refusals.EXIT_OK, err
    [entry] = log_entries(deployment)
    assert entry["target_dir"] is None and entry["files"] == []


@pytest.mark.parametrize("reason", ["", "   ", "bad ‮ reason", "x" * 2001])
def test_a_bad_reason_has_its_own_code(fake, deployment, capsys, reason) -> None:
    fake.volume("bob")
    code = run(
        deployment, "reject", f"bob/{RID}", "--reviewer", "carol", "--reason", reason, capsys=capsys
    )[0]
    assert code == refusals.EXIT_BAD_REASON
    assert log_entries(deployment) == []


# --- prune ------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fake")
def test_replaced_directories_are_pruned_by_age_only(deployment) -> None:
    dep = deployment_module.Deployment(deployment, {}, library_of(deployment), "docker")
    bob = deployment / "var" / "canopy-push" / "bob"
    old = bob / "replaced" / f"{RID}.20260901T000000Z"
    recent = bob / "replaced" / f"{OTHER_RID}.20261007T000000Z"
    odd = bob / "replaced" / "not-a-rid"
    stage = bob / RID
    fresh = bob / f"{OTHER_RID}.abcd"
    for path in (old, recent, odd, stage, fresh):
        path.mkdir(parents=True)
    then = time.time() - 15 * 24 * 3600
    for path in (old, odd, stage, fresh):
        os.utime(path, (then, then))
    verbs._prune_replaced(dep)
    assert not old.exists()
    assert recent.exists() and odd.exists()
    assert stage.exists() and fresh.exists(), "the replaced prune touches no stage"
    verbs._prune_replaced(dep, now=time.time() + 15 * 24 * 3600)
    assert not recent.exists()


# --- the emit container and the writes (CANOPY) ----------------------------------------------------------


@pytest.mark.canopy
def test_accept_writes_exactly_the_pushed_files_and_the_report(
    fake, deployment, capsys, stamps, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    provenance = {"snapshot": "p2c-cycle4", "overview": "sr/overview", "wheels": ["canopy-0.0"]}
    write_report(library, provenance=provenance)
    monkeypatch.setenv("SUDO_USER", "thorsten")
    before = library_files(library)
    digest = shown_digest(deployment, capsys)
    stage = deployment / "var" / "canopy-push" / "bob" / RID
    notes = json.loads((stage / verbs.CHECK_NOTES_NAME).read_text(encoding="ascii"))
    code, out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_OK, err
    after = library_files(library)
    pushed = {
        "screens/sr/rf/view.screen.json",
        "screens/sr/rf/parts/inner.screen.json",
        "screens/sr/rf/img/logo.png",
    }
    assert set(after) - set(before) == pushed
    assert set(before) - set(after) == set()
    assert {p for p in before if before[p] != after[p]} == {REPORT_JSON, REPORT_MD}
    assert after["screens/sr/rf/img/logo.png"] == PNG
    assert not [p for p in after if ".tmp" in p]
    report = json.loads(after[REPORT_JSON])
    assert report["provenance"] == provenance
    assert list(report)[-1] == "provenance"
    assert report["files"]["digest"] == judge(scan(library)).digest
    assert after[REPORT_JSON] == (json.dumps(report, indent=2, ensure_ascii=True) + "\n").encode()
    assert after[REPORT_MD].isascii() and after[REPORT_MD] != b"# forged\n"
    assert library_is_fresh(library)
    for path in (
        *pushed,
        REPORT_JSON,
        REPORT_MD,
        "screens/sr",
        "screens/sr/rf",
        "screens/sr/rf/img",
    ):
        mode = stat.S_IMODE((library / path).lstat().st_mode)
        assert mode == (0o755 if (library / path).is_dir() else 0o644), path
    check = real_report_check(fake, library)
    assert check.returncode == 0, check.stdout + check.stderr
    [entry] = log_entries(deployment)
    assert set(entry) == LOG_FIELDS | {"replaced_dir", "spec_unavailable"}
    assert notes["specUnavailable"], "the fake payload stages no spec"
    assert entry["spec_unavailable"] == verbs.spec_unavailable_record(notes["specUnavailable"])
    spec_lines = [line for line in out.splitlines() if line.startswith("spec_unavailable:")]
    assert spec_lines == [verbs.spec_unavailable_line(notes["specUnavailable"])]
    assert entry["decision"] == "accepted" and entry["digest"] == digest
    assert entry["rid"] == RID and entry["volume_user"] == "bob" and entry["target_dir"] == "sr/rf"
    assert entry["reviewer_attested"] == "carol" and entry["sudo_user"] == "thorsten"
    assert entry["files"] == sorted(
        ({"path": p, "sha256": _sha(after[p])} for p in pushed), key=lambda f: f["path"]
    )
    assert entry["replaced"] == [REPORT_JSON, REPORT_MD]
    [replaced] = replaced_dirs(deployment)
    assert entry["replaced_dir"] == str(replaced) and replaced.name == f"{RID}.{stamps[0]}"
    assert (replaced / REPORT_JSON).read_bytes() == before[REPORT_JSON]
    assert (replaced / REPORT_MD).read_bytes() == before[REPORT_MD]
    assert not (replaced / "screens").exists()
    assert "no repository of its own" in out
    assert f"git -C {deployment} add -- {LIBRARY}" in out
    message = shlex.quote(f"canopy: accept bob/{RID} (sr/rf), reviewer carol, digest {digest}")
    assert f"git -C {deployment} commit -m {message} -- {LIBRARY}" in out
    assert "run osprey build, then osprey up" in out
    assert "\r" not in out
    assert len(fake.runs("accept-emit")) == 1
    emit = fake.runs("accept-emit")[0]
    assert emit[emit.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    stage = deployment / "var" / "canopy-push" / "bob" / RID
    assert emit[emit.index("--mount") + 1] == f"type=bind,src={stage},dst=/s,readonly"
    assert stage.is_dir(), "accept leaves the checked stage; show's prune takes it in 14 days"


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_with_a_repository_the_no_repository_line_is_absent(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    (deployment / ".git").mkdir()
    digest = shown_digest(deployment, capsys)
    code, out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert "no repository of its own" not in out


@pytest.mark.canopy
def test_an_emit_container_failure_leaves_the_library_and_the_log_untouched(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    digest = shown_digest(deployment, capsys)
    before = library_files(library)
    monkeypatch.setenv("FAKE_RUNTIME_FAIL", "accept-emit:9")
    assert accept(deployment, digest, capsys=capsys)[0] == refusals.EXIT_CONTAINER_FAILED
    monkeypatch.setenv("FAKE_RUNTIME_FAIL", "accept-emit:126")
    assert accept(deployment, digest, capsys=capsys)[0] == refusals.EXIT_CONTAINER_START
    assert library_files(library) == before
    assert log_entries(deployment) == [] and replaced_dirs(deployment) == []


@pytest.mark.canopy
def test_an_emitted_report_that_does_not_cover_the_overlay_is_refused(
    fake, deployment, capsys
) -> None:
    """The stage's would-be copy is not trusted: its emitted ``files.digest`` must equal listing + overlay."""
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    digest = shown_digest(deployment, capsys)
    stage = deployment / "var" / "canopy-push" / "bob" / RID
    (stage / "would-be" / "screens" / "sr" / "extra.png").write_bytes(PNG)
    before = library_files(library)
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_EMIT_DIGEST
    assert "overlay" in err
    assert library_files(library) == before
    assert log_entries(deployment) == [] and replaced_dirs(deployment) == []


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_a_replaced_directory_that_already_exists_is_refused_before_the_log(
    fake, deployment, capsys
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    digest = shown_digest(deployment, capsys)
    taken = deployment / "var" / "canopy-push" / "bob" / "replaced" / f"{RID}.20261007T000000Z"
    taken.mkdir(parents=True)
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_REPLACED_EXISTS
    assert log_entries(deployment) == []
    assert library_is_fresh(library_of(deployment))
    assert "screens/sr" not in library_files(library_of(deployment))
    assert replaced_dirs(deployment) == [taken]


# --- the interrupted-accept table, one case per row ------------------------------------------------------


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_killed_after_the_log_append_leaves_the_library_fresh_and_both_pass(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    before = library_files(library)
    digest = shown_digest(deployment, capsys)

    real = verbs._copy_replaced

    def killed(*_args, **_kwargs):
        raise Kill

    monkeypatch.setattr(verbs, "_copy_replaced", killed)
    with pytest.raises(Kill):
        accept(deployment, digest, capsys=capsys)
    capsys.readouterr()
    assert library_files(library) == before, "no byte landed"
    assert library_is_fresh(library)
    assert accepted_pairs(deployment) == [("bob", RID)]
    monkeypatch.setattr(verbs, "_copy_replaced", real)
    assert shown_digest(deployment, capsys) == digest
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert accepted_pairs(deployment) == [("bob", RID), ("bob", RID)]
    assert len(replaced_dirs(deployment)) == 2


def _kill_place_file_after(monkeypatch, count: int, *, when_rel: str | None = None) -> list[str]:
    """Kill after ``count`` library writes, or just before the write of ``when_rel``; once only."""
    real = verbs._place_file
    placed: list[str] = []
    armed = [True]

    def seam(root, rel, data):
        if armed[0] and when_rel is not None and rel == when_rel:
            armed[0] = False
            raise Kill
        real(root, rel, data)
        if not armed[0]:
            return
        placed.append(rel)
        if when_rel is None and len(placed) >= count:
            armed[0] = False
            raise Kill

    monkeypatch.setattr(verbs, "_place_file", seam)
    return placed


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_killed_during_the_file_writes_refuses_both_and_the_keep_sequence_completes(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    digest = shown_digest(deployment, capsys)
    placed = _kill_place_file_after(monkeypatch, 2)
    with pytest.raises(Kill):
        accept(deployment, digest, capsys=capsys)
    capsys.readouterr()
    assert len(placed) == 2 and all(p.startswith("screens/sr/rf/") for p in placed)
    assert accepted_pairs(deployment) == [("bob", RID)]
    for verb in ("show", "accept"):
        if verb == "show":
            code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
        else:
            code, _out, err = accept(deployment, digest, capsys=capsys)
        assert code == refusals.EXIT_STALE, verb
        for rel in placed:
            assert f"added: {rel}" in err
        assert "discard" in err and "keep" in err and f"bob/{RID}" in err
    # keep: regenerate the report for real, then show (new digest) and accept without --replace
    real_report_write(fake, library)
    assert library_is_fresh(library)
    new_digest = shown_digest(deployment, capsys)
    assert new_digest != digest, "the library changed, so the digest did"
    code, _out, err = accept(deployment, new_digest, capsys=capsys)
    assert code == refusals.EXIT_OK, err
    after = library_files(library)
    for rel in (
        "screens/sr/rf/view.screen.json",
        "screens/sr/rf/parts/inner.screen.json",
        "screens/sr/rf/img/logo.png",
    ):
        assert rel in after
    assert library_is_fresh(library)
    assert real_report_check(fake, library).returncode == 0
    assert accepted_pairs(deployment) == [("bob", RID), ("bob", RID)]
    assert log_entries(deployment)[-1]["replaced"] == [*sorted(placed), REPORT_JSON, REPORT_MD]


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_killed_between_the_two_report_files_refuses_both(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    before = library_files(library)
    digest = shown_digest(deployment, capsys)
    placed = _kill_place_file_after(monkeypatch, 0, when_rel=REPORT_JSON)
    with pytest.raises(Kill):
        accept(deployment, digest, capsys=capsys)
    capsys.readouterr()
    after = library_files(library)
    assert placed[-1] == REPORT_MD and after[REPORT_MD] != before[REPORT_MD]
    assert after[REPORT_JSON] == before[REPORT_JSON], "the JSON is the commit point"
    for verb in ("show", "accept"):
        if verb == "show":
            code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
        else:
            code, _out, err = accept(deployment, digest, capsys=capsys)
        assert code == refusals.EXIT_STALE, verb
        assert "added: screens/sr/rf/view.screen.json" in err


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_killed_after_the_json_report_is_complete_and_a_repeat_needs_no_replace(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    digest = shown_digest(deployment, capsys)

    real = verbs._epilogue

    def killed(*_args, **_kwargs):
        raise Kill

    monkeypatch.setattr(verbs, "_epilogue", killed)
    with pytest.raises(Kill):
        accept(deployment, digest, capsys=capsys)
    capsys.readouterr()
    assert library_is_fresh(library)
    complete = library_files(library)
    assert "screens/sr/rf/img/logo.png" in complete
    monkeypatch.setattr(verbs, "_epilogue", real)
    new_digest = shown_digest(deployment, capsys)
    assert new_digest != digest
    code, out, err = accept(deployment, new_digest, capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert "COLLID" not in out
    assert library_files(library) == complete, "identical bytes land as identical bytes"
    assert accepted_pairs(deployment) == [("bob", RID), ("bob", RID)]


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_replaced_bytes_are_kept_and_the_discard_sequence_recovers(
    fake, deployment, capsys, monkeypatch
) -> None:
    """A ``--replace`` accept killed during the writes: discard from the replaced-bytes directory, then redo."""
    bob = fake.volume("bob")
    files = {"KnobPanel.screen.json": _bytes(NEW_KNOB)}
    write_bundle(
        bob,
        files,
        request=request_doc(files, source_id="x/KnobPanel", source_dir="x", target_dir="common"),
    )
    library = library_of(deployment)
    original = (library / "screens" / "common" / "KnobPanel.screen.json").read_bytes()
    digest = shown_digest(deployment, capsys)
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_COLLISION
    placed = _kill_place_file_after(monkeypatch, 1)
    with pytest.raises(Kill):
        accept(deployment, digest, "--replace", capsys=capsys)
    capsys.readouterr()
    assert placed == ["screens/common/KnobPanel.screen.json"]
    assert (library / "screens" / "common" / "KnobPanel.screen.json").read_bytes() == _bytes(
        NEW_KNOB
    )
    [entry] = log_entries(deployment)
    replaced = Path(entry["replaced_dir"])
    assert replaced.is_dir() and replaced.parent.name == "replaced"
    assert (replaced / "screens" / "common" / "KnobPanel.screen.json").read_bytes() == original
    assert entry["replaced"] == ["screens/common/KnobPanel.screen.json", REPORT_JSON, REPORT_MD]
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_STALE
    assert "changed: screens/common/KnobPanel.screen.json" in err and f"bob/{RID}" in err
    # discard: restore the changed path from the replaced-bytes directory the log entry names
    for path in sorted(replaced.rglob("*")):
        if path.is_file() and path.relative_to(replaced).as_posix().startswith("screens/"):
            (library / path.relative_to(replaced)).write_bytes(path.read_bytes())
    for leftover in library.rglob(".*.tmp"):
        leftover.unlink()
    assert library_is_fresh(library)
    assert shown_digest(deployment, capsys) == digest, "the library is back, so the digest is"
    code, _out, err = accept(deployment, digest, "--replace", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert (library / "screens" / "common" / "KnobPanel.screen.json").read_bytes() == _bytes(
        NEW_KNOB
    )
    assert accepted_pairs(deployment) == [("bob", RID), ("bob", RID)]
    assert len(replaced_dirs(deployment)) == 2
    assert (replaced / "screens" / "common" / "KnobPanel.screen.json").read_bytes() == original
    # replaced bytes survive a later show
    write_bundle(bob, simple_files(), rid=OTHER_RID)
    shown_digest(deployment, capsys, rid=OTHER_RID)
    assert len(replaced_dirs(deployment)) == 2
    assert (replaced / "screens" / "common" / "KnobPanel.screen.json").read_bytes() == original


@pytest.mark.canopy
def test_accept_prunes_replaced_directories_older_than_14_days_after_its_last_step(
    fake, deployment, capsys, stamps
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    root = deployment / "var" / "canopy-push" / "bob" / "replaced"
    old = root / f"{OTHER_RID}.20260901T000000Z"
    recent = root / f"{OTHER_RID}.20261006T000000Z"
    old.mkdir(parents=True)
    recent.mkdir()
    then = time.time() - 15 * 24 * 3600
    os.utime(old, (then, then))
    digest = shown_digest(deployment, capsys)
    assert old.is_dir(), "show prunes no replaced-bytes directory"
    code, _out, err = accept(deployment, digest, capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert not old.exists()
    assert recent.is_dir()
    assert [p.name for p in replaced_dirs(deployment)] == sorted(
        [recent.name, f"{RID}.{stamps[0]}"]
    )


# --- spec_unavailable ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [b"not json", b'{"specUnavailable": "x"}', b"[]", b"\xff"])
def test_a_malformed_check_notes_file_refuses_the_stage(fake, deployment, capsys, body) -> None:
    fake.volume("bob")
    stage, request = stage_bundle(deployment, simple_files())
    (stage / verbs.CHECK_NOTES_NAME).write_bytes(body)
    code, _out, err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_BAD_STAGE, err
    assert verbs.CHECK_NOTES_NAME in err
    assert log_entries(deployment) == [] and fake.runs("accept-emit") == []


@pytest.mark.canopy
@pytest.mark.usefixtures("stamps")
def test_a_stage_without_check_notes_logs_spec_unavailable_null(fake, deployment, capsys) -> None:
    """A stage written before the notes existed reads as no notes: no line, ``null`` logged."""
    fake.volume("bob")
    write_report(library_of(deployment))
    stage, request = stage_bundle(deployment, simple_files())
    assert not (stage / verbs.CHECK_NOTES_NAME).exists()
    code, out, err = accept(deployment, digest_for(deployment, request), capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert "spec_unavailable" not in out
    [entry] = log_entries(deployment)
    assert entry["spec_unavailable"] is None
