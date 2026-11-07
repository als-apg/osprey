"""``push list`` and ``push show`` against fake containers.

The fake ``docker`` on PATH records every argv, answers the runtime probes and
``volume inspect``, and runs ``docker run`` by executing the fixed script with
this interpreter, with ``/v`` and ``/s`` mapped to host directories. The fixed
scripts therefore run for real; what is faked is the container around them.
Cases that run the CANOPY check script need CANOPY installed and skip without
it (the gate installs the pinned export and refuses a skip); every refusal
before the check container runs with the standard library alone.
"""

from __future__ import annotations

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
import yaml

from osprey.deployment import runtime_helper
from osprey.integrations.canopy import push
from osprey.integrations.canopy.library import judge, scan
from osprey.integrations.canopy.push import container, grammar, refusals, verbs
from osprey.integrations.canopy.push import deployment as deployment_module

RID = "0123456789abcdef"
OTHER_RID = "fedcba9876543210"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
USERS = {"alice": "readwrite", "bob": "readonly", "carol": "admin"}
#: The tiers of the fixture roster, and whether each render holds the
#: deployment-editing setup tool: the admin tier does, the others deny it.
TIERS = {"readonly": False, "readwrite": False, "admin": True}


def roster_document(image_source: str, tiers: dict[str, bool] = TIERS) -> dict:
    """The ``modules.web_terminals`` block of the fixture deployment's config."""
    from tests._renders import tier_render

    return {
        "enabled": True,
        "image_source": image_source,
        "default_tier": "readonly",
        "users": [
            {"name": name, "index": at, "tier": tier}
            for at, (name, tier) in enumerate(USERS.items())
        ],
        "renders": {
            tier: tier_render(
                tier,
                f"proj-{tier}",
                project_path=f"build/proj-{tier}",
                build_profile=f"tiers/{tier}.yml",
            )
            for tier in tiers
        },
    }


def write_renders(root: Path, tiers: dict[str, bool] = TIERS) -> None:
    """Write each tier's rendered ``config.yml``; a tier without the setup tool denies it."""
    from osprey.cli.profile_conventions import SETUP_PATCH_TOOL

    for tier, holds_setup in tiers.items():
        project = root / "build" / f"proj-{tier}"
        project.mkdir(parents=True, exist_ok=True)
        deny = [] if holds_setup else [SETUP_PATCH_TOOL]
        document = {"project_name": f"proj-{tier}", "claude_code": {"permissions": {"deny": deny}}}
        (project / "config.yml").write_text(yaml.safe_dump(document), encoding="utf-8")


#: The fake runtime: a stdlib-only script, run with ``-S`` and its inner interpreter
#: without the ``COVERAGE_*`` variables, so neither starts coverage. Under the
#: subprocess patch (pyproject's ``[tool.coverage.run]``) each start costs ~0.2 s and
#: measures nothing: the fake is test code, and the in-container scripts are ``-c``
#: strings and ``canopy_mcp``, outside the measured source.
FAKE_DOCKER = f"""#!{sys.executable} -S
import json, os, subprocess, sys, time
argv = sys.argv[1:]
with open(os.environ["FAKE_RUNTIME_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\\n")
if argv[:2] == ["compose", "version"] or argv[:1] == ["ps"] or argv[:1] == ["kill"]:
    sys.exit(0)
if argv[:2] == ["volume", "inspect"]:
    sys.exit(0 if argv[2] in open(os.environ["FAKE_VOLUMES"]).read().split() else 1)
if argv[:1] != ["run"]:
    sys.exit(2)
mounts, env, at = {{}}, {{}}, 1
pairs = {{"--name", "--label", "--network", "--tmpfs", "--user", "--entrypoint", "--cap-drop",
         "--security-opt", "--memory", "--pids-limit"}}
while at < len(argv):
    flag = argv[at]
    if flag == "--mount":
        spec = dict(part.split("=", 1) for part in argv[at + 1].split(",") if "=" in part)
        mounts[spec["dst"]] = spec["src"]
        at += 2
    elif flag == "-e":
        key, _, value = argv[at + 1].partition("=")
        env[key] = value
        at += 2
    elif flag in pairs:
        at += 2
    elif flag in ("--rm", "--read-only"):
        at += 1
    else:
        break
image, rest = argv[at], argv[at + 1:]
assert rest[:2] == ["-I", "-c"], rest[:2]
script, args = rest[2], rest[3:]
name = argv[argv.index("--name") + 1]
fail = os.environ.get("FAKE_RUNTIME_FAIL", "")
if fail and name.startswith("osprey-canopy-push-" + fail.split(":")[0] + "-"):
    sys.exit(int(fail.split(":")[1]))
sleep = os.environ.get("FAKE_RUNTIME_SLEEP", "")
if sleep and name.startswith("osprey-canopy-push-" + sleep.split(":")[0] + "-"):
    time.sleep(float(sleep.split(":")[1]))
def host(path):
    for dst, src in mounts.items():
        if path == dst or path.startswith(dst + "/"):
            base = src if dst == "/s" else os.path.join(os.environ["FAKE_VOLUME_ROOT"], src)
            return base + path[len(dst):]
    return path
payload = os.environ.get("FAKE_PAYLOAD_DIR")
for key in ("CANOPY_VIEWER_DIST",):
    if key in env and payload:
        env[key] = os.path.join(payload, env[key].rsplit("/", 1)[1])
full = {{k: v for k, v in os.environ.items() if not k.startswith("COVERAGE_")}}
full.update(env)
sys.exit(subprocess.call([sys.executable, "-I", "-c", script, *[host(a) for a in args]], env=full))
"""


# --- fixtures ----------------------------------------------------------------------------------


class Fake:
    """The fake runtime, its volumes and the log of every call."""

    def __init__(self, root: Path) -> None:
        self.bin = root / "fake-bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(FAKE_DOCKER, encoding="utf-8")
        (self.bin / "docker").chmod(0o755)
        self.log = root / "runtime.log"
        self.volumes_file = root / "volumes.txt"
        self.volumes_file.write_text("", encoding="utf-8")
        self.volume_root = root / "volumes"
        self.volume_root.mkdir()
        self.payload = root / "payload"
        (self.payload / "viewer").mkdir(parents=True)
        types = ("group", "label", "action-button", "embedded-display", "image-view", "pv-label")
        (self.payload / "viewer" / "ir-map.json").write_text(
            json.dumps({t: {"consumes": [], "verbatim": False} for t in types}), encoding="utf-8"
        )

    def volume(self, user: str) -> Path:
        name = f"proj_{user}-agent-data"
        with self.volumes_file.open("a", encoding="utf-8") as handle:
            handle.write(name + "\n")
        path = self.volume_root / name / "canopy" / "push"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def runs(self, verb: str) -> list[list[str]]:
        return [
            c for c in self.calls() if c[:1] == ["run"] and f"osprey-canopy-push-{verb}-" in c[3]
        ]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    fake = Fake(tmp_path)
    monkeypatch.setenv("PATH", f"{fake.bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_RUNTIME_LOG", str(fake.log))
    monkeypatch.setenv("FAKE_VOLUMES", str(fake.volumes_file))
    monkeypatch.setenv("FAKE_VOLUME_ROOT", str(fake.volume_root))
    monkeypatch.setenv("FAKE_PAYLOAD_DIR", str(fake.payload))
    monkeypatch.delenv("FAKE_RUNTIME_FAIL", raising=False)
    monkeypatch.delenv("FAKE_RUNTIME_SLEEP", raising=False)
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    runtime_helper.reset_runtime_cache()
    yield fake
    runtime_helper.reset_runtime_cache()


def _node(nid, ntype="label", props=None, children=None, **extra):
    node = {
        "id": nid,
        "type": ntype,
        "props": props or {},
        "geometry": {"x": 0, "y": 0, "width": 10, "height": 10},
    }
    if children is not None:
        node["children"] = children
    node.update(extra)
    return node


def _rule(target: str) -> dict:
    return {
        "id": "r",
        "name": "r",
        "targetProperty": target,
        "default": "x",
        "pvs": [],
        "conditions": [],
    }


def _doc(*children, title="t", description=None):
    metadata = {"title": title, "source": {"type": "native"}, "size": {"width": 10, "height": 10}}
    if description is not None:
        metadata["description"] = description
    return {
        "schemaVersion": "1.0",
        "kind": "screen",
        "id": "t",
        "metadata": metadata,
        "macros": [],
        "root": _node("root", "group", children=list(children)),
    }


def _bytes(doc) -> bytes:
    return (json.dumps(doc, indent=2) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_report(library: Path, **rows) -> None:
    """A report that passes ``library_is_fresh``: schema, digest and list, plus whatever rows are given."""
    judged = judge(scan(library))
    doc = {
        "generator": {"schema": "canopy-library-report/1", "canopy_version": "0"},
        "files": {
            "digest": judged.digest,
            "list": [{"path": p, "sha256": s} for p, s in judged.files],
        },
        **rows,
    }
    (library / "LIBRARY-REPORT.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (library / "LIBRARY-REPORT.md").write_text("# forged\n", encoding="utf-8")


KNOB = _doc(
    _node(
        "k",
        "action-button",
        {"actions": [{"type": "write_pv", "pvName": "SR:KNOB:SP", "value": "1"}]},
    )
)


@pytest.fixture
def deployment(tmp_path: Path) -> Path:
    root = tmp_path / "mca"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: proj\n", encoding="utf-8")
    config = {"project_name": "proj", "modules": {"web_terminals": roster_document("local")}}
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_renders(root)
    library = root / "data" / "canopy" / "library"
    for path, doc in (
        ("common/KnobPanel", KNOB),
        ("gtb/x", _doc(_node("l", "label", {"text": "gtb"}))),
    ):
        target = library / "screens" / f"{path}.screen.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_bytes(doc))
    write_report(library)
    return root


def library_of(deployment: Path) -> Path:
    return deployment / "data" / "canopy" / "library"


def request_doc(files: dict[str, bytes], **overrides) -> dict:
    doc = {
        "format": "canopy-push/1",
        "rid": RID,
        "source_id": "rf/view",
        "source_dir": "rf",
        "rev": _sha(files.get("view.screen.json", b"")),
        "target_dir": "sr/rf",
        "note": "RF overview, used every shift",
        "created": "2026-10-07T00:00:00Z",
        "files": [{"path": p, "size": len(d), "sha256": _sha(d)} for p, d in sorted(files.items())],
    }
    doc.update(overrides)
    return doc


def write_bundle(
    volume: Path, files: dict[str, bytes], rid: str = RID, request: dict | None = None
) -> Path:
    bundle = volume / rid
    for path, data in files.items():
        target = bundle / "files" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    doc = request_doc(files, rid=rid) if request is None else request
    (bundle / "request.json").write_bytes(
        (json.dumps(doc, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    )
    return bundle


def simple_files(**extra) -> dict[str, bytes]:
    view = _doc(
        _node(
            "b",
            "action-button",
            {
                "actions": [
                    {"type": "write_pv", "pvName": "SR:RF:SP", "value": "2"},
                    {"type": "open_webpage", "url": "https://example.test/doc"},
                ]
            },
        ),
        _node("l", "label", {"text": "free label text"}),
        _node("e", "embedded-display", {"file": "parts/inner.screen.json"}),
        _node("i", "image-view", {"src": "img/logo.png"}),
        description="only in metadata",
    )
    inner = _doc(_node("p", "pv-label", {"text": "inner"}))
    files = {
        "view.screen.json": _bytes(view),
        "parts/inner.screen.json": _bytes(inner),
        "img/logo.png": PNG,
    }
    files.update(extra)
    return files


def run(deployment: Path, *args: str, capsys) -> tuple[int, str, str]:
    code = push.main(["--repo", str(deployment), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --- list ----------------------------------------------------------------------------------------


def test_list_prints_one_line_per_request_and_skips_users_without_a_volume(
    fake, deployment, capsys
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    write_bundle(
        bob,
        simple_files(),
        rid=OTHER_RID,
        request=request_doc(simple_files(), rid=OTHER_RID, note="x\ny"),
    )
    (bob / "not-a-rid").mkdir()
    (bob / "0000000000000000").mkdir()
    code, out, err = run(deployment, "list", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    lines = out.splitlines()
    assert lines[0] == "alice: no volume"
    assert (
        f"bob/{RID}  created=2026-10-07T00:00:00Z  rf/view -> sr/rf  note=RF overview, used every shift"
        in lines
    )
    assert f"bob/{OTHER_RID}  created=2026-10-07T00:00:00Z  rf/view -> sr/rf  note=x\\ny" in lines
    assert "bob: (ignored: not-a-rid)" in lines
    assert "bob/0000000000000000: (unreadable: FileNotFoundError)" in lines
    assert lines[-1] == "carol: no volume"
    assert len(fake.runs("list")) == 1
    assert "\r" not in out


def test_list_reads_a_request_through_the_cap(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    bundle = write_bundle(bob, simple_files())
    (bundle / "request.json").write_bytes(b"{" + b" " * grammar.MAX_REQUEST_BYTES + b"}")
    code, out, _err = run(deployment, "list", capsys=capsys)
    assert code == refusals.EXIT_OK
    assert f"bob/{RID}: (unreadable: request.json too large)" in out.splitlines()


# --- common rules ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "code"),
    [
        ("bob", refusals.EXIT_BAD_RID),
        ("bob/short", refusals.EXIT_BAD_RID),
        ("bob/" + "G" * 16, refusals.EXIT_BAD_RID),
        ("Bob/" + RID, refusals.EXIT_BAD_USER),
        ("bob\n/" + RID, refusals.EXIT_BAD_USER),
        ("mallory/" + RID, refusals.EXIT_NOT_ON_ROSTER),
        ("alice/" + RID, refusals.EXIT_NO_VOLUME),
    ],
)
def test_a_bad_user_or_rid_has_its_own_code(fake, deployment, capsys, target, code) -> None:
    fake.volume("bob")
    assert run(deployment, "show", target, capsys=capsys)[0] == code
    assert fake.runs("show-stream") == []


def test_a_non_owner_show_is_refused(fake, deployment, capsys, monkeypatch) -> None:
    fake.volume("bob")
    other = os.geteuid() + 1
    monkeypatch.setattr(deployment_module, "_effective_uid", lambda: other)
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_NOT_OWNER
    assert "run as the deployment's owner" in err


def test_a_missing_library_is_refused(fake, deployment, capsys) -> None:
    fake.volume("bob")
    code, _out, _err = run(
        deployment, "--library", str(deployment / "absent"), "show", f"bob/{RID}", capsys=capsys
    )
    assert code == refusals.EXIT_NO_LIBRARY


@pytest.mark.usefixtures("fake")
def test_no_deployment_is_refused(tmp_path, capsys) -> None:
    (tmp_path / "empty").mkdir()
    assert run(tmp_path / "empty", "list", capsys=capsys)[0] == refusals.EXIT_NO_DEPLOYMENT
    (tmp_path / "empty" / "profile.yml").write_text("name: x\n", encoding="utf-8")
    assert run(tmp_path / "empty", "list", capsys=capsys)[0] == refusals.EXIT_NO_DEPLOYMENT


def test_a_stale_library_refuses_show_and_prints_the_differing_path(
    fake, deployment, capsys
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    library = library_of(deployment)
    (library / "screens" / "gtb" / "x.screen.json").write_bytes(_bytes(_doc(title="changed")))
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_STALE
    assert "screens/gtb/x.screen.json" in err and "canopy_mcp.report" in err
    assert fake.runs("show-stream") == []


def test_a_forged_report_that_matches_the_files_passes_freshness_but_never_feeds_the_output(
    fake, deployment, capsys
) -> None:
    """Only ``files`` is trusted; a forged ``writeBindings`` row list is not read at all."""
    library = library_of(deployment)
    write_report(library, writeBindings=[{"top": "forged", "pvResolved": "FORGED:PV"}])
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    code, out, _err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code != refusals.EXIT_STALE
    assert "FORGED" not in out


@pytest.mark.usefixtures("fake")
def test_list_is_a_verb_held_under_the_push_lock(deployment, capsys) -> None:
    lock = deployment / "var" / "canopy" / "push.lock"
    lock.parent.mkdir(parents=True)
    import fcntl

    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        code, _out, err = run(deployment, "list", capsys=capsys)
    finally:
        os.close(fd)
    assert code == refusals.EXIT_BUSY and "another push review holds" in err


# --- stream ----------------------------------------------------------------------------------------


def test_a_missing_bundle_says_it_was_evicted(fake, deployment, capsys) -> None:
    fake.volume("bob")
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_BUNDLE_MISSING
    assert "bundle evicted from bob's volume (newest-8 limit); ask the user to re-push" in err


def test_a_symlink_in_the_bundle_is_refused_in_the_container(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    bundle = write_bundle(bob, simple_files())
    (bundle / "files" / "img" / "logo.png").unlink()
    (bundle / "files" / "img" / "logo.png").symlink_to(deployment / "profile.yml")
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_STREAM_LINK and "img/logo.png" in err


def test_a_hard_link_in_the_bundle_is_refused(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    bundle = write_bundle(bob, simple_files())
    os.link(bundle / "files" / "img" / "logo.png", bundle / "files" / "img" / "twin.png")
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_STREAM_LINK and "hard link" in err


def test_a_fifo_in_the_bundle_is_refused(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    bundle = write_bundle(bob, simple_files())
    os.mkfifo(bundle / "files" / "pipe.png")
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_STREAM_LINK


def test_an_oversized_member_is_refused_at_the_stream(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files(**{"img/big.png": PNG + b"\x00" * push.MAX_ASSET_BYTES}))
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_STREAM_CAP and "per-file cap" in err


def test_too_many_members_are_refused_at_the_stream(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    many = {f"img/{n}.png": PNG for n in range(push.MAX_FILES)}
    write_bundle(bob, simple_files(**many))
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_STREAM_CAP and "too many files" in err


def test_the_host_kills_a_stream_that_breaches_a_cap_itself() -> None:
    """The host re-applies every cap on what it reads, whatever the container claimed."""
    import tarfile

    info = tarfile.TarInfo("files/x.png")
    info.size = push.MAX_ASSET_BYTES + 1
    assert verbs._member_breach(info, 0, 0) == "files/x.png: over the per-file cap"
    info.size = 1
    assert verbs._member_breach(info, push.MAX_FILES, 0) == "too many files"
    assert verbs._member_breach(info, 0, push.MAX_TOTAL_BYTES) == "bundle too large"
    link = tarfile.TarInfo("files/l")
    link.type = tarfile.SYMTYPE
    assert verbs._member_breach(link, 0, 0) == "files/l: not a regular file"


def test_a_container_that_cannot_start_has_its_own_code(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    monkeypatch.setenv("FAKE_RUNTIME_FAIL", "show-stream:125")
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_CONTAINER_START
    monkeypatch.setenv("FAKE_RUNTIME_FAIL", "show-stream:9")
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_CONTAINER_FAILED


def test_a_container_that_hangs_is_killed_by_name(fake, deployment, capsys, monkeypatch) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    monkeypatch.setattr(container, "CONTAINER_TIMEOUT_S", 1)
    monkeypatch.setenv("FAKE_RUNTIME_SLEEP", "show-stream:30")
    started = time.monotonic()
    assert (
        run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_CONTAINER_TIMEOUT
    )
    assert time.monotonic() - started < 20
    kills = [c for c in fake.calls() if c[:1] == ["kill"]]
    assert kills and kills[0][1].startswith("osprey-canopy-push-show-stream-")


# --- validate -----------------------------------------------------------------------------------------


def test_the_layout_is_request_at_the_top_and_everything_under_files() -> None:
    with pytest.raises(push.Refusal) as refused:
        verbs._validate_layout({"request.json": b"{}", "files/a.png": PNG, "extra": b""})
    assert refused.value.code == refusals.EXIT_BAD_LAYOUT
    for name in ("../x", "/abs", "files//a", "files/./a", "files/../a"):
        with pytest.raises(push.Refusal) as refused:
            verbs._validate_layout({"request.json": b"{}", name: PNG})
        assert refused.value.code == refusals.EXIT_BAD_LAYOUT, name
    with pytest.raises(push.Refusal) as refused:
        verbs._validate_layout({"files/a.png": PNG})
    assert refused.value.code == refusals.EXIT_BAD_LAYOUT


def test_an_extra_file_beside_request_json_is_refused(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    bundle = write_bundle(bob, simple_files())
    (bundle / "stray").write_bytes(b"x")
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_BAD_LAYOUT


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"format": "canopy-push/2"}, refusals.EXIT_BAD_REQUEST),
        ({"rid": OTHER_RID}, refusals.EXIT_BAD_REQUEST),
        ({"note": "x" * 2001}, refusals.EXIT_BAD_REQUEST),
        ({"note": "bad ‮ note"}, refusals.EXIT_BAD_REQUEST),
        ({"rev": "nothex"}, refusals.EXIT_BAD_REQUEST),
        ({"source_id": "../view"}, refusals.EXIT_BAD_SOURCE_DIR),
        ({"source_dir": "other"}, refusals.EXIT_BAD_SOURCE_DIR),
        ({"source_dir": ""}, refusals.EXIT_BAD_SOURCE_DIR),
        ({"target_dir": "../x"}, refusals.EXIT_BAD_TARGET_DIR),
        ({"target_dir": "library/x"}, refusals.EXIT_BAD_TARGET_DIR),
        ({"target_dir": ""}, refusals.EXIT_BAD_TARGET_DIR),
        ({"target_dir": "sr/var"}, refusals.EXIT_BAD_TARGET_DIR),
        ({"target_dir": "sr/.git"}, refusals.EXIT_BAD_TARGET_DIR),
        ({"files": []}, refusals.EXIT_FILES_MISMATCH),
    ],
)
def test_a_bad_request_json_has_its_own_code(fake, deployment, capsys, override, code) -> None:
    bob = fake.volume("bob")
    files = simple_files()
    write_bundle(bob, files, request={**request_doc(files), **override})
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == code


def test_a_request_json_with_other_keys_is_refused(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    files = simple_files()
    write_bundle(bob, files, request={**request_doc(files), "extra": 1})
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_BAD_REQUEST


def test_a_member_whose_bytes_differ_from_the_request_is_refused(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    bundle = write_bundle(bob, simple_files())
    (bundle / "files" / "img" / "logo.png").write_bytes(PNG + b"!")
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_FILES_MISMATCH


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        ({"var/x.png": PNG}, refusals.EXIT_RESERVED_SEGMENT),
        ({".git/x.png": PNG}, refusals.EXIT_RESERVED_SEGMENT),
        ({"img/logo.svg": b"<svg/>"}, refusals.EXIT_SUFFIX),
        ({"img/notes.txt": b"x"}, refusals.EXIT_SUFFIX),
        ({"img/bad\u0007.png": PNG}, refusals.EXIT_BAD_SEGMENT),
        ({"img/bad‮.png": PNG}, refusals.EXIT_BAD_SEGMENT),
        ({"café.screen.json": _bytes(_doc())}, refusals.EXIT_BAD_SEGMENT),
        ({"parts/c1.screen.json": _bytes(_doc(title="a\u009bb"))}, refusals.EXIT_INVALID_TEXT),
        ({"parts/bidi.screen.json": _bytes(_doc(title="a‮b"))}, refusals.EXIT_INVALID_TEXT),
        ({"parts/sur.screen.json": b'{"root": "\\ud800"}'}, refusals.EXIT_INVALID_TEXT),
        ({"parts/bom.screen.json": b"\xef\xbb\xbf{}"}, refusals.EXIT_INVALID_TEXT),
        ({"parts/notjson.screen.json": b"nope"}, refusals.EXIT_INVALID_TEXT),
    ],
)
def test_the_grammar_is_re_applied_on_the_host(fake, deployment, capsys, extra, code) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files(**extra))
    got, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert got == code, err
    assert "\r" not in err and "‮" not in err and "\u0007" not in err


def test_two_members_that_differ_only_in_case_are_refused() -> None:
    """Judged on the bytes read, since this host cannot hold such a pair on disk (APFS)."""
    for extra in ({"parts/Inner.screen.json": _bytes(_doc())}, {"IMG/x.png": PNG}):
        with pytest.raises(push.Refusal) as refused:
            grammar._check_grammar(simple_files(**extra))
        assert refused.value.code == refusals.EXIT_CASE_COLLISION
        assert "differ only in case" in refused.value.lines[0]
    with pytest.raises(push.Refusal) as refused:
        grammar._check_grammar({f"img/{n}.png": PNG for n in range(push.MAX_FILES + 1)})
    assert refused.value.code == refusals.EXIT_TOO_MANY_FILES
    with pytest.raises(push.Refusal) as refused:
        grammar._check_grammar({f"img/{n}.png": PNG + b"\x00" * (4 * 2**20) for n in range(9)})
    assert refused.value.code == refusals.EXIT_BUNDLE_TOO_LARGE
    with pytest.raises(push.Refusal) as refused:
        grammar._check_grammar({"img/big.png": PNG + b"\x00" * push.MAX_ASSET_BYTES})
    assert refused.value.code == refusals.EXIT_FILE_TOO_LARGE


def test_an_invalid_text_refusal_names_its_json_pointer(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files(**{"parts/c1.screen.json": _bytes(_doc(title="a\u009bb"))}))
    _code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert "at '/metadata/title'" in err


def test_a_destination_that_differs_only_in_case_from_a_library_file_is_refused(
    fake, deployment, capsys
) -> None:
    bob = fake.volume("bob")
    files = {"knobpanel.screen.json": _bytes(KNOB)}
    write_bundle(
        bob,
        files,
        request=request_doc(files, source_id="x/knobpanel", source_dir="x", target_dir="common"),
    )
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_CASE_FILE
    assert "'screens/common/KnobPanel.screen.json'" in err


def test_a_destination_directory_that_differs_only_in_case_from_a_library_directory_is_refused(
    fake, deployment, capsys
) -> None:
    bob = fake.volume("bob")
    files = {"y.screen.json": _bytes(_doc())}
    write_bundle(
        bob, files, request=request_doc(files, source_id="x/y", source_dir="x", target_dir="GTB")
    )
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_CASE_DIR
    assert "'screens/gtb'" in err


def test_the_case_check_is_against_the_listing_as_spelled() -> None:
    judged = judge(scan(Path(__file__).parent / "fixtures" / "push_library"))
    files = {"Main.screen.json": b"{}"}
    ok = verbs.Request(RID, "x/Main", "x", "top", "", "", "0" * 64, b"", files)
    verbs._check_case_against_library(ok, judged)
    bad = verbs.Request(
        RID, "x/main", "x", "top", "", "", "0" * 64, b"", {"main.screen.json": b"{}"}
    )
    with pytest.raises(push.Refusal) as refused:
        verbs._check_case_against_library(bad, judged)
    assert refused.value.code == refusals.EXIT_CASE_FILE
    worse = verbs.Request(RID, "x/y", "x", "Top", "", "", "0" * 64, b"", {"y.screen.json": b"{}"})
    with pytest.raises(push.Refusal) as refused:
        verbs._check_case_against_library(worse, judged)
    assert refused.value.code == refusals.EXIT_CASE_DIR


# --- prune --------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fake")
def test_stages_are_pruned_by_age_and_by_rejection_only(deployment) -> None:
    dep = deployment_module.Deployment(deployment, {}, library_of(deployment), "docker")
    stages = deployment / "var" / "canopy-push" / "bob"
    old = stages / f"{RID}.aaaa"
    rejected = stages / OTHER_RID
    kept = stages / "1111111111111111"
    replaced = stages / "replaced" / f"{RID}.2026"
    for path in (old, rejected, kept, replaced):
        path.mkdir(parents=True)
    then = time.time() - 15 * 24 * 3600
    os.utime(old, (then, then))
    os.utime(replaced, (then, then))
    log = deployment / "var" / "canopy" / "push-log.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps({"rid": OTHER_RID, "volume_user": "bob", "decision": "rejected"})
        + "\nnot json\n"
    )
    verbs._prune_stages(dep)
    assert not old.exists() and not rejected.exists()
    assert kept.exists() and replaced.exists()


# --- the check container and the output (CANOPY) ------------------------------------------------------------


@pytest.mark.canopy
def test_show_stages_checks_and_prints_the_review(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    code, out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    lines = out.splitlines()
    assert lines[0] == "filed from bob's volume: an ask approval is not proof the user filed it"
    assert f"request: bob/{RID}  source=rf/view  source_dir='rf'  target_dir=sr/rf" in lines[1]
    assert lines[2] == "note: RF overview, used every shift"
    assert "  screens/sr/rf/view.screen.json  " in out and "  screens/sr/rf/img/logo.png  " in out
    assert "pvResolved=SR:RF:SP" in out and "kind=write_pv" in out
    assert "url=https://example.test/doc" in out
    assert "== deltas: would-be minus current (both recomputed) ==" in out
    assert "  writeBindings: +1 -0" in out
    assert "  screens/sr/rf/view.screen.json  new" in out
    assert "  screens/sr/rf/view.screen.json /root/children/1/props/text: free label text" in out
    assert "  screens/sr/rf/view.screen.json /metadata/description: only in metadata" in out
    assert "SR:RF:SP" not in out.split("== free text")[1]
    digests = [line for line in lines if line.startswith("digest: ")]
    assert len(digests) == 1 and len(digests[0]) == len("digest: ") + 64
    assert "\r" not in out
    stage = deployment / "var" / "canopy-push" / "bob" / RID
    assert sorted(p.name for p in stage.iterdir()) == [
        "check-notes.json",
        "current",
        "overlay",
        "request.json",
        "would-be",
    ]
    assert (stage / "overlay" / "screens" / "sr" / "rf" / "img" / "logo.png").read_bytes() == PNG
    assert (stage / "would-be" / "screens" / "sr" / "rf" / "view.screen.json").exists()
    assert (stage / "current" / "screens" / "common" / "KnobPanel.screen.json").exists()
    assert not (stage / "current" / "screens" / "sr").exists()
    for path in stage.rglob("*"):
        mode = stat.S_IMODE(path.lstat().st_mode)
        assert mode == (0o755 if path.is_dir() else 0o644), path
    assert [p.name for p in stage.parent.iterdir()] == [RID]
    assert len(fake.runs("show-stream")) == 1 and len(fake.runs("show-check")) == 1
    check = fake.runs("show-check")[0]
    assert check[check.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


@pytest.mark.canopy
def test_a_second_show_replaces_the_stage_and_repeats_the_digest(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    _code, first, _err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    code, second, _err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK
    digest = [line for line in first.splitlines() if line.startswith("digest: ")]
    assert digest == [line for line in second.splitlines() if line.startswith("digest: ")]
    assert [p.name for p in (deployment / "var" / "canopy-push" / "bob").iterdir()] == [RID]


@pytest.mark.canopy
def test_the_digest_changes_with_the_library(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    _code, first, _err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    library = library_of(deployment)
    (library / "screens" / "gtb" / "x.screen.json").write_bytes(_bytes(_doc(title="changed")))
    write_report(library)
    _code, second, _err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    one = [line for line in first.splitlines() if line.startswith("digest: ")]
    two = [line for line in second.splitlines() if line.startswith("digest: ")]
    assert one and two and one != two


@pytest.mark.canopy
def test_the_rows_come_from_the_recomputed_report_not_the_stored_one(
    fake, deployment, capsys
) -> None:
    """A forged library report passes freshness (its files match) and feeds nothing printed."""
    library = library_of(deployment)
    write_report(library, writeBindings=[{"top": "common/KnobPanel", "pvResolved": "FORGED:PV"}])
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    code, out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert "FORGED" not in out and "pvResolved=SR:RF:SP" in out


@pytest.mark.canopy
def test_a_collision_prints_the_replace_note_and_the_changed_rows(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    new = _doc(
        _node(
            "k",
            "action-button",
            {"actions": [{"type": "write_pv", "pvName": "SR:KNOB:NEW", "value": "1"}]},
        )
    )
    files = {"KnobPanel.screen.json": _bytes(new)}
    write_bundle(
        bob,
        files,
        request=request_doc(files, source_id="x/KnobPanel", source_dir="x", target_dir="common"),
    )
    code, out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert "  screens/common/KnobPanel.screen.json  COLLIDES with an existing library file" in out
    assert "accept needs --replace" in out and "var/canopy-push/bob/replaced/" in out
    assert "    + " in out and "pvResolved=SR:KNOB:NEW" in out
    assert "    - " in out and "pvResolved=SR:KNOB:SP" in out


@pytest.mark.canopy
def test_terminal_controls_in_a_pv_name_and_a_note_never_start_a_line(
    fake, deployment, capsys
) -> None:
    bob = fake.volume("bob")
    view = _doc(
        _node(
            "b",
            "action-button",
            {"actions": [{"type": "write_pv", "pvName": "SR:A\rcover text", "value": "1"}]},
        )
    )
    files = {"view.screen.json": _bytes(view)}
    write_bundle(bob, files, request=request_doc(files, note="ok\ndigest: " + "f" * 64))
    code, out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    assert "\r" not in out
    assert "pvResolved=SR:A\\rcover text" in out
    assert "note: ok\\ndigest: " in out
    assert len([line for line in out.splitlines() if line.startswith("digest: ")]) == 1


@pytest.mark.canopy
@pytest.mark.parametrize(
    ("files", "code"),
    [
        (
            {
                "view.screen.json": _bytes(_doc(_node("i", "image-view", {"src": "img/x.png"}))),
                "img/x.png": b"not an image",
            },
            refusals.EXIT_NOT_IMAGE,
        ),
        (
            {
                "view.screen.json": _bytes(
                    _doc(_node("e", "embedded-display", {"file": "${P}/x.screen.json"}))
                )
            },
            refusals.EXIT_MACRO_REF,
        ),
        (
            {
                "view.screen.json": _bytes(
                    _doc(_node("e", "embedded-display", {"file": "parts/absent.screen.json"}))
                )
            },
            refusals.EXIT_UNRESOLVED_REF,
        ),
        (
            {
                "view.screen.json": _bytes(
                    _doc(_node("e", "embedded-display", {"file": "../gtb/x.screen.json"}))
                )
            },
            refusals.EXIT_REF_LEAVES_SOURCE,
        ),
        (
            {
                "view.screen.json": _bytes(
                    _doc(_node("e", "embedded-display", {"file": "/gtb/x.screen.json"}))
                )
            },
            refusals.EXIT_ROOT_RELATIVE_REF,
        ),
        (
            {
                "view.screen.json": _bytes(
                    _doc(
                        _node(
                            "e",
                            "embedded-display",
                            {"file": "parts/inner.screen.json"},
                            rules=[_rule("file")],
                        )
                    )
                ),
                "parts/inner.screen.json": _bytes(_doc()),
            },
            refusals.EXIT_RULE_RETARGET,
        ),
        (
            {
                "view.screen.json": _bytes(
                    _doc(
                        _node(
                            "l",
                            "pv-label",
                            {"pv": "A"},
                            rules=[_rule("pvName")],
                        )
                    )
                )
            },
            refusals.EXIT_RULE_RETARGET,
        ),
        (
            {"view.screen.json": _bytes(_doc()), "parts/extra.screen.json": _bytes(_doc())},
            refusals.EXIT_CLOSURE_MISMATCH,
        ),
        (
            {"view.screen.json": _bytes(_doc(_node("w", "no-such-widget")))},
            refusals.EXIT_CHECK_ERRORS,
        ),
    ],
)
def test_every_refusal_of_the_check_container_has_its_own_code(
    fake, deployment, capsys, files, code
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, files)
    got, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert got == code, err
    assert "the staged push failed its checks" in err
    assert not list((deployment / "var" / "canopy-push").rglob(RID + "*")), (
        "a failed stage is removed"
    )


@pytest.mark.canopy
def test_a_failed_check_keeps_the_previous_checked_stage(fake, deployment, capsys) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_OK
    bundle = bob / RID
    for path in bundle.rglob("*"):
        if path.is_file():
            path.unlink()
    files = {"view.screen.json": _bytes(_doc(_node("w", "no-such-widget")))}
    write_bundle(bob, files)
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_CHECK_ERRORS
    assert [p.name for p in (deployment / "var" / "canopy-push" / "bob").iterdir()] == [RID]


def test_a_check_container_that_fails_to_start_has_its_own_code(
    fake, deployment, capsys, monkeypatch
) -> None:
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    monkeypatch.setenv("FAKE_RUNTIME_FAIL", "show-check:126")
    assert run(deployment, "show", f"bob/{RID}", capsys=capsys)[0] == refusals.EXIT_CONTAINER_START
    assert not list((deployment / "var" / "canopy-push").rglob(RID + "*"))


@pytest.mark.canopy
def test_library_override_reviews_against_that_library(fake, deployment, capsys, tmp_path) -> None:
    other = tmp_path / "other-library"
    (other / "screens" / "sr").mkdir(parents=True)
    (other / "screens" / "sr" / "a.screen.json").write_bytes(_bytes(_doc()))
    write_report(other)
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    code, out, err = run(deployment, "--library", str(other), "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    stage = deployment / "var" / "canopy-push" / "bob" / RID
    assert (stage / "current" / "screens" / "sr" / "a.screen.json").exists()
    assert not (stage / "current" / "screens" / "common").exists()
    assert shlex.quote(str(other)) not in out.split("digest:")[1]


# --- spec_unavailable ---------------------------------------------------------------------------


@pytest.mark.canopy
def test_show_without_a_staged_spec_prints_one_spec_unavailable_line_and_keeps_it(
    fake, deployment, capsys
) -> None:
    """The fake payload stages no spec/, so every screen is checked without the IR schema."""
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    code, out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_OK, err
    lines = out.splitlines()
    spec_lines = [line for line in lines if line.startswith("spec_unavailable:")]
    assert len(spec_lines) == 1, out
    assert (
        lines.index(spec_lines[0])
        == next(at for at, line in enumerate(lines) if line.startswith("stage: ")) + 1
    ), "the line follows stage:"
    stage = deployment / "var" / "canopy-push" / "bob" / RID
    notes = json.loads((stage / verbs.CHECK_NOTES_NAME).read_text(encoding="ascii"))
    assert "sr/rf/view" in {note["screen"] for note in notes["specUnavailable"]}
    assert all(note["message"] for note in notes["specUnavailable"])
    assert spec_lines[0] == verbs.spec_unavailable_line(notes["specUnavailable"])


@pytest.mark.canopy
@pytest.mark.parametrize("bad", ["nope", [{"screen": 1, "message": "m"}], [{"screen": "s"}], ["x"]])
def test_a_malformed_spec_unavailable_in_the_verdict_has_its_own_code(
    fake, deployment, capsys, monkeypatch, bad
) -> None:
    real = container._run_container

    def doctored(runtime, argv, **kwargs):
        done = real(runtime, argv, **kwargs)
        if "-c" in argv and argv[argv.index("-c") + 1] == container.CHECK_SCRIPT:
            verdict = json.loads(done.stdout)
            verdict["specUnavailable"] = bad
            done = subprocess.CompletedProcess(
                done.args, done.returncode, json.dumps(verdict).encode(), done.stderr
            )
        return done

    monkeypatch.setattr(container, "_run_container", doctored)
    bob = fake.volume("bob")
    write_bundle(bob, simple_files())
    code, _out, err = run(deployment, "show", f"bob/{RID}", capsys=capsys)
    assert code == refusals.EXIT_BAD_VERDICT, err
    assert "malformed specUnavailable" in err
    assert not list((deployment / "var" / "canopy-push").rglob(RID + "*"))


def test_a_verdict_without_spec_unavailable_reads_as_none() -> None:
    """A CANOPY that predates the staged spec reports no notes; that is none, not malformed."""
    assert verbs.spec_unavailable_notes({"ok": True}) == []
    assert verbs.spec_unavailable_line([]) is None
    assert verbs.spec_unavailable_record([]) is None


def test_the_spec_unavailable_line_names_each_screen_once() -> None:
    notes = [
        {"screen": "b", "message": "no schema"},
        {"screen": "a", "message": "no schema"},
        {"screen": "a", "message": "no props"},
    ]
    assert verbs.spec_unavailable_record(notes) == {
        "screens": ["a", "b"],
        "messages": ["no props", "no schema"],
    }
    line = verbs.spec_unavailable_line(notes)
    assert line is not None and "\n" not in line
    assert (
        line.startswith("spec_unavailable: ") and "2 screen(s) (a, b): no props; no schema" in line
    )
