"""The notebook sidecar: a real ``jupyter_server`` on port 0 under a temp shared root.

Every test that spawns a server does so through :class:`JupyterSidecar` itself,
so what is pinned here is the launch contract the terminal relies on: the
token gate, the single kernelspec, where the secrets live, and that nothing
survives ``stop()`` or a dead parent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import nbformat
import pytest

from osprey.interfaces.web_terminal import jupyter_sidecar
from osprey.interfaces.web_terminal.jupyter_sidecar import (
    KERNELSPEC_NAME,
    LAB_DISABLED_EXTENSIONS,
    JupyterSidecar,
    lab_page_config,
    lab_theme_override,
    seed_starter_notebook,
    starter_read_channel,
)
from osprey.mcp_server.python_executor.executor import resolve_agent_interpreter

READY_TIMEOUT = 90.0

spawns = pytest.mark.slow


@pytest.fixture
def shared_root(tmp_path: Path) -> Path:
    return tmp_path / "shared"


@pytest.fixture
def config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    # Keep every per-launch tempdir under the test's own tree.
    monkeypatch.setenv("TMPDIR", str(tmp_path))


def _running(shared_root: Path, outer_prefix: str = "") -> Iterator[JupyterSidecar]:
    sidecar = JupyterSidecar(shared_root, outer_prefix, None)
    try:
        sidecar.spawn()
        sidecar.wait_ready(READY_TIMEOUT)
        yield sidecar
    finally:
        sidecar.stop()


@pytest.fixture
def sidecar(config_env: None, shared_root: Path) -> Iterator[JupyterSidecar]:
    yield from _running(shared_root)


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """An opener that reports a redirect instead of following it."""

    def redirect_request(self, *args: object, **kwargs: object) -> urllib.request.Request | None:
        return None


_no_redirect_opener = urllib.request.build_opener(_NoRedirect)


def _get_no_redirect(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    """GET *url* without following a redirect; returns status and ``Location``."""
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with _no_redirect_opener.open(request, timeout=10) as response:
            return int(response.status), response.headers.get("Location", "")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.headers.get("Location", "")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


# ---------------------------------------------------------------------------
# A running sidecar
# ---------------------------------------------------------------------------


@spawns
def test_status_answers_with_the_token_and_refuses_without(sidecar: JupyterSidecar) -> None:
    with_token, _ = _get(f"{sidecar.url}/api/status", sidecar.auth_headers)
    without, _ = _get(f"{sidecar.url}/api/status")

    assert with_token == 200
    assert without == 403


@spawns
def test_only_the_osprey_kernelspec_is_listed(sidecar: JupyterSidecar) -> None:
    status, body = _get(f"{sidecar.url}/api/kernelspecs", sidecar.auth_headers)

    assert status == 200
    assert sorted(json.loads(body)["kernelspecs"]) == [KERNELSPEC_NAME]


@spawns
def test_the_url_names_the_panel_without_a_trailing_slash(sidecar: JupyterSidecar) -> None:
    assert sidecar.pid is not None
    assert sidecar.url.startswith("http://127.0.0.1:")
    assert sidecar.url.endswith("/panel/jupyter")
    assert sidecar.token is not None
    assert sidecar.auth_headers == {"authorization": f"Bearer {sidecar.token}"}


@spawns
def test_the_root_redirects_to_jupyterlab_not_the_server_landing_page(
    sidecar: JupyterSidecar,
) -> None:
    status, location = _get_no_redirect(f"{sidecar.url}/", sidecar.auth_headers)
    assert status == 302
    assert location == f"{sidecar.base_url}lab?"

    lab_status, body = _get(f"{sidecar.url}/lab", sidecar.auth_headers)
    assert lab_status == 200
    assert b"<title>JupyterLab</title>" in body
    assert b"A Jupyter Server is running" not in body


@spawns
def test_the_outer_prefix_is_part_of_the_base_url(config_env: None, shared_root: Path) -> None:
    for prefixed in _running(shared_root, "/u/alice"):
        assert prefixed.url.endswith("/u/alice/panel/jupyter")
        status, _ = _get(f"{prefixed.url}/api/status", prefixed.auth_headers)
        assert status == 200


@spawns
def test_the_runtime_dir_is_outside_the_shared_root(
    sidecar: JupyterSidecar, shared_root: Path
) -> None:
    runtime_dir = sidecar.runtime_dir

    assert runtime_dir is not None
    assert not runtime_dir.resolve().is_relative_to(shared_root.resolve())
    assert (runtime_dir / f"jpserver-{sidecar.pid}.json").exists()
    assert list(shared_root.rglob("jpserver-*.json")) == []
    assert (runtime_dir.parent.stat().st_mode & 0o777) == 0o700


@spawns
def test_the_kernelspec_runs_the_agent_interpreter_with_the_two_env_keys(
    sidecar: JupyterSidecar, shared_root: Path
) -> None:
    spec = json.loads(sidecar.kernelspec_path.read_text(encoding="utf-8"))

    assert spec["argv"] == [
        str(resolve_agent_interpreter()),
        "-m",
        "osprey.jupyter_kernel",
        "-f",
        "{connection_file}",
    ]
    assert spec["language"] == "python"
    assert spec["env"] == {"JUPYTER_TOKEN": "", "OSPREY_AGENT_DATA_ROOT": str(shared_root)}
    assert sidecar.notebooks_dir == shared_root / "notebooks"
    assert sidecar.notebooks_dir.is_dir()


@spawns
def test_stop_leaves_no_process_and_removes_the_tempdir(
    config_env: None, shared_root: Path
) -> None:
    sidecar = JupyterSidecar(shared_root, "", None)
    sidecar.stop()  # nothing spawned yet: a no-op
    sidecar.spawn()
    try:
        sidecar.wait_ready(READY_TIMEOUT)
        pid = sidecar.pid
        runtime_dir = sidecar.runtime_dir
        assert pid is not None and runtime_dir is not None
    finally:
        sidecar.stop()

    assert not _alive(pid)
    assert not runtime_dir.parent.exists()
    assert sidecar.pid is None
    assert sidecar.runtime_dir is None
    sidecar.stop()  # a second stop is a no-op too


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_preflight_names_the_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    with pytest.raises(RuntimeError) as failure:
        JupyterSidecar(Path("/unused"), "", None).preflight()

    assert str(failure.value) == "OSPREY_CONFIG is not set"


def test_preflight_names_an_interpreter_that_cannot_import(
    config_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "python"
    fake.write_text(
        "#!/bin/sh\necho \"ModuleNotFoundError: No module named 'ipykernel'\" >&2\nexit 1\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(jupyter_sidecar, "resolve_agent_interpreter", lambda: fake)

    with pytest.raises(RuntimeError) as failure:
        JupyterSidecar(tmp_path / "shared", "", None).preflight()

    message = str(failure.value)
    assert "\n" not in message
    assert message.startswith(f"{fake} cannot import ipykernel and osprey.runtime: ")
    assert message.endswith("ModuleNotFoundError: No module named 'ipykernel'")


def test_preflight_passes_with_the_real_interpreter(config_env: None, tmp_path: Path) -> None:
    JupyterSidecar(tmp_path / "shared", "", None).preflight()


@spawns
def test_an_early_exit_surfaces_the_stderr_tail(
    config_env: None, shared_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "python"
    fake.write_text('#!/bin/sh\necho "first line" >&2\necho "the reason" >&2\nexit 3\n')
    fake.chmod(0o755)
    monkeypatch.setattr(jupyter_sidecar.sys, "executable", str(fake))
    sidecar = JupyterSidecar(shared_root, "", None)
    sidecar.spawn()
    try:
        with pytest.raises(RuntimeError) as failure:
            sidecar.wait_ready(10)
    finally:
        sidecar.stop()

    message = str(failure.value)
    assert message.startswith("Notebook sidecar exited with status 3 before it was ready")
    assert message.endswith("first line\nthe reason")


# ---------------------------------------------------------------------------
# Dev host: the sidecar follows its parent
# ---------------------------------------------------------------------------

_PARENT_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from osprey.interfaces.web_terminal.jupyter_sidecar import JupyterSidecar

    sidecar = JupyterSidecar(Path(sys.argv[1]), "", None)
    sidecar.spawn()
    sidecar.wait_ready(float(sys.argv[2]))
    print(sidecar.pid, flush=True)
    time.sleep(3600)
    """
)


@spawns
def test_the_sidecar_exits_when_its_parent_is_killed(
    config_env: None, shared_root: Path, tmp_path: Path
) -> None:
    script = tmp_path / "parent.py"
    script.write_text(_PARENT_SCRIPT)
    parent = subprocess.Popen(
        [sys.executable, str(script), str(shared_root), str(READY_TIMEOUT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    sidecar_pid: int | None = None
    try:
        assert parent.stdout is not None
        line = parent.stdout.readline()
        assert line.strip(), "the parent did not report a sidecar pid"
        sidecar_pid = int(line)
        assert _alive(sidecar_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(10)

        assert _wait_gone(sidecar_pid, 5.0), "the sidecar outlived its parent"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if sidecar_pid is not None and _alive(sidecar_pid):
            os.killpg(sidecar_pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Seeds at spawn: the pinned theme and the starter notebook
# ---------------------------------------------------------------------------

THEME_PLUGIN = "@jupyterlab/apputils-extension:themes"
OVERRIDE_RELPATH = Path("labconfig") / "default_setting_overrides.json"
PAGE_CONFIG_RELPATH = Path("labconfig") / "page_config.json"


@pytest.mark.parametrize(
    ("pinned_mode", "expected"),
    [
        ("dark", {THEME_PLUGIN: {"theme": "JupyterLab Dark"}}),
        ("light", {THEME_PLUGIN: {"theme": "JupyterLab Light"}}),
        (None, None),
        ("retro", None),
        ("desy", None),
    ],
)
def test_only_dark_and_light_name_a_jupyterlab_theme(
    pinned_mode: str | None, expected: dict[str, dict[str, str]] | None
) -> None:
    assert lab_theme_override(pinned_mode) == expected


def test_a_pinned_theme_is_seeded_into_the_launch_config_dir(
    shared_root: Path, tmp_path: Path
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    JupyterSidecar(shared_root, "", "dark")._seed(config_dir)

    override = json.loads((config_dir / OVERRIDE_RELPATH).read_text(encoding="utf-8"))
    assert override == {THEME_PLUGIN: {"theme": "JupyterLab Dark"}}


def test_a_family_theme_seeds_no_override_file(shared_root: Path, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    JupyterSidecar(shared_root, "", "retro")._seed(config_dir)

    assert not (config_dir / OVERRIDE_RELPATH).exists()
    assert [p.name for p in config_dir.rglob("*") if p.is_file()] == ["page_config.json"]


def test_the_page_config_switches_off_the_panel_foreign_plugins(
    shared_root: Path, tmp_path: Path
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    JupyterSidecar(shared_root, "", None)._seed(config_dir)

    page_config = json.loads((config_dir / PAGE_CONFIG_RELPATH).read_text(encoding="utf-8"))
    assert page_config == lab_page_config()
    assert page_config == {
        "disabledExtensions": {
            "@jupyterlab/extensionmanager-extension": True,
            "@jupyterlab/apputils-extension:announcements": True,
        }
    }
    assert set(page_config["disabledExtensions"]) == set(LAB_DISABLED_EXTENSIONS)


def test_the_argv_pins_the_server_traits(shared_root: Path) -> None:
    argv = JupyterSidecar(shared_root, "/u/alice", None)._argv()

    assert argv[:3] == [sys.executable, "-m", "osprey.interfaces.web_terminal.jupyter_sidecar_main"]
    assert "--FileContentsManager.delete_to_trash=False" in argv
    assert "--FileContentsManager.always_delete_dir=True" in argv
    assert "--LabApp.news_url=None" in argv
    assert (
        "--LabApp.check_for_updates_class=jupyterlab.handlers.announcements.NeverCheckForUpdate"
        in argv
    )
    assert "--ServerApp.base_url=/u/alice/panel/jupyter/" in argv
    assert f"--KernelSpecManager.allowed_kernelspecs=['{KERNELSPEC_NAME}']" in argv


def test_the_starter_notebook_is_written_into_an_empty_notebooks_dir(
    shared_root: Path, tmp_path: Path, monkeypatch
) -> None:
    # Pinned to a deployment that declares no channel, so this asserts the
    # starter's shape rather than whatever config the test host happens to
    # resolve. The channel-bearing case is covered separately.
    monkeypatch.setattr(jupyter_sidecar, "_deployment_config", dict)
    sidecar = JupyterSidecar(shared_root, "", None)

    sidecar._seed(tmp_path / "config")

    notebook = nbformat.read(sidecar.notebooks_dir / "getting-started.ipynb", as_version=4)
    assert [(cell.cell_type, cell.source) for cell in notebook.cells] == [
        (
            "markdown",
            "This kernel follows your terminal session. Open a chat session before writing.",
        ),
        ("code", "from osprey.runtime import read_channel, write_channel"),
    ]
    assert notebook.metadata["kernelspec"]["name"] == KERNELSPEC_NAME


def test_a_notebook_in_a_subdirectory_suppresses_the_starter(tmp_path: Path) -> None:
    notebooks_dir = tmp_path / "notebooks"
    (notebooks_dir / "runs").mkdir(parents=True)
    (notebooks_dir / "runs" / "yesterday.ipynb").write_text("{}", encoding="utf-8")

    assert seed_starter_notebook(notebooks_dir) is None
    assert list(notebooks_dir.glob("*.ipynb")) == []


def test_the_starter_is_never_rewritten(tmp_path: Path) -> None:
    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    written = seed_starter_notebook(notebooks_dir)
    assert written is not None
    written.write_text("edited by the user", encoding="utf-8")
    before = written.stat().st_mtime_ns

    assert seed_starter_notebook(notebooks_dir) is None

    assert written.read_text(encoding="utf-8") == "edited by the user"
    assert written.stat().st_mtime_ns == before


@spawns
def test_the_starter_notebook_is_listed_by_a_running_sidecar(sidecar: JupyterSidecar) -> None:
    status, body = _get(f"{sidecar.url}/api/contents/", sidecar.auth_headers)

    assert status == 200
    listed = {entry["name"]: entry["type"] for entry in json.loads(body)["content"]}
    assert listed["getting-started.ipynb"] == "notebook"


# ---------------------------------------------------------------------------
# What the running server does with the pinned traits and the page config
# ---------------------------------------------------------------------------


def _request(
    method: str, url: str, headers: dict[str, str], body: bytes | None = None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


@spawns
def test_raw_files_obey_the_same_confinement_as_the_contents_api(
    sidecar: JupyterSidecar, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (sidecar.notebooks_dir / "escape-file").symlink_to(outside / "secret.txt")
    (sidecar.notebooks_dir / "escape-dir").symlink_to(outside, target_is_directory=True)
    (sidecar.notebooks_dir / "alias.ipynb").symlink_to(
        sidecar.notebooks_dir / "getting-started.ipynb"
    )
    headers = sidecar.auth_headers

    status, body = _get(f"{sidecar.url}/files/getting-started.ipynb", headers)
    assert status == 200
    assert b'"cells"' in body

    status, body = _request("HEAD", f"{sidecar.url}/files/getting-started.ipynb", headers)
    assert status == 200
    assert body == b""

    status, _ = _get(f"{sidecar.url}/files/alias.ipynb", headers)
    assert status == 200

    for escape in (
        "escape-file",
        "escape-dir/secret.txt",
        "../outside/secret.txt",
        "..%2Foutside%2Fsecret.txt",
        "%2e%2e/outside/secret.txt",
    ):
        files_status, files_body = _get(f"{sidecar.url}/files/{escape}", headers)
        contents_status, _ = _get(f"{sidecar.url}/api/contents/{escape}", headers)
        assert files_status == 404, (escape, files_body)
        assert contents_status == 404, escape
        assert b"secret" not in files_body


@spawns
def test_delete_removes_the_file_instead_of_trashing_it(
    sidecar: JupyterSidecar, shared_root: Path
) -> None:
    headers = {**sidecar.auth_headers, "content-type": "application/json"}
    document = json.dumps({"type": "file", "format": "text", "content": "doomed"}).encode()

    status, _ = _request("PUT", f"{sidecar.url}/api/contents/doomed.txt", headers, document)
    assert status == 201
    assert (sidecar.notebooks_dir / "doomed.txt").read_text(encoding="utf-8") == "doomed"

    status, _ = _request("DELETE", f"{sidecar.url}/api/contents/doomed.txt", headers)
    assert status == 204

    assert not (sidecar.notebooks_dir / "doomed.txt").exists()

    folder = sidecar.notebooks_dir / "doomed-dir"
    folder.mkdir()
    (folder / "kept.txt").write_text("inside", encoding="utf-8")

    status, body = _request("DELETE", f"{sidecar.url}/api/contents/doomed-dir", headers)
    assert status == 204, body

    assert not folder.exists()
    assert list(shared_root.rglob(".Trash*")) == []


@spawns
def test_checkpoints_obey_the_same_confinement_as_the_contents_api(
    sidecar: JupyterSidecar, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    (outside / ".ipynb_checkpoints").mkdir(parents=True)
    (outside / "x.ipynb").write_text("{}", encoding="utf-8")
    (outside / ".ipynb_checkpoints" / "x-checkpoint.ipynb").write_text("{}", encoding="utf-8")
    (sidecar.notebooks_dir / "escape-dir").symlink_to(outside, target_is_directory=True)
    headers = sidecar.auth_headers
    escaped = f"{sidecar.url}/api/contents/escape-dir/x.ipynb/checkpoints"

    status, _ = _get(escaped, headers)
    assert status == 404
    status, _ = _request("DELETE", f"{escaped}/checkpoint", headers)
    assert status == 404
    status, _ = _request("POST", escaped, headers, b"")
    assert status == 404
    assert (outside / ".ipynb_checkpoints" / "x-checkpoint.ipynb").exists()

    inside = f"{sidecar.url}/api/contents/getting-started.ipynb/checkpoints"
    status, body = _request("POST", inside, headers, b"")
    assert status == 201
    checkpoint_id = json.loads(body)["id"]
    status, body = _get(inside, headers)
    assert status == 200
    assert [entry["id"] for entry in json.loads(body)] == [checkpoint_id]
    status, _ = _request("POST", f"{inside}/{checkpoint_id}", headers, b"")
    assert status == 204
    status, _ = _request("DELETE", f"{inside}/{checkpoint_id}", headers)
    assert status == 204
    assert list(sidecar.notebooks_dir.rglob("*-checkpoint.ipynb")) == []


def _page_config(lab_html: bytes) -> dict[str, object]:
    marker = b'id="jupyter-config-data" type="application/json">'
    start = lab_html.index(marker) + len(marker)
    end = lab_html.index(b"</script>", start)
    return json.loads(lab_html[start:end])


@spawns
def test_the_lab_page_lists_the_disabled_plugins_and_no_news(sidecar: JupyterSidecar) -> None:
    status, html = _get(f"{sidecar.url}/lab", sidecar.auth_headers)
    assert status == 200

    page_config = _page_config(html)
    disabled = page_config["disabledExtensions"]
    assert isinstance(disabled, list)
    assert set(LAB_DISABLED_EXTENSIONS) <= set(disabled)
    assert page_config["news"] == {"disabled": True}


# ---------------------------------------------------------------------------
# A sidecar that dies later
# ---------------------------------------------------------------------------


def _exit_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and " exited (rc=" in record.getMessage()
    ]


@spawns
def test_a_sidecar_that_dies_later_is_reported_once_with_its_stderr_tail(
    sidecar: JupyterSidecar, caplog: pytest.LogCaptureFixture
) -> None:
    fired = threading.Event()
    sidecar.on_exit = fired.set
    pid = sidecar.pid
    assert pid is not None

    with caplog.at_level("WARNING", logger=jupyter_sidecar.__name__):
        os.kill(pid, signal.SIGKILL)
        assert fired.wait(5.0), "on_exit did not fire"

    records = _exit_records(caplog)
    assert len(records) == 1
    message = records[0]
    assert message.startswith(f"Notebook sidecar exited (rc={-signal.SIGKILL})")
    tail = sidecar.stderr_tail
    assert tail, "the sidecar wrote nothing to stderr before it was killed"
    assert message.endswith(tail)


@spawns
def test_on_exit_registered_after_the_death_fires_at_once(sidecar: JupyterSidecar) -> None:
    reaped = threading.Event()
    sidecar.on_exit = reaped.set
    pid = sidecar.pid
    assert pid is not None
    os.kill(pid, signal.SIGKILL)
    assert reaped.wait(5.0)

    late = threading.Event()
    sidecar.on_exit = late.set

    assert late.is_set()


@spawns
def test_stop_reports_no_exit_and_fires_no_callback(
    config_env: None, shared_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sidecar = JupyterSidecar(shared_root, "", None)
    fired = threading.Event()
    sidecar.on_exit = fired.set
    sidecar.spawn()
    sidecar.wait_ready(READY_TIMEOUT)

    with caplog.at_level("WARNING", logger=jupyter_sidecar.__name__):
        sidecar.stop()
        time.sleep(0.2)

    assert not fired.is_set()
    assert _exit_records(caplog) == []


#: A stand-in sidecar: hands its stderr to a child in its own session, the way a
#: kernel inherits it, then idles until it is signalled.
_ORPHAN_SCRIPT = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen(['sleep', '3600'], start_new_session=True)\n"
    "open(sys.argv[1], 'w').write(str(child.pid))\n"
    "time.sleep(3600)\n"
)


@spawns
def test_stop_returns_while_an_orphan_still_holds_the_stderr_pipe(
    config_env: None, shared_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "orphan.pid"
    script = tmp_path / "orphan.py"
    script.write_text(_ORPHAN_SCRIPT, encoding="utf-8")
    fake = tmp_path / "python"
    fake.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "{pid_file}"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(jupyter_sidecar.sys, "executable", str(fake))
    sidecar = JupyterSidecar(shared_root, "", None)
    sidecar.spawn()
    orphan: int | None = None
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        orphan = int(pid_file.read_text())
        assert _alive(orphan)

        stopper = threading.Thread(target=sidecar.stop, daemon=True)
        started = time.monotonic()
        stopper.start()
        stopper.join(jupyter_sidecar._STOP_GRACE + 5)

        assert not stopper.is_alive(), "stop() hung on the orphan's pipe"
        assert time.monotonic() - started < jupyter_sidecar._STOP_GRACE + 5
        assert _alive(orphan), "the orphan is the test's own; stop() must not need it gone"
    finally:
        if orphan is not None and _alive(orphan):
            os.kill(orphan, signal.SIGKILL)
        sidecar.stop()


# ---------------------------------------------------------------------------
# The straggler reap
# ---------------------------------------------------------------------------


def _reap_with(monkeypatch, snapshots: list[list[int]]) -> list[tuple[int, int]]:
    """Run ``_reap_stragglers`` against a scripted sequence of pgrep results.

    Args:
        monkeypatch: The fixture used to stub the reaper's two shell touchpoints.
        snapshots: One ``_pids_naming`` return per call; the last repeats once
            the script runs out, so a test only spells out the interesting head.

    Returns:
        The ``(pid, signal)`` pairs the reap sent, in order.
    """
    calls = iter(snapshots)
    last: list[list[int]] = [snapshots[-1]]

    def _naming(_runtime_dir: str) -> list[int]:
        last[0] = next(calls, last[0])
        return list(last[0])

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(jupyter_sidecar.shutil, "which", lambda _name: "/usr/bin/pgrep")
    monkeypatch.setattr(jupyter_sidecar, "_pids_naming", _naming)
    monkeypatch.setattr(jupyter_sidecar, "_signal_pid", lambda pid, sig: sent.append((pid, sig)))
    jupyter_sidecar._reap_stragglers("/tmp/does-not-matter")
    return sent


def test_a_kernel_that_appears_after_the_first_snapshot_is_still_reaped(monkeypatch) -> None:
    """The reap waits for a straggler rather than trusting one sample.

    A kernel the server forked just before it died does not name the runtime
    dir until it execs, so the first snapshot can legitimately come back empty
    while a kernel is still on its way up. Returning there left it running with
    the control target and write posture it launched with, and nothing above it.
    """
    sent = _reap_with(monkeypatch, [[], [], [4242], []])

    assert (4242, signal.SIGTERM) in sent


def test_a_clean_stop_signals_nothing_and_does_not_wait_out_the_grace(monkeypatch) -> None:
    """Nothing to reap is the common path: no signals, and no _REAP_GRACE spent."""
    started = time.monotonic()
    sent = _reap_with(monkeypatch, [[]])
    elapsed = time.monotonic() - started

    assert sent == []
    # The settle window is bounded and the SIGTERM grace is never entered.
    assert elapsed < jupyter_sidecar._REAP_SETTLE + jupyter_sidecar._REAP_GRACE


def test_a_straggler_that_ignores_sigterm_is_killed(monkeypatch) -> None:
    """SIGTERM first, then SIGKILL once the grace elapses with it still listed."""
    sent = _reap_with(monkeypatch, [[99]])

    assert (99, signal.SIGTERM) in sent
    assert (99, signal.SIGKILL) in sent
    assert sent.index((99, signal.SIGTERM)) < sent.index((99, signal.SIGKILL))


def test_a_straggler_that_takes_sigterm_is_not_killed(monkeypatch) -> None:
    """A process that goes away during the grace never receives SIGKILL."""
    sent = _reap_with(monkeypatch, [[7], [7], []])

    assert (7, signal.SIGTERM) in sent
    assert (7, signal.SIGKILL) not in sent


class TestTheStarterNotebooksExampleRead:
    """Which channel the starter notebook reads, and when it reads nothing.

    A deployment is entitled to serve no channel at all — the hello-world
    preset runs a mock connector and stands up no server of its own — so the
    read line is earned, never assumed. Two config blocks already name a
    channel the deployment says it can read; nothing here invents a third.
    """

    def test_the_archivers_canary_channel_is_the_example(self) -> None:
        """A freshness canary is declared to keep moving, so it shows a value.

        The archiver check names the one channel a facility promises is still
        changing. That is what makes a demo read worth running twice.
        """
        config = {
            "health": {
                "categories": {
                    "archiver": {
                        "checks": [
                            {
                                "type": "archiver_freshness",
                                "channel": "SR:DIAG:DCCT:01:CURRENT:RB",
                            }
                        ]
                    }
                }
            }
        }

        assert starter_read_channel(config) == "SR:DIAG:DCCT:01:CURRENT:RB"

    def test_a_targets_probe_channel_is_the_fallback(self) -> None:
        """Without an archiver, the target switch still names a readable channel."""
        config = {
            "control_system": {
                "connector": {"va": {"probe_channel": "SR:VAC:GAUGE:SR01:PRESSURE:RB"}}
            }
        }

        assert starter_read_channel(config) == "SR:VAC:GAUGE:SR01:PRESSURE:RB"

    def test_the_canary_outranks_a_probe_channel(self) -> None:
        config = {
            "health": {
                "categories": {
                    "archiver": {
                        "checks": [
                            {
                                "type": "archiver_freshness",
                                "channel": "SR:DIAG:DCCT:01:CURRENT:RB",
                            }
                        ]
                    }
                }
            },
            "control_system": {
                "connector": {"va": {"probe_channel": "SR:VAC:GAUGE:SR01:PRESSURE:RB"}}
            },
        }

        assert starter_read_channel(config) == "SR:DIAG:DCCT:01:CURRENT:RB"

    def test_the_generic_templates_placeholder_is_not_a_channel(self) -> None:
        """``YOUR:PROBE:CHANNEL`` ships unfilled and names nothing.

        The build refuses to write a placeholder probe channel for the same
        reason: it would make a target look reachable while naming a channel
        nothing serves. A starter cell reading it fails on its first run.
        """
        config = {
            "control_system": {"connector": {"live": {"probe_channel": "YOUR:PROBE:CHANNEL"}}}
        }

        assert starter_read_channel(config) is None

    def test_a_mock_only_deployment_names_no_channel(self) -> None:
        """The hello-world shape: a mock connector, no server, no declaration."""
        config = {"control_system": {"type": "mock", "writes_enabled": False}}

        assert starter_read_channel(config) is None

    def test_the_starter_reads_the_channel_it_is_given(self, tmp_path: Path) -> None:
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()

        seed_starter_notebook(notebooks_dir, "SR:DIAG:DCCT:01:CURRENT:RB")

        notebook = nbformat.read(notebooks_dir / "getting-started.ipynb", as_version=4)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        assert code == [
            "from osprey.runtime import read_channel, write_channel\n\n"
            'read_channel("SR:DIAG:DCCT:01:CURRENT:RB")'
        ]

    def test_no_channel_leaves_the_import_line_alone(self, tmp_path: Path) -> None:
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()

        seed_starter_notebook(notebooks_dir, None)

        notebook = nbformat.read(notebooks_dir / "getting-started.ipynb", as_version=4)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        assert code == ["from osprey.runtime import read_channel, write_channel"]

    def test_a_built_control_assistant_offers_its_beam_current(self, tmp_path: Path) -> None:
        """The resolver is pinned to the shape the build really emits.

        The canary is declared as ``va_archiver.freshness_channel`` and reaches
        the deployed config only after the build derives a health check from it
        and writes that under a dotted key. Hand-building the nested dict here
        would test the resolver against an assumption; this renders the config
        through ``osprey build``'s own override machinery instead, so a change
        in how those keys are written fails here rather than in a deployment.
        """
        import yaml

        from osprey.cli.build_cmd import _apply_config_overrides
        from osprey.cli.build_profile_archiver import (
            VAArchiverConfig,
            va_archiver_config_overrides,
        )

        project = tmp_path / "project"
        project.mkdir()
        (project / "config.yml").write_text(
            "control_system:\n"
            "  type: mock\n"
            "  connector:\n"
            "    va:\n"
            "      probe_channel: SR:VAC:GAUGE:SR01:PRESSURE:RB\n",
            encoding="utf-8",
        )
        # The channel the control-assistant preset names as its canary.
        overrides = va_archiver_config_overrides(
            VAArchiverConfig(freshness_channel="SR:DIAG:DCCT:01:CURRENT:RB")
        )
        _apply_config_overrides(project, overrides)
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))

        assert starter_read_channel(config) == "SR:DIAG:DCCT:01:CURRENT:RB"

    def test_the_seed_reads_the_deployments_own_channel(
        self, shared_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """The sidecar resolves the channel from the deployment it runs in."""
        monkeypatch.setattr(
            jupyter_sidecar,
            "_deployment_config",
            lambda: {
                "control_system": {"connector": {"va": {"probe_channel": "SR:MY:OWN:CHANNEL"}}}
            },
        )
        sidecar = JupyterSidecar(shared_root, "", None)

        sidecar._seed(tmp_path / "config")

        notebook = nbformat.read(sidecar.notebooks_dir / "getting-started.ipynb", as_version=4)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        assert code == [
            'from osprey.runtime import read_channel, write_channel\n\nread_channel("SR:MY:OWN:CHANNEL")'
        ]

    def test_an_unreadable_config_still_seeds_a_notebook(
        self, shared_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """A config that will not load costs the read line, never the panel.

        The starter notebook is a convenience. Failing the sidecar's seed over
        it would take the whole tab down for a cosmetic line.
        """

        def _boom() -> dict[str, object]:
            raise RuntimeError("no config here")

        monkeypatch.setattr(jupyter_sidecar, "_deployment_config", _boom)
        sidecar = JupyterSidecar(shared_root, "", None)

        sidecar._seed(tmp_path / "config")

        notebook = nbformat.read(sidecar.notebooks_dir / "getting-started.ipynb", as_version=4)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        assert code == ["from osprey.runtime import read_channel, write_channel"]
