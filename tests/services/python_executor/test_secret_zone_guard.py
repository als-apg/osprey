"""Executed code cannot open the deployment's secret files.

The filesystem guard both sandboxes emit refuses, in both postures and for
reads as well as writes, an open of a ``.env`` or ``.env.*`` file under a
secret root and of any ``/proc/<...>/environ`` or ``/proc/<...>/cmdline``.
The check runs before the bypass markers and the permitted roots, so nothing
carves a secret file back out.

Renderer cases run the emitted guard in a real subprocess, as
``test_fs_guard.py`` does, so the patching never leaks into the test process.
The executor and visualization-sandbox cases run the real spawn paths.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from osprey.services.python_executor.execution.fs_guard import (
    DEFAULT_DENYLIST_PREFIX,
    EXECUTOR_PATCH_TARGETS,
    SANDBOX_PATCH_TARGETS,
    SANDBOX_WRITE_MODES_ONLY_TARGETS,
    render_fs_guard,
)
from osprey.services.python_executor.execution.wrapper import (
    READONLY_REFUSAL_MARKER,
    ExecutionWrapper,
)

_BYPASS = ("site-packages", "lib/python", sys.prefix)
_SECRET_DETAIL = "Secret files and process environments cannot be opened by executed code."

# --------------------------------------------------------------------------- #
# Renderer, run in a real subprocess
# --------------------------------------------------------------------------- #


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    """A project holding the env chain, plus an unrelated directory beside it."""
    tmp = tmp_path.resolve()
    project = tmp / "project"
    (project / "build").mkdir(parents=True)
    (project / ".env").write_text("EXAMPLE_PROVIDER_API_KEY=secret\n")
    (project / ".env.shared").write_text("EXAMPLE_SHARED=secret\n")
    (project / "build" / ".env.merged").write_text("EXAMPLE_MERGED=secret\n")
    (project / "data.csv").write_text("a,b\n1,2\n")
    outside = tmp / "outside"
    outside.mkdir()
    (outside / ".env").write_text("UNRELATED=1\n")
    return {"tmp": tmp, "project": project, "outside": outside}


def _denylist(layout: dict[str, Path], **overrides) -> str:
    kwargs = {
        "default_deny": False,
        "permitted_roots": (),
        "protected_roots": (),
        "read_roots": (),
        "patch_targets": EXECUTOR_PATCH_TARGETS,
        "secret_roots": (layout["project"],),
    }
    kwargs.update(overrides)
    return render_fs_guard(**kwargs)


def _allowlist(layout: dict[str, Path], **overrides) -> str:
    kwargs = {
        "default_deny": True,
        "permitted_roots": (layout["project"], layout["outside"]),
        "protected_roots": (),
        "read_roots": (),
        "bypass_prefixes": _BYPASS,
        "patch_targets": EXECUTOR_PATCH_TARGETS,
        "secret_roots": (layout["project"],),
    }
    kwargs.update(overrides)
    return render_fs_guard(**kwargs)


def _probe(tmp: Path, guard: str, attempts: dict[str, str]) -> dict[str, str]:
    """Run each attempt under *guard*; map its label to ``OK`` or the refusal text."""
    lines = [guard, "import os, shutil", "from pathlib import Path", "_results = {}"]
    for label, expression in attempts.items():
        lines.append(
            textwrap.dedent(
                f"""
                try:
                    {expression}
                    _results[{label!r}] = "OK"
                except PermissionError as _exc:
                    _results[{label!r}] = str(_exc)
                except OSError:
                    _results[{label!r}] = "OK"
                """
            )
        )
    lines.append("import json; print('RESULT ' + json.dumps(_results))")
    script = tmp / "probe_script.py"
    script.write_text("\n".join(lines), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    import json

    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line.removeprefix("RESULT "))


def _chain_attempts(project: Path) -> dict[str, str]:
    return {
        "read .env": f"open({str(project / '.env')!r}).read()",
        "write .env": f"open({str(project / '.env')!r}, 'a').close()",
        "read .env.shared": f"open({str(project / '.env.shared')!r}).read()",
        "write .env.shared": f"open({str(project / '.env.shared')!r}, 'a').close()",
        "read build/.env.merged": f"open({str(project / 'build' / '.env.merged')!r}).read()",
        "write build/.env.merged": (
            f"open({str(project / 'build' / '.env.merged')!r}, 'a').close()"
        ),
    }


@pytest.mark.parametrize("posture", ["denylist", "allowlist"])
def test_env_chain_files_are_refused_in_both_postures(layout, posture):
    guard = (_denylist if posture == "denylist" else _allowlist)(layout)

    results = _probe(layout["tmp"], guard, _chain_attempts(layout["project"]))

    for label, outcome in results.items():
        verb = label.split(" ", 1)[0]
        assert f"{verb} denied" in outcome, (label, outcome)
        assert _SECRET_DETAIL in outcome, (label, outcome)


def test_env_file_outside_every_secret_root_is_not_refused(layout):
    results = _probe(
        layout["tmp"],
        _denylist(layout),
        {"read": f"open({str(layout['outside'] / '.env')!r}).read()"},
    )
    assert results == {"read": "OK"}


def test_proc_environment_is_refused_with_no_secret_roots(layout):
    results = _probe(
        layout["tmp"],
        _denylist(layout, secret_roots=()),
        {
            "environ": "open('/proc/self/environ', 'rb').read()",
            "cmdline": "open('/proc/1/cmdline', 'rb').read()",
            "task environ": "open('/proc/1/task/1/environ', 'rb').read()",
        },
    )
    for label, outcome in results.items():
        assert "read denied" in outcome and _SECRET_DETAIL in outcome, (label, outcome)


def test_an_open_file_descriptor_passes(layout):
    data = layout["project"] / "data.csv"
    results = _probe(
        layout["tmp"],
        _denylist(layout),
        {"fd": f"open(os.open({str(data)!r}, os.O_RDONLY)).read()"},
    )
    assert results == {"fd": "OK"}


def test_a_permitted_root_holding_the_project_does_not_carve_env_back_out(layout):
    results = _probe(
        layout["tmp"],
        _denylist(layout, permitted_roots=(layout["tmp"],)),
        {"read": f"open({str(layout['project'] / '.env')!r}).read()"},
    )
    assert "read denied" in results["read"]


def test_write_modes_only_target_still_refuses_a_secret_read(layout):
    project = layout["project"]
    guard = _allowlist(
        layout,
        permitted_roots=(),
        read_roots=(project,),
        patch_targets=SANDBOX_PATCH_TARGETS,
        write_modes_only_targets=SANDBOX_WRITE_MODES_ONLY_TARGETS,
    )

    results = _probe(
        layout["tmp"],
        guard,
        {
            "env": f"Path({str(project / '.env')!r}).read_text()",
            "data": f"Path({str(project / 'data.csv')!r}).read_text()",
        },
    )

    assert "read denied" in results["env"] and _SECRET_DETAIL in results["env"]
    assert results["data"] == "OK"


def test_wrapper_renders_its_secret_roots(tmp_path):
    root = tmp_path.resolve()
    script = ExecutionWrapper(secret_roots=(root,)).create_wrapper("print(1)", tmp_path)
    assert f"_OSPREY_FS_SECRET_ROOTS = {(str(root),)!r}" in script


def test_bare_wrapper_still_renders_and_runs(tmp_path):
    """No roots needed: the ``/proc`` rule stands alone."""
    execution_folder = tmp_path / "run"
    execution_folder.mkdir()
    script = ExecutionWrapper().create_wrapper("print('RAN')", execution_folder)
    assert "_OSPREY_FS_SECRET_ROOTS = ()" in script
    path = execution_folder / "wrapped_script.py"
    path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
    )
    assert "RAN" in proc.stdout, proc.stderr


# --------------------------------------------------------------------------- #
# The python executor, real subprocess
# --------------------------------------------------------------------------- #


@pytest.fixture
def reset_config_caches(monkeypatch):
    """Reset every config cache around a test that writes its own ``config.yml``."""
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()

    import osprey.utils.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    saved_cache = _cfg._config_cache.copy()
    _cfg._config_cache.clear()

    yield

    reset_config_cache()
    _cfg._config_cache.clear()
    _cfg._config_cache.update(saved_cache)


@pytest.fixture
def mock_project(tmp_path, monkeypatch, request) -> Path:
    """A mock-control-system project holding the env chain, as the working directory."""
    request.getfixturevalue("reset_config_caches")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "control_system": {"type": "mock", "limits_checking": {"enabled": False}},
                "execution": {"execution_method": "subprocess"},
                "python_executor": {"execution_timeout_seconds": 120},
            }
        )
    )
    (tmp_path / ".env").write_text("EXAMPLE_PROVIDER_API_KEY=from-dotenv\n")
    (tmp_path / ".env.shared").write_text("EXAMPLE_SHARED=from-shared\n")
    (tmp_path / "var" / "agent_data").mkdir(parents=True)
    return tmp_path


_EXECUTOR_ATTEMPTS = {
    "open": "open('.env').read()",
    "read_text": "from pathlib import Path\nPath('.env.shared').read_text()",
    "copy": "import shutil\nshutil.copy('.env', 'var/agent_data/copied.txt')",
    "proc": "open('/proc/self/environ', 'rb').read()",
}


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
@pytest.mark.parametrize("attempt", sorted(_EXECUTOR_ATTEMPTS))
async def test_executor_refuses_secret_opens(mock_project, mode, attempt):
    from osprey.mcp_server.python_executor.executor import execute_code

    result = await execute_code(_EXECUTOR_ATTEMPTS[attempt], mode, "secret zone probe")

    assert not result.success
    assert "read denied" in result.stderr
    assert "from-dotenv" not in result.stdout + result.stderr
    assert "from-shared" not in result.stdout + result.stderr
    if mode == "readonly":
        assert READONLY_REFUSAL_MARKER in result.stderr
    else:
        assert DEFAULT_DENYLIST_PREFIX in result.stderr
        assert READONLY_REFUSAL_MARKER not in result.stderr
    assert not (mock_project / "var" / "agent_data" / "copied.txt").exists()


@pytest.mark.usefixtures("mock_project")
@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
async def test_executor_still_reads_the_config(mode):
    from osprey.mcp_server.python_executor.executor import execute_code

    result = await execute_code("print(len(open('config.yml').read()) > 0)", mode, "control")

    assert result.success, result.stderr
    assert "True" in result.stdout


@pytest.mark.usefixtures("mock_project")
async def test_readonly_secret_refusal_is_audited(monkeypatch):
    """The readonly stderr is the whole input the audit path reads; it reports it."""
    from osprey.mcp_server.python_executor.executor import execute_code
    from osprey.mcp_server.python_executor.tools import _execution_gates

    result = await execute_code("open('.env').read()", "readonly", "audit probe")
    assert READONLY_REFUSAL_MARKER in result.stderr

    recorded: list[dict] = []
    alerts: list[tuple] = []

    def fake_record(**kwargs):
        recorded.append(kwargs)

    async def fake_notify(*args, **kwargs):
        alerts.append((args, kwargs))

    monkeypatch.setattr(_execution_gates, "_record_write_refusal", fake_record)
    monkeypatch.setattr(_execution_gates, "notify_agent_activity_async", fake_notify)

    reported = await _execution_gates.report_runtime_refusal(
        tool="execute",
        stderr=result.stderr,
        code="open('.env').read()",
        description="audit probe",
        execution_mode="readonly",
    )

    assert reported is True
    assert len(recorded) == 1 and len(alerts) == 1
    trigger = "\n".join(recorded[0]["trigger"])
    assert "read denied" in trigger
    assert ".env" in trigger


# --------------------------------------------------------------------------- #
# The visualization sandbox, real subprocess
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "code",
    ["print(open('.env').read())", "from pathlib import Path\nprint(Path('.env').read_text())"],
    ids=["open", "read_text"],
)
async def test_visualization_sandbox_refuses_env(mock_project, code):
    from osprey.mcp_server.workspace.execution.sandbox_executor import execute_sandbox_code

    execution_folder = mock_project / "viz_run"
    execution_folder.mkdir()

    result = await execute_sandbox_code(code, execution_folder, timeout=120)

    assert not result.success
    assert "Sandbox: read denied" in result.stderr + (result.error_message or "")
    assert "from-dotenv" not in result.stdout


async def test_visualization_sandbox_still_reads_project_data(mock_project):
    from osprey.mcp_server.workspace.execution.sandbox_executor import execute_sandbox_code

    (mock_project / "data.csv").write_text("a,b\n1,2\n")
    execution_folder = mock_project / "viz_run"
    execution_folder.mkdir()

    result = await execute_sandbox_code(
        "from pathlib import Path\nprint(Path('data.csv').read_text())",
        execution_folder,
        timeout=120,
    )

    assert result.success, result.stderr
    assert "a,b" in result.stdout
