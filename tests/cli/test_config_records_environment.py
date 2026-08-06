"""A generated ``config.yml`` records the environment *declaration*, not a host path.

The ``execution:`` block of every rendered project states which environment the
project was built from — ``environment: {python, packages, inherit_exclude}``,
exactly as the build profile declared it — so a build can be reproduced from the
config alone.

What it deliberately does **not** record is the interpreter that runs agent
Python. That path used to be baked in as ``execution.python_env_path`` at build
time, which made every generated config a snapshot of one machine's filesystem:
move the project, mount it into a container, or rebuild the venv elsewhere and
the recorded path pointed at nothing. The interpreter is now resolved at run
time (the project's own ``.venv`` when it has one, else the interpreter running
OSPREY), so no absolute interpreter path is written to config.yml at all. These
tests assert that absence directly — it is what lets the heal-and-strip
workarounds that existed to repair such stale paths be deleted.

Configs already deployed with the retired key must keep loading: it is ignored
with a debug log, never an error.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.config import ConfigBuilder

# The three bundled templates that render an ``execution:`` block. The other app
# bundles (ariel_standalone, channel_finder_standalone) ship no such block and so
# never recorded an interpreter path.
CONFIG_TEMPLATES = (
    "project/config.yml.j2",
    "apps/control_assistant/config.yml.j2",
    "apps/hello_world/config.yml.j2",
)

# Any path that names a Python interpreter, e.g. /Users/x/proj/.venv/bin/python3.11.
_INTERPRETER_PATH = re.compile(r"bin/python[\d.]*$")

# ``osprey.utils.config`` logs under a flat name, not its module path.
_CONFIG_LOGGER = "CONFIG"


def _walk_scalars(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every scalar in a parsed config."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_scalars(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_scalars(value, f"{path}[{index}]")
    else:
        yield path.lstrip("."), node


def _interpreter_paths(config: dict) -> list[tuple[str, str]]:
    """Return every ``(dotted_path, value)`` in *config* that names an interpreter."""
    return [
        (path, value)
        for path, value in _walk_scalars(config)
        if isinstance(value, str) and _INTERPRETER_PATH.search(value)
    ]


def _build_project(output_dir: Path, project_name: str, *extra_args: str) -> Path:
    """Build the control-assistant preset in-process and return the project dir.

    ``--skip-deps --skip-lifecycle`` keeps the build Docker- and network-free:
    the rendered config.yml is all these tests read.
    """
    result = CliRunner().invoke(
        build,
        [
            project_name,
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(output_dir),
            *extra_args,
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = output_dir / project_name
    assert (project_dir / "config.yml").exists()
    return project_dir


def _fake_venv_interpreter(root: Path) -> Path:
    """Create a venv-shaped tree and return its interpreter path.

    ``environment.python`` is validated as an existing executable file, and
    ``inherit_exclude`` additionally requires a venv base (a ``pyvenv.cfg``
    beside ``bin/``). With ``--skip-deps`` the interpreter is never executed, so
    a stub is enough to exercise the declaration end to end.
    """
    venv = root / "declared-venv"
    (venv / "bin").mkdir(parents=True)
    interpreter = venv / "bin" / "python"
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    return interpreter


class TestTemplatesRenderTheDeclaration:
    """All three config templates render an ``environment:`` record and no host path."""

    @pytest.fixture(scope="class")
    def jinja_env(self):
        return TemplateManager().jinja_env

    @pytest.mark.parametrize("template_name", CONFIG_TEMPLATES)
    def test_renders_without_a_declaration(self, jinja_env, template_name):
        """With nothing declared the block still renders, as empty defaults."""
        rendered = jinja_env.get_template(template_name).render({})
        execution = yaml.safe_load(rendered)["execution"]

        assert execution["execution_method"] == "subprocess"
        assert execution["environment"] == {
            "python": None,
            "packages": [],
            "inherit_exclude": [],
        }

    @pytest.mark.parametrize("template_name", CONFIG_TEMPLATES)
    def test_declaration_round_trips(self, jinja_env, template_name):
        """The declared values reach the rendered YAML unchanged."""
        rendered = jinja_env.get_template(template_name).render(
            {
                "environment_python": "/opt/facility/analysis/bin/python",
                "environment_packages": ["numpy>=2", "lmfit"],
                "environment_inherit_exclude": ["pytest", "ruff"],
            }
        )
        environment = yaml.safe_load(rendered)["execution"]["environment"]

        assert environment == {
            "python": "/opt/facility/analysis/bin/python",
            "packages": ["numpy>=2", "lmfit"],
            "inherit_exclude": ["pytest", "ruff"],
        }

    @pytest.mark.parametrize("template_name", CONFIG_TEMPLATES)
    def test_no_python_env_path_key(self, jinja_env, template_name):
        """The retired key is gone from the templates entirely."""
        rendered = jinja_env.get_template(template_name).render({})

        assert "python_env_path" not in rendered
        assert "python_env_path" not in yaml.safe_load(rendered)["execution"]


class TestGeneratedConfigRecordsNoInterpreterPath:
    """A built project's config.yml carries no absolute interpreter path.

    This is the precondition for deleting the heal (``osprey claude regen``) and
    strip (container staging) steps: neither has anything left to repair once the
    build stops recording a path.
    """

    @pytest.fixture(scope="class")
    def built_config_text(self, tmp_path_factory) -> str:
        """Build once per class — every test here only reads the rendered config."""
        output_dir = tmp_path_factory.mktemp("records-environment")
        project_dir = _build_project(output_dir, "plain-build")
        return (project_dir / "config.yml").read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def built_config(self, built_config_text: str) -> dict:
        return yaml.safe_load(built_config_text)

    def test_execution_method_is_subprocess(self, built_config):
        assert built_config["execution"]["execution_method"] == "subprocess"

    def test_undeclared_environment_records_empty_defaults(self, built_config):
        """A profile that declares no environment records exactly that."""
        assert built_config["execution"]["environment"] == {
            "python": None,
            "packages": [],
            "inherit_exclude": [],
        }

    def test_python_env_path_is_absent(self, built_config):
        assert "python_env_path" not in built_config["execution"]

    def test_no_interpreter_path_anywhere_in_the_config(self, built_config):
        """Not just under ``execution:`` — nowhere in the whole document."""
        found = _interpreter_paths(built_config)
        assert found == [], f"generated config records interpreter path(s): {found}"

    def test_build_interpreter_is_not_written_to_the_file(self, built_config_text):
        """The interpreter that ran the build leaves no trace, comments included."""
        assert sys.executable not in built_config_text


class TestDeclaredEnvironmentIsRecordedVerbatim:
    """A profile that declares an environment gets that declaration back."""

    @pytest.fixture(scope="class")
    def declared(self, tmp_path_factory) -> tuple[Path, dict]:
        output_dir = tmp_path_factory.mktemp("declared-environment")
        interpreter = _fake_venv_interpreter(output_dir)
        project_dir = _build_project(
            output_dir,
            "declared-build",
            "--set",
            f"environment.python={interpreter}",
            "--set",
            'environment.packages=["numpy>=2", "lmfit"]',
            "--set",
            'environment.inherit_exclude=["pytest"]',
        )
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
        return interpreter, config

    def test_declaration_round_trips_into_the_config(self, declared):
        interpreter, config = declared
        assert config["execution"]["environment"] == {
            "python": str(interpreter),
            "packages": ["numpy>=2", "lmfit"],
            "inherit_exclude": ["pytest"],
        }

    def test_python_env_path_is_absent(self, declared):
        _, config = declared
        assert "python_env_path" not in config["execution"]

    def test_declared_base_is_the_only_recorded_interpreter(self, declared):
        """The declared base is an input, recorded as provenance. Nothing else —
        in particular no build-machine path — joins it."""
        interpreter, config = declared
        assert _interpreter_paths(config) == [("execution.environment.python", str(interpreter))]


class TestRetiredPythonEnvPathIsIgnored:
    """Projects deployed with the old key must keep loading, unchanged."""

    @staticmethod
    def _legacy_config(tmp_path: Path) -> Path:
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "project_root: /srv/als-assistant\n"
            "execution:\n"
            '  execution_method: "local"\n'
            "  python_env_path: /vanished/host/.venv/bin/python\n"
            "  modes:\n"
            "    read_only:\n"
            "      allows_writes: false\n",
            encoding="utf-8",
        )
        return config_file

    def test_old_config_loads_without_error(self, tmp_path):
        builder = ConfigBuilder(str(self._legacy_config(tmp_path)), load_env=False)

        assert builder.configurable["execution"]["execution_method"] == "local"

    def test_retired_keys_are_dropped_from_the_execution_config(self, tmp_path):
        builder = ConfigBuilder(str(self._legacy_config(tmp_path)), load_env=False)
        execution = builder.configurable["execution"]

        assert "python_env_path" not in execution
        # Jupyter-era kernel/gateway descriptions: retired alongside it.
        assert "modes" not in execution

    def test_sibling_execution_keys_survive(self, tmp_path):
        builder = ConfigBuilder(str(self._legacy_config(tmp_path)), load_env=False)

        assert builder.configurable["execution"]["execution_method"] == "local"

    def test_it_is_logged_at_debug_not_warning(self, tmp_path, caplog):
        config_file = self._legacy_config(tmp_path)

        with caplog.at_level(logging.DEBUG, logger=_CONFIG_LOGGER):
            ConfigBuilder(str(config_file), load_env=False)

        records = [r for r in caplog.records if "python_env_path" in r.getMessage()]
        assert records, "expected a log line about the retired key"
        assert [r.levelno for r in records] == [logging.DEBUG]
        assert "/vanished/host/.venv/bin/python" in records[0].getMessage()

    def test_a_current_config_says_nothing_about_the_retired_key(self, tmp_path, caplog):
        """No noise on the configs the build actually generates."""
        config_file = tmp_path / "config.yml"
        config_file.write_text('execution:\n  execution_method: "subprocess"\n', encoding="utf-8")

        with caplog.at_level(logging.DEBUG, logger=_CONFIG_LOGGER):
            ConfigBuilder(str(config_file), load_env=False)

        assert not [r for r in caplog.records if "python_env_path" in r.getMessage()]

    def test_defaults_record_no_interpreter_path(self, tmp_path):
        """The synthesized defaults for a config with no ``execution:`` block
        must not reintroduce an interpreter path either."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_root: /srv/als-assistant\n", encoding="utf-8")

        execution = ConfigBuilder(str(config_file), load_env=False).configurable["execution"]

        assert execution["execution_method"] == "subprocess"
        assert "python_env_path" not in execution
        assert _interpreter_paths(execution) == []
