"""The ambient environment is not a source for the built project's ``.env``.

The build profile is the single source of the project ``.env``. Nothing the
operator's shell happens to export, and nothing sitting in the directory
``osprey build`` was launched from, may reach the rendered file: an ambient
value makes a build non-reproducible in a way that is invisible in the output —
the same profile on two machines yields different ``.env`` contents, and for a
provider key the difference is a secret that reaches the project, and every
container started from it, without ever being recorded in the profile that is
supposed to be the audit surface.

Absence is the correct outcome. A key appearing without the profile declaring
it is the bug these tests pin.
"""

from __future__ import annotations

from pathlib import Path

from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.scaffolding import provider_api_key_entries
from osprey.utils.dotenv import parse_dotenv_file

#: A provider key from the real registry, so the pin cannot drift from the set
#: the template actually iterates.
_A_PROVIDER_KEY = "CBORG_API_KEY"


def _build(tmp_path: Path, name: str, **kwargs) -> Path:
    """Build a minimal project and return its directory."""
    return TemplateManager().create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="hello_world",
        context={"default_provider": "anthropic"},
        **kwargs,
    )


def _project_env(project_dir: Path) -> dict[str, str]:
    return parse_dotenv_file(project_dir / ".env")


class TestAmbientProviderKeysDoNotReachTheProject:
    """A key the shell exports but the profile does not declare is absent."""

    def test_exported_key_absent_when_the_profile_does_not_declare_it(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(_A_PROVIDER_KEY, "from-the-shell")

        project = _build(tmp_path, "ambient-key-project")

        assert _A_PROVIDER_KEY not in _project_env(project)
        assert "from-the-shell" not in (project / ".env").read_text(encoding="utf-8")

    def test_no_registry_provider_key_leaks(self, tmp_path, monkeypatch):
        """Pinned across the whole registry, not just one variable."""
        monkeypatch.chdir(tmp_path)
        entries = provider_api_key_entries()
        assert entries  # the loop below must not pass vacuously
        for entry in entries:
            monkeypatch.setenv(entry["var"], f"shell-{entry['provider']}")

        project = _build(tmp_path, "every-key-project")

        rendered = _project_env(project)
        leaked = [entry["var"] for entry in entries if entry["var"] in rendered]
        assert not leaked, f"ambient provider keys reached the project .env: {leaked}"

    def test_profile_declared_key_is_rendered(self, tmp_path, monkeypatch):
        """The other half: a key the profile declares DOES land."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(_A_PROVIDER_KEY, raising=False)

        project = _build(
            tmp_path,
            "declared-key-project",
            profile_env_text=f"{_A_PROVIDER_KEY}=from-the-profile\n",
        )

        assert _project_env(project)[_A_PROVIDER_KEY] == "from-the-profile"

    def test_profile_wins_over_the_shell(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(_A_PROVIDER_KEY, "from-the-shell")

        project = _build(
            tmp_path,
            "both-set-project",
            profile_env_text=f"{_A_PROVIDER_KEY}=from-the-profile\n",
        )

        assert _project_env(project)[_A_PROVIDER_KEY] == "from-the-profile"
        assert "from-the-shell" not in (project / ".env").read_text(encoding="utf-8")


class TestAmbientPathVarsDoNotReachTheProject:
    """``PROJECT_ROOT``/``LOCAL_PYTHON_VENV`` are profile-declared or absent.

    Both are generic enough to be exported for unrelated reasons; harvesting
    them pointed a built project at whatever the operator's shell happened to
    hold. The render still carries them as commented build-computed hints.
    """

    def test_exported_project_root_is_not_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PROJECT_ROOT", "/from/shell")

        project = _build(tmp_path, "project-root-project")

        assert "PROJECT_ROOT" not in _project_env(project)
        assert "/from/shell" not in (project / ".env").read_text(encoding="utf-8")

    def test_exported_local_python_venv_is_not_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LOCAL_PYTHON_VENV", "/from/shell/bin/python")

        project = _build(tmp_path, "venv-project")

        assert "LOCAL_PYTHON_VENV" not in _project_env(project)

    def test_profile_declared_path_vars_are_rendered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PROJECT_ROOT", raising=False)

        project = _build(
            tmp_path,
            "declared-path-project",
            profile_env_text="PROJECT_ROOT=/from/the/profile\n",
        )

        assert _project_env(project)["PROJECT_ROOT"] == "/from/the/profile"


class TestWorkingDirectoryIsNotASource:
    """A ``.env`` beside wherever the build ran is not harvested either.

    State the built project cannot reproduce; letting it seed the project made
    a build's output depend on the directory it was launched from.
    """

    def test_cwd_dotenv_key_is_not_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(_A_PROVIDER_KEY, raising=False)
        (tmp_path / ".env").write_text(f"{_A_PROVIDER_KEY}=from-cwd\n", encoding="utf-8")

        project = _build(tmp_path, "cwd-dotenv-project")

        assert _A_PROVIDER_KEY not in _project_env(project)


class TestTimezoneIsNotAnEnvVar:
    """The scaffolded ``.env`` never carries TZ — ``system.timezone`` owns it."""

    def test_shell_tz_is_not_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TZ", "America/Chicago")

        project = _build(tmp_path, "tz-project")

        assert "TZ" not in _project_env(project)
