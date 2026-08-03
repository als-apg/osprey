"""Tests for detect_environment_variables() in the scaffolding module."""

from osprey.cli.templates.scaffolding import detect_environment_variables


class TestTimezoneIsNotAnEnvVar:
    """The scaffolded ``.env`` never carries TZ — ``system.timezone`` owns it."""

    def test_shell_tz_is_not_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TZ", "America/Chicago")

        assert "TZ" not in detect_environment_variables()

    def test_dotenv_tz_is_not_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TZ", raising=False)
        (tmp_path / ".env").write_text("TZ=America/Chicago\n", encoding="utf-8")

        assert "TZ" not in detect_environment_variables()

    def test_documented_vars_still_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

        assert detect_environment_variables()["PROJECT_ROOT"] == str(tmp_path)
