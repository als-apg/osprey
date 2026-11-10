"""Tests for the config backup scheme: where a backup lands, and that it is taken."""

from osprey.utils.config_writer import (
    CONFIG_BACKUP_DIRNAME,
    config_backup_path,
    write_config_backup,
)
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR


class TestConfigBackup:
    """Backups land in the agent-data state zone, anchored on the repo root."""

    def test_creates_backup(self, tmp_path):
        """The backup is a byte copy of the config, taken into the state zone.

        The backup lands in the agent-data state zone, not beside the config.
        The zone is anchored on the deployment repo the config sits in, which
        for a flat project is the config's own directory.
        """
        config_file = tmp_path / "config.yml"
        original_content = "project_name: original\n"
        config_file.write_text(original_content)

        backup_path = write_config_backup(config_file)

        assert backup_path.exists()
        assert backup_path.read_text() == original_content
        assert backup_path == config_backup_path(config_file)
        assert backup_path.parent.name == CONFIG_BACKUP_DIRNAME
        # The point of the move: nothing new appears next to the config itself.
        assert not (tmp_path / "config.yml.bak").exists()
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".bak"] == []

    def test_backup_anchors_on_the_repo_root_not_the_render(self, tmp_path):
        """The anchor is the repo, not the directory config.yml happens to be in.

        ``agent_data.base_dir`` is rendered and documented relative to the
        deployment repo ROOT -- the tree holding ``profile.yml`` and the durable
        ``var/`` -- while ``config.yml`` lives one level down in the render that
        every build re-creates. Anchoring on the config's own parent would put
        the backup inside that disposable zone, and inside a container it would
        put it inside the ROOT-OWNED one: the mkdir of ``<render>/var`` raises
        ``PermissionError`` before a byte of the save is written. This is the
        real container layout, ``/app/<project>/build/config.yml``, with the
        state zone the image creates and chowns one level up.
        """
        repo = tmp_path / "app" / "psplit"
        render = repo / "build"
        render.mkdir(parents=True)
        config_file = render / "config.yml"
        original_content = "project_name: original\n"
        config_file.write_text(original_content)

        backup_path = write_config_backup(config_file)

        assert (
            backup_path
            == repo / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )
        assert backup_path.read_text() == original_content
        # Nothing at all was created in the render zone.
        assert not (render / "var").exists()
        assert [p.name for p in render.iterdir()] == ["config.yml"]

    def test_backup_of_a_relocated_render_config_still_follows_the_config(self, tmp_path):
        """Both rules at once: repo-anchored, and the repo's own zone is read.

        A render-zone config that relocates ``agent_data.base_dir`` to a
        relative path must resolve it against the repo root as well -- the pair
        a caller could get individually right and jointly wrong.
        """
        repo = tmp_path / "psplit"
        render = repo / "build"
        render.mkdir(parents=True)
        config_file = render / "config.yml"
        config_file.write_text("agent_data:\n  base_dir: state/data\n", encoding="utf-8")

        backup_path = config_backup_path(config_file)

        assert backup_path == repo / "state" / "data" / CONFIG_BACKUP_DIRNAME / "config.yml.bak"

    def test_backup_follows_a_relocated_agent_data_root(self, tmp_path):
        """The zone is read from the config being written, never assumed.

        The case a hard-coded ``var/agent_data`` would pass and this scheme must
        not: the root is moved somewhere no default would guess, and the backup
        has to go with it. Read off the file itself because a backup has no other
        source of truth for the project it is copying.
        """
        relocated = tmp_path / "elsewhere" / "state"
        config_file = tmp_path / "config.yml"
        original_content = f"agent_data:\n  base_dir: {relocated}\nproject_name: original\n"
        config_file.write_text(original_content)

        backup_path = write_config_backup(config_file)

        assert backup_path == relocated / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        assert backup_path.read_text() == original_content
        assert not (tmp_path / "var").exists()

    def test_backup_follows_the_configs_own_project_root_key(self, tmp_path):
        """The anchor mirrors ``resolve_project_root``, key first.

        The shape the two rules disagree on, and it is a real one: a project
        directory literally NAMED ``build`` reads as a render to
        ``repo_root_for_config`` — its documented non-goal — so the tail rule
        alone would put the backup one level ABOVE the project while every
        runtime reader of the state zone looked inside it. The config's own
        ``project_root`` key is what settles it, exactly as it does at runtime.
        """
        project = tmp_path / "build"
        project.mkdir()
        config_file = project / "config.yml"
        config_file.write_text(f"project_root: {project}\n", encoding="utf-8")

        backup_path = config_backup_path(config_file)

        assert (
            backup_path
            == project / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )
        assert not (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).exists()

    def test_backup_ignores_a_project_root_that_is_not_here(self, tmp_path):
        """A build recorded on another machine names a path that is not here.

        Same qualifier ``resolve_project_root`` applies: the key wins only when
        it names a directory that exists, so a container reading a config staged
        on the host does not anchor its state zone on the host's path.
        """
        repo = tmp_path / "psplit"
        render = repo / "build"
        render.mkdir(parents=True)
        config_file = render / "config.yml"
        config_file.write_text("project_root: /build/machine/psplit\n", encoding="utf-8")

        assert (
            config_backup_path(config_file)
            == repo / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )

    def test_backup_of_an_unreadable_config_uses_the_default_zone(self, tmp_path):
        """A config too broken to name its own zone is the one most worth copying."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system: [unterminated\n")

        backup_path = config_backup_path(config_file)

        assert (
            backup_path
            == tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / CONFIG_BACKUP_DIRNAME / "config.yml.bak"
        )
