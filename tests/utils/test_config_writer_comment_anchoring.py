"""Section comments must stay anchored when config mutations append entries.

ruamel.yaml attaches a comment block that sits between two sections to the
deepest last node of the PRECEDING section. A plain append to that section's
list or mapping therefore lands *after* the comment — splitting a list in two
around a section banner, or emitting new service blocks below the header of
the next section. These tests pin the corrected behavior: appended entries
render inside their section and the trailing comment block stays at the
section boundary.
"""

from __future__ import annotations

from ruamel.yaml import YAML

from osprey.utils.config_writer import (
    anchored_append,
    anchored_put,
    config_add_to_list,
    config_update_fields,
    update_yaml_file,
)

SAMPLE = """\
claude_code:
  default_model: haiku
  timeout: 300

# ============================================================
# WEB PANEL CONFIGURATION
# ============================================================
# Prose about panels.

web:
  panels:
    ariel:
      enabled: true
    # trailing panel docs

services:
  postgresql:
    path: ./services/postgresql
  openobserve:
    path: ./services/openobserve
    port: 5080          # Host port

# Services to deploy with `osprey up`
deployed_services:
  - postgresql
  - openobserve

# ============================================================
# SAFETY CONTROLS
# ============================================================
approval:
  enabled: true
"""
# NOTE: list items are indented under their key, which is what the shared
# writer emits (`_yaml.indent(sequence=4, offset=2)`) and how every config this
# framework generates is written. Matching it here keeps the full-text
# comparison below about the helpers under test rather than about
# re-indentation, which is round-trip behavior they do not control.


def _write_sample(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def _line_no(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines()):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found in:\n{text}")


def _load(path):
    yaml = YAML(typ="safe")
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh)


# ---------------------------------------------------------------------------
# Node-level helpers
# ---------------------------------------------------------------------------


class TestAnchoredAppend:
    def test_appended_items_render_before_section_banner(self, tmp_path):
        path = _write_sample(tmp_path)
        yaml = YAML()
        config = yaml.load(path.read_text(encoding="utf-8"))

        anchored_append(config["deployed_services"], "virtual_accelerator")
        anchored_append(config["deployed_services"], "bluesky")

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        text = path.read_text(encoding="utf-8")

        assert _line_no(text, "- bluesky") < _line_no(text, "# SAFETY CONTROLS")
        assert _line_no(text, "- virtual_accelerator") < _line_no(text, "- bluesky")
        assert _load(path)["deployed_services"] == [
            "postgresql",
            "openobserve",
            "virtual_accelerator",
            "bluesky",
        ]

    def test_plain_list_still_appends(self, tmp_path):
        lst = ["a"]
        anchored_append(lst, "b")
        assert lst == ["a", "b"]


class TestAnchoredPut:
    def test_new_service_block_renders_inside_section(self, tmp_path):
        path = _write_sample(tmp_path)
        yaml = YAML()
        config = yaml.load(path.read_text(encoding="utf-8"))

        anchored_put(
            config["services"],
            "virtual_accelerator",
            {"path": "./services/virtual_accelerator", "port": 5064},
        )

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        text = path.read_text(encoding="utf-8")

        # New block sits under services:, before the deployed_services comment.
        assert _line_no(text, "virtual_accelerator:") < _line_no(text, "# Services to deploy")
        # The inline comment on the previous last entry survives in place.
        assert "port: 5080          # Host port" in text
        assert _load(path)["services"]["virtual_accelerator"] == {
            "path": "./services/virtual_accelerator",
            "port": 5064,
        }

    def test_split_keeps_blank_line_before_moved_comment(self, tmp_path):
        path = _write_sample(tmp_path)
        yaml = YAML()
        config = yaml.load(path.read_text(encoding="utf-8"))

        anchored_put(config["services"], "va", {"port": 5064})

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        lines = path.read_text(encoding="utf-8").splitlines()

        comment_line = _line_no("\n".join(lines), "# Services to deploy")
        assert lines[comment_line - 1].strip() == ""

    def test_existing_key_update_leaves_layout_untouched(self, tmp_path):
        path = _write_sample(tmp_path)
        yaml = YAML()
        config = yaml.load(path.read_text(encoding="utf-8"))

        anchored_put(config["services"], "postgresql", {"path": "./services/postgresql"})

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        text = path.read_text(encoding="utf-8")

        assert _line_no(text, "# Services to deploy") < _line_no(text, "deployed_services:")
        assert "virtual_accelerator" not in text

    def test_nested_panel_addition_stays_inside_panels(self, tmp_path):
        path = _write_sample(tmp_path)
        yaml = YAML()
        config = yaml.load(path.read_text(encoding="utf-8"))

        anchored_put(
            config["web"]["panels"],
            "events",
            {"url": "http://localhost:8020", "path": "/dashboard"},
        )

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        text = path.read_text(encoding="utf-8")

        assert _line_no(text, "events:") < _line_no(text, "# trailing panel docs")
        assert _line_no(text, "# trailing panel docs") < _line_no(text, "services:")


# ---------------------------------------------------------------------------
# File-level mutators
# ---------------------------------------------------------------------------


class TestConfigUpdateFields:
    def test_new_nested_key_renders_before_next_banner(self, tmp_path):
        path = _write_sample(tmp_path)

        config_update_fields(path, {"claude_code.servers.bluesky.enabled": True})

        text = path.read_text(encoding="utf-8")
        assert _line_no(text, "servers:") < _line_no(text, "# WEB PANEL CONFIGURATION")
        assert _load(path)["claude_code"]["servers"]["bluesky"]["enabled"] is True

    def test_new_root_key_still_appends_at_tail(self, tmp_path):
        # Top-level appends are intentional tail appends (the template
        # documents appended sections at the file tail, below their banner).
        path = _write_sample(tmp_path)

        config_update_fields(path, {"facility.prefix": "ca"})

        text = path.read_text(encoding="utf-8")
        assert _line_no(text, "# SAFETY CONTROLS") < _line_no(text, "facility:")
        assert text.rstrip().endswith("prefix: ca")

    def test_existing_scalar_update_is_layout_neutral(self, tmp_path):
        path = _write_sample(tmp_path)
        before = path.read_text(encoding="utf-8")

        config_update_fields(path, {"claude_code.default_model": "sonnet"})

        after = path.read_text(encoding="utf-8")
        assert after.replace("sonnet", "haiku") == before


class TestUpdateYamlFile:
    def test_nested_dict_form_new_key_stays_in_section(self, tmp_path):
        path = _write_sample(tmp_path)

        update_yaml_file(
            path,
            {"claude_code": {"servers": {"health": {"enabled": True}}}},
            create_backup=False,
        )

        text = path.read_text(encoding="utf-8")
        assert _line_no(text, "servers:") < _line_no(text, "# WEB PANEL CONFIGURATION")
        assert _load(path)["claude_code"]["servers"]["health"]["enabled"] is True


class TestConfigAddToList:
    def test_appended_item_renders_before_section_banner(self, tmp_path):
        path = _write_sample(tmp_path)

        config_add_to_list(path, ["deployed_services"], "virtual_accelerator")

        text = path.read_text(encoding="utf-8")
        assert _line_no(text, "- virtual_accelerator") < _line_no(text, "# SAFETY CONTROLS")
