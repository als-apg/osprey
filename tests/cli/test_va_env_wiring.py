"""Tests the ``.env`` wiring that points the VA container at a built manifest.

``VA_CHANNELS_FILE`` alone would silently disable the PyAT physics bridge (the
container entrypoint defaults the lattice to ``none`` for a file-backed channel
source), so the two keys are written together or not at all. Both are relative
to the ``data/simulation`` directory the compose file already mounts, which is
why this feature needs no compose or entrypoint change.
"""

from __future__ import annotations

from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.scaffolding import provider_api_key_entries
from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
from osprey.utils.dotenv import BUILD_DERIVED_KEYS, parse_dotenv_file


def _render(ctx: dict) -> str:
    manager = TemplateManager()
    return manager.jinja_env.get_template("project/env.j2").render(**ctx)


def _ctx(**extra) -> dict:
    base = {
        "project_name": "test-project",
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "env": {"CBORG_API_KEY": "x"},
        "provider_api_keys": provider_api_key_entries(),
    }
    base.update(extra)
    return base


def _parsed(tmp_path, text) -> dict[str, str]:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return parse_dotenv_file(path)


class TestVirtualAcceleratorEnvKeys:
    def test_absent_without_a_generated_manifest(self, tmp_path):
        parsed = _parsed(tmp_path, _render(_ctx()))

        assert "VA_CHANNELS_FILE" not in parsed
        assert "VA_LATTICE" not in parsed

    def test_both_keys_written_when_the_build_generated_one(self, tmp_path):
        parsed = _parsed(
            tmp_path,
            _render(_ctx(va_channels_file=MANIFEST_FILENAME, va_lattice="builtin")),
        )

        # A bare filename: the entrypoint resolves it against the mounted
        # data dir, so no compose change is needed to reach it.
        assert parsed["VA_CHANNELS_FILE"] == MANIFEST_FILENAME
        assert "/" not in parsed["VA_CHANNELS_FILE"]
        # Without this the physics bridge would not be built at all.
        assert parsed["VA_LATTICE"] == "builtin"

    def test_provider_keys_still_render_alongside(self, tmp_path):
        parsed = _parsed(
            tmp_path,
            _render(_ctx(va_channels_file=MANIFEST_FILENAME, va_lattice="builtin")),
        )

        assert parsed["CBORG_API_KEY"] == "x"

    def test_written_keys_are_the_declared_build_derived_set(self, tmp_path):
        with_va = _parsed(
            tmp_path,
            _render(_ctx(va_channels_file=MANIFEST_FILENAME, va_lattice="builtin")),
        )
        without_va = _parsed(tmp_path, _render(_ctx()))

        # Whatever the template adds must be exactly what the merge exempts,
        # or a skipped build could not un-write it.
        assert with_va.keys() - without_va.keys() == BUILD_DERIVED_KEYS
