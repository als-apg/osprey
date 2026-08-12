"""Service injectors must not displace config.yml section comments.

Regression tests for the rendered-config corruption where injector appends to
``services:`` / ``deployed_services`` landed *after* the comment block that
separates the SERVICE CONFIGURATION section from SAFETY CONTROLS — splitting
the ``deployed_services`` list in two around the banner.
"""

from __future__ import annotations

import yaml as pyyaml

from osprey.cli.build_injectors import _inject_va
from osprey.cli.build_profile_schema import VAConfig

CONFIG_TEMPLATE = """\
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

# Approval workflow for sensitive operations
approval:
  enabled: true
"""


def _line_no(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines()):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found in:\n{text}")


def test_inject_va_keeps_section_banner_after_deployed_services(tmp_path):
    (tmp_path / "config.yml").write_text(CONFIG_TEMPLATE, encoding="utf-8")

    _inject_va(VAConfig(port=5064), tmp_path)

    text = (tmp_path / "config.yml").read_text(encoding="utf-8")

    # The appended list item renders inside the list, before the banner.
    assert _line_no(text, "- virtual_accelerator") < _line_no(text, "# SAFETY CONTROLS")
    # The new service block renders inside services:, before the list comment.
    assert _line_no(text, "  virtual_accelerator:") < _line_no(text, "# Services to deploy")
    # The banner still directly precedes the approval section.
    assert _line_no(text, "# SAFETY CONTROLS") < _line_no(text, "approval:")
    assert _line_no(text, "# Approval workflow") < _line_no(text, "approval:")

    config = pyyaml.safe_load(text)
    assert config["deployed_services"] == ["postgresql", "openobserve", "virtual_accelerator"]
    assert config["services"]["virtual_accelerator"] == {
        "path": "./services/virtual_accelerator",
        "port": 5064,
    }
