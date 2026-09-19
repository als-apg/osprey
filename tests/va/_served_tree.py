"""One deployment tree, built by a real ``osprey mml`` run over a fixture.

Every other module in this suite hands the model a tree assembled by the test
that reads it. The modules that import this one do not: they drive the whole
chain -- import, map, emit -- over a committed 2.0 export and then serve what
that run wrote, so what they pin is the seam between the two halves of this
repository rather than a fixture author's idea of it.

The run is expensive enough to want once per module, so the helper takes the
directory to build in and each caller wraps it in a module-scoped fixture of
its own. It lives here, outside any test module, because a tree built by the
chain is not the property of whichever test happened to need it first.

The export describes an invented machine and every family name in it is
invented too; nothing here spells one.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.services.virtual_accelerator.manifest.paths import DEFAULT_TIER
from tests.cli.test_mml_map import _fill

__all__ = ["SYNTHETIC_EXPORT", "emit_served_tree"]

#: The 2.0 export the chain is run over: an invented ring small enough to read
#: by eye, covering every binding kind and both slice shapes.
SYNTHETIC_EXPORT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "mml" / "synthetic" / "quokka.sr.ao.json"
)


def emit_served_tree(root: Path) -> Path:
    """Run the export chain into *root* and return the served data directory.

    The deployment repository is built the way a facility builds one: the
    export is imported, a mapping skeleton is written and filled, and ``emit``
    turns the checked mapping into the artifacts a deployment reads. The tier
    directory is staged first, so the channel database lands where the
    manifest generator reads it and the manifest is the one this tree's own
    namespace produces rather than another tree's.

    Args:
        root: An empty directory to build the deployment repository in.

    Returns:
        The repository's ``data/`` directory -- what a served container is
        handed, and what the served model is pointed at.
    """
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    (root / "data" / "channel_databases" / "tiers" / f"tier{DEFAULT_TIER}").mkdir(parents=True)

    runner = CliRunner()
    previous = Path.cwd()
    os.chdir(root)
    try:
        imported = runner.invoke(
            cli, ["mml", "import", str(SYNTHETIC_EXPORT)], catch_exceptions=False
        )
        assert imported.exit_code == 0, imported.output
        init = runner.invoke(cli, ["mml", "map", "--init"], catch_exceptions=False)
        assert init.exit_code == 0, init.output

        mapping = root / "data" / "mml" / "mapping.yaml"
        mapping.write_text(
            yaml.safe_dump(
                _fill(yaml.safe_load(mapping.read_text(encoding="utf-8"))), sort_keys=False
            ),
            encoding="utf-8",
        )

        emitted = runner.invoke(cli, ["mml", "emit"], catch_exceptions=False)
        assert emitted.exit_code == 0, emitted.output
    finally:
        os.chdir(previous)
    return root / "data"
