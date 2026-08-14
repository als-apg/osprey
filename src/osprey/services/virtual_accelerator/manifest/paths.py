"""Filesystem locations for the paradigm channel databases and scenario data.

Every source the manifest generator reads is anchored on ONE data root, so
the same generator serves the bundled control-assistant tree and a facility's
own profile ``data:`` tree without a second code path. :data:`PACKAGE_PATHS`
is that anchor for the bundled tree; ``osprey build`` constructs a
:class:`ManifestPaths` over the tree it is actually building from.

The templates root is discovered from the installed ``osprey.templates``
package (same convention as ``osprey.cli.templates.manager.TemplateManager``)
rather than climbed via a fixed number of ``__file__`` parents, so this works
identically from an editable checkout, a built wheel, or the wheel-drop
context an image build stages -- see docker/virtual-accelerator/Containerfile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import osprey.templates

_TEMPLATES_ROOT = Path(osprey.templates.__file__).parent

_CONTROL_ASSISTANT_DATA = _TEMPLATES_ROOT / "apps" / "control_assistant" / "data"

# Build-resolved default tier for the control-assistant preset. The preset's
# `channel_finder_mode` defaults to "hierarchical"
# (src/osprey/profiles/presets/control-assistant.yml), and
# BuildProfile.resolved_tier() maps that to tier 3 (in_context -> tier 1,
# hierarchical/middle_layer -> tier 3; see src/osprey/cli/build_profile.py).
# All three paradigm DBs are address-identical at tier 3 (verified by
# build.build_manifest()'s cross-paradigm check), so tier 3 is the default
# tier this generator expands.
DEFAULT_TIER = 3


@dataclass(frozen=True)
class ManifestPaths:
    """Every manifest source file, resolved against one facility data tree.

    ``data_root`` is a project/preset ``data/`` directory: the bundled
    ``apps/<bundle>/data`` tree, or the tree a build profile points at with
    its ``data:`` key. ``tier`` selects which ``channel_databases/tiers/``
    subdirectory the three paradigm DBs are read from -- the same
    build-resolved tier ``osprey build`` materializes into the project.
    """

    data_root: Path
    tier: int = DEFAULT_TIER

    @property
    def tier_dir(self) -> Path:
        return self.data_root / "channel_databases" / "tiers" / f"tier{self.tier}"

    @property
    def hierarchical_db(self) -> Path:
        return self.tier_dir / "hierarchical.json"

    @property
    def in_context_db(self) -> Path:
        return self.tier_dir / "in_context.json"

    @property
    def middle_layer_db(self) -> Path:
        return self.tier_dir / "middle_layer.json"

    @property
    def machine_json(self) -> Path:
        return self.data_root / "simulation" / "machine.json"

    @property
    def machine_state_channels(self) -> Path:
        return self.data_root / "machine_state_channels.json"

    @property
    def channel_limits(self) -> Path:
        return self.data_root / "channel_limits.json"

    @property
    def required_sources(self) -> tuple[Path, ...]:
        """Every file :func:`build.build_manifest` reads, in read order.

        The one list: both the presence check below and the failure-path probe
        that names a corrupt file (``build._first_unreadable_source``) walk it,
        so a source added here cannot be missed by either. ``channel_limits``
        is deliberately absent -- the generator never reads it; the build step
        requires it separately, as the file that must ship *beside* a manifest.

        All THREE paradigm DBs are required at any tier, not just the one the
        project's ``channel_finder_mode`` selects: the generator's premise is
        that they describe one namespace, and ``build.build_manifest`` checks
        exactly that. Tier 1 ships only ``in_context``, so a tier-1 tree
        always reports the other two missing and falls back.
        """
        return (
            self.hierarchical_db,
            self.in_context_db,
            self.middle_layer_db,
            self.machine_json,
            self.machine_state_channels,
        )

    def missing_sources(self) -> list[Path]:
        """Return the :attr:`required_sources` that are absent.

        An empty list means the tree can back a generated manifest. A
        non-empty one is the build-time skip signal: a facility tree that
        ships no paradigm DBs (only the control-assistant bundle does) or no
        scenario seed is not a defect, it just doesn't drive a virtual
        accelerator -- see ``build.prepare_project_manifest``.
        """
        return [path for path in self.required_sources if not path.is_file()]


# The bundled control-assistant tree: the container-runtime source, and the
# default for every loader in this package.
PACKAGE_PATHS = ManifestPaths(data_root=_CONTROL_ASSISTANT_DATA, tier=DEFAULT_TIER)

MANIFEST_OUTPUT = Path(__file__).resolve().parent / "channel_manifest.json"
