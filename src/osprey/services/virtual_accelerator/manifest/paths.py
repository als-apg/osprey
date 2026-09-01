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
# every other mode -> tier 3; see src/osprey/cli/build_profile.py).
# All three file-backed paradigm DBs are address-identical at tier 3 (verified
# by build.build_manifest()'s cross-paradigm check), so tier 3 is the default
# tier this generator expands. The `graph` mode has no tier file at all, so it
# adds no fourth DB here.
DEFAULT_TIER = 3


@dataclass(frozen=True)
class ManifestPaths:
    """Every manifest source file, resolved against one facility data tree.

    ``data_root`` is a project/preset ``data/`` directory: the bundled
    ``apps/<bundle>/data`` tree, or the tree a build profile points at with
    its ``data:`` key. ``tier`` selects which ``channel_databases/tiers/``
    subdirectory the file-backed paradigm DBs are read from -- the same
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
    def paradigm_databases(self) -> dict[str, Path]:
        """Each file-backed paradigm database, in the order the generator reads them.

        The one list of paradigms this generator knows, so a paradigm added to
        the framework is added here and reaches the staged/absent split, the
        read order and the agreement gate together. ``graph`` is absent on
        purpose: its store is seeded from the facility corpus TTL and it ships
        no tier database, so it stages nothing here.
        """
        return {
            "hierarchical": self.hierarchical_db,
            "in_context": self.in_context_db,
            "middle_layer": self.middle_layer_db,
        }

    @property
    def staged_paradigms(self) -> tuple[str, ...]:
        """The paradigm databases this tree carries at :attr:`tier`.

        Any non-empty subset backs a manifest, built from exactly these. A
        project's namespace is whatever it staged: tier 1 ships ``in_context``
        alone, a facility may stage only the paradigm its
        ``channel_finder_mode`` selects, and the bundle ships all three. What
        the subset costs is checking -- the cross-paradigm agreement gate can
        only compare the databases that are here -- and, without
        ``hierarchical``, the identity keys that pair a setpoint with its
        readback, since it is the one paradigm that declares a hierarchy path.
        An EMPTY tuple is the case with no database answer: the tree stages
        none, and a build deploying a virtual accelerator refuses -- unless the
        project's roster source is the knowledge-graph corpus, which
        ``build.prepare_project_manifest`` consults exactly then. This asks
        whether the FILE is there, not whether it names anything; a staged
        database that expands to nothing is caught one layer up, by
        ``build.prepare_project_manifest``, and refuses the same way.
        """
        return tuple(name for name, path in self.paradigm_databases.items() if path.is_file())

    @property
    def absent_paradigms(self) -> tuple[str, ...]:
        """The paradigm databases this tree does not stage at :attr:`tier`.

        Not a defect, and not an error: the build names them so an operator
        reads which databases fed the channel set their accelerator serves.
        """
        return tuple(name for name, path in self.paradigm_databases.items() if not path.is_file())

    @property
    def required_sources(self) -> tuple[Path, ...]:
        """Every file :func:`build.build_manifest` reads, in read order.

        The one list: both the presence check below and the failure-path probe
        that names a corrupt file (``build._first_unreadable_source``) walk it,
        so a source added here cannot be missed by either. ``channel_limits``
        is deliberately absent -- the generator never reads it; the build step
        requires it separately, as the file that must ship *beside* a manifest.

        Only the paradigm databases the tree STAGES are here, because those are
        the ones the generator reads. The scenario seed and the machine-state
        list are required outright: they are one per tree rather than one per
        paradigm, so a tree naming channels while missing them is incomplete
        rather than partial.
        """
        return (
            *(self.paradigm_databases[name] for name in self.staged_paradigms),
            self.machine_json,
            self.machine_state_channels,
        )

    def missing_sources(self) -> list[Path]:
        """Return the :attr:`required_sources` that are absent.

        An empty list means the tree can back a generated manifest from
        whatever it staged. A non-empty one names files a build has to be
        given: with a virtual accelerator deployed the build refuses on them
        by name, because the alternative -- serving the framework's built-in
        demo namespace under the project's own name -- is the thing this must
        never do. Absent paradigm databases are NOT reported here; see
        :attr:`absent_paradigms`.
        """
        return [path for path in self.required_sources if not path.is_file()]


# The bundled control-assistant tree: the container-runtime source, and the
# default for every loader in this package.
PACKAGE_PATHS = ManifestPaths(data_root=_CONTROL_ASSISTANT_DATA, tier=DEFAULT_TIER)

MANIFEST_OUTPUT = Path(__file__).resolve().parent / "channel_manifest.json"
