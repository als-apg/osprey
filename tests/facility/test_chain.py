"""The whole chain on a two-model tree: every layer, the build, every view.

One scratch deployment takes both exports of the committed NSLS-II tree in one
``osprey facility import mml`` call, under the tree's reviewed mapping, builds
the facility file from what the import left under ``data/facility/`` and
renders it into ``build/``, twice over. Everything asserted here reads that one
chain, so what is pinned is one facility installed the way a facility is
installed:

* each export becomes one model, with its deck and its response matrix filed
  under that model's name, and the two decks are two lattices;
* both layers the tree has -- the importer's records and the files the import
  seeds beside them -- reach the facility file;
* a second pass changes no byte of any source and no byte of any view;
* every view is written, each holding the facility the others hold: the
  simulator serves every channel and wires each model to channels of its own,
  the limits view holds the limits records, and the channel-finder index holds
  every channel.

The import's first check stands in front of that chain: a deck is held to the
lattice fingerprint its export states, so an export beside the deck of another
lattice is refused by the fact the two disagree on, and one that differs in
its stated energy alone imports and says so.

The chain ends at the views. Its other half, the virtual accelerator serving
every channel of both models from a container, is
``tests/va/e2e/test_chain_boot.py``, which runs :func:`run_chain` under the
config a served render carries.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from tests.facility._mml_built import FIXTURES, export_files

# One worker builds the chain: the module-scoped fixture is one chain per
# worker process, and under ``--dist loadgroup`` this mark keeps every test
# that reads it on one of them.
pytestmark = pytest.mark.xdist_group("facility_chain")

#: The committed two-model tree.
TREE = "nsls2"

#: The model each export of the tree becomes. Listed, not discovered: which
#: model an export is imported as is a fact under test.
MODELS = ("LTB", "StorageRing")

#: The beam energy each model's deck states, in eV.
DECK_ENERGY = {"LTB": 0.2e9, "StorageRing": 3.0e9}

#: The scratch deployment's name: its ``profile.yml`` name and the project name
#: the facility file is built under.
PROJECT = "scratch"

#: The rendered config every render of this module shares: the simulated
#: target with every setpoint writable, and one Bluesky lane.
_SHARED: dict[str, Any] = {
    "control_system": {
        "type": "virtual_accelerator",
        "limits_checking": {"enabled": True, "mode": "optional"},
    },
    "services": {"bluesky": {}},
}

#: The config the chain's own render carries, and the one the middle-layer
#: index is rendered under.
HIERARCHICAL: dict[str, Any] = {**_SHARED, "channel_finder": {"pipeline_mode": "hierarchical"}}
MIDDLE_LAYER: dict[str, Any] = {**_SHARED, "channel_finder": {"pipeline_mode": "middle_layer"}}
IN_CONTEXT: dict[str, Any] = {**_SHARED, "channel_finder": {"pipeline_mode": "in_context"}}

#: The files each view writes for this tree, relative to a render's ``data/``.
VIEW_FILES: dict[str, tuple[str, ...]] = {
    "simulator": (
        "simulator/addresses.json",
        "simulator/decks/LTB.json",
        "simulator/decks/StorageRing.json",
        "simulator/scenarios.json",
        "simulator/seeds.json",
        "simulator/served_models.json",
        "simulator/variables.json",
    ),
    "limits": ("channel_limits.json",),
    "facts": ("facility_facts.json", "facility_facts.md"),
    "bluesky": ("bluesky_devices.yml",),
    "hierarchical": ("channel_finder/hierarchical.json",),
    "middle_layer": (
        "channel_finder/middle_layer.duckdb",
        "channel_finder/middle_layer.json",
    ),
    "graph": ("graph/facility.ttl",),
}

#: The one view this tree cannot carry: it tags no channel ``in_context``.
UNSUPPORTED_VIEW = "in_context"


# ===================================================================
# Running the chain
# ===================================================================


@dataclass(frozen=True)
class Pass:
    """What one pass of the chain left behind.

    Attributes:
        sources: Every file under ``data/facility/``, keyed by its path
            relative to the deployment.
        rendered: Every file of the render, keyed by its path relative to the
            render.
    """

    sources: dict[str, bytes]
    rendered: dict[str, bytes]


@dataclass(frozen=True)
class Chain:
    """The tree installed end to end, and everything read back off it.

    Attributes:
        root: The scratch deployment the chain ran in.
        document: The facility file the last pass built.
        passes: One :class:`Pass` per run of the chain, in order.
    """

    root: Path
    document: dict[str, Any]
    passes: tuple[Pass, ...]

    @property
    def facility(self) -> Path:
        """The deployment's ``data/facility`` directory."""
        return self.root / "data" / "facility"

    @property
    def build(self) -> Path:
        """The render the last pass wrote."""
        return self.root / "build"

    def view(self, name: str) -> Any:
        """One JSON file of the render's ``data/``, parsed."""
        return json.loads((self.build / "data" / name).read_text(encoding="utf-8"))


def _files(directory: Path, root: Path) -> dict[str, bytes]:
    """Every file under ``directory``, keyed by its path relative to ``root``."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run_chain(root: Path, config: Mapping[str, Any], *, passes: int = 2) -> Chain:
    """Install the two-model tree into a scratch deployment, ``passes`` times over.

    Each pass imports both exports in one ``osprey facility import mml`` call,
    builds the facility file from ``data/facility/`` and renders it into
    ``build/`` under ``config``, over whatever the pass before left there.

    The reviewed mapping is installed once, before the first pass. An import
    may rewrite the mapping it reads -- a ``facility:`` block moves into
    ``identity.yaml`` -- so the installed file is the deployment's from the
    first import on, and every later pass imports under that file, never under
    a fresh copy of the fixture's.

    Args:
        root: The deployment to build in. Created if absent.
        config: The rendered config the render is written under.
        passes: How many times to run the whole chain.

    Returns:
        The finished chain, with one :class:`Pass` per run.
    """
    from osprey.cli.main import cli
    from osprey.facility.build import build_facility
    from osprey.facility.layers.mml.mapping import MAPPING_FILE
    from osprey.facility.render import render_facility_outputs

    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text(f"name: {PROJECT}\ndata: data\n", encoding="utf-8")
    facility = root / "data" / "facility"
    installed = facility / MAPPING_FILE
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / TREE / MAPPING_FILE, installed)
    exports = [str(path) for path in export_files(TREE)]
    render = root / "build"
    render.mkdir(exist_ok=True)

    document: dict[str, Any] = {}
    records: list[Pass] = []
    for _ in range(passes):
        result = CliRunner().invoke(
            cli,
            ["facility", "import", "mml", *exports, "--repo", str(root)],
            catch_exceptions=False,
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 0, f"osprey facility import mml:\n{result.output}"
        document = build_facility(facility, project_name=PROJECT)
        render_facility_outputs(render, document, config, facility)
        records.append(Pass(sources=_files(facility, root), rendered=_files(render, render)))
    return Chain(root=root, document=document, passes=tuple(records))


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Chain:
    """The tree installed end to end, once, and shared by every assertion."""
    return run_chain(tmp_path_factory.mktemp("chain") / PROJECT, HIERARCHICAL)


@pytest.fixture(scope="module")
def middle_layer(chain: Chain, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The chain's facility file rendered once more, under the middle-layer index."""
    from osprey.facility.render import render_facility_outputs

    render = tmp_path_factory.mktemp("chain-middle-layer")
    render_facility_outputs(render, chain.document, MIDDLE_LAYER, chain.facility)
    return render


def _layer(chain: Chain) -> Path:
    """The directory the import wrote its records under."""
    from osprey.facility.layers.mml.importer import LAYER_DIR

    return chain.facility / LAYER_DIR


def _physics(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every model of a document that has a deck behind it, by name."""
    return {str(model["name"]): model for model in document["models"] if model.get("deck")}


# ===================================================================
# The layers
# ===================================================================


class TestTheLayers:
    def test_each_export_becomes_one_model(self, chain: Chain) -> None:
        """Both exports go in one call, and each is a model of the facility file."""
        models = yaml.safe_load((_layer(chain) / "models.yaml").read_text(encoding="utf-8"))

        assert sorted(model["name"] for model in models) == list(MODELS)
        assert sorted(_physics(chain.document)) == list(MODELS)

    def test_the_siblings_land_under_their_own_models(self, chain: Chain) -> None:
        """The deck and the response matrix beside each export are filed by model.

        The siblings are never named on the command line: the import pairs
        them with the export they sit beside.
        """
        layer = _layer(chain)

        assert sorted(path.stem for path in (layer / "decks").glob("*.json")) == list(MODELS)
        assert sorted(path.name for path in layer.glob("*.response.json")) == [
            f"{model}.response.json" for model in MODELS
        ]

    def test_the_two_models_are_two_lattices(self, chain: Chain) -> None:
        """Each model's deck is its own export's: another energy, another lattice."""
        decks = {
            name: json.loads((chain.facility / str(model["deck"])).read_text(encoding="utf-8"))
            for name, model in _physics(chain.document).items()
        }

        assert {name: deck["properties"]["energy"] for name, deck in decks.items()} == DECK_ENERGY
        assert len({len(deck["elements"]) for deck in decks.values()}) == len(MODELS)

    def test_every_layer_of_the_tree_reaches_the_facility_file(self, chain: Chain) -> None:
        """The importer's records and the files seeded beside them both merge.

        The layers are read off the tree rather than listed: each directory
        under ``imported/`` is a layer, and the files the import seeds at the
        top of ``data/facility/`` merge as the authored one. Every one of them
        is named by the provenance of some record.
        """
        from osprey.facility.sources import AUTHORED

        on_disk = {path.name for path in (chain.facility / "imported").iterdir() if path.is_dir()}
        on_disk.add(AUTHORED)
        merged = {
            str(source["layer"])
            for kind in ("places", "devices", "channels", "groups", "models")
            for record in chain.document[kind]
            for source in (record.get("provenance") or {}).get("sources", [])
        }

        assert len(on_disk) > 1
        assert merged == on_disk

    def test_a_second_pass_changes_no_source_byte(self, chain: Chain) -> None:
        first, second = chain.passes

        assert first.sources
        assert second.sources == first.sources


# ===================================================================
# The views
# ===================================================================


class TestTheViews:
    def test_the_view_table_names_every_view(self) -> None:
        """Every view the build has is one this module accounts for."""
        from osprey.facility import views

        assert sorted(view.name for view in views.VIEWS) == sorted([*VIEW_FILES, UNSUPPORTED_VIEW])

    def test_every_view_of_the_render_is_written(self, chain: Chain) -> None:
        """The chain's render holds the facility file and each of its views, no more."""
        carried = [name for name in VIEW_FILES if name != "middle_layer"]
        expected = {f"data/{file}" for name in carried for file in VIEW_FILES[name]}

        assert set(chain.passes[-1].rendered) == {"facility.json", *expected}

    def test_the_other_index_is_written_in_its_own_render(self, middle_layer: Path) -> None:
        """Selecting the middle-layer index swaps one view and leaves the rest."""
        carried = [name for name in VIEW_FILES if name != "hierarchical"]
        expected = {f"data/{file}" for name in carried for file in VIEW_FILES[name]}

        assert set(_files(middle_layer, middle_layer)) == {"facility.json", *expected}

    def test_the_in_context_index_stops_on_a_tree_that_tags_no_channel(
        self, chain: Chain, tmp_path: Path
    ) -> None:
        """The one view this tree cannot carry is refused, not written empty."""
        from osprey.facility.errors import FacilityBuildError
        from osprey.facility.render import render_facility_outputs

        assert not any("in_context" in (c.get("tags") or []) for c in chain.document["channels"])

        with pytest.raises(FacilityBuildError, match="view-unsupported") as stop:
            render_facility_outputs(tmp_path, chain.document, IN_CONTEXT, chain.facility)

        assert "in_context" in str(stop.value)
        assert not (tmp_path / "data" / "channel_finder").exists()

    def test_a_second_pass_changes_no_view_byte(self, chain: Chain) -> None:
        first, second = chain.passes

        assert second.rendered == first.rendered

    def test_the_simulator_serves_every_channel_of_the_facility_file(self, chain: Chain) -> None:
        channels = {str(channel["id"]) for channel in chain.document["channels"]}
        imported = {
            str(channel["id"])
            for channel in yaml.safe_load(
                (_layer(chain) / "channels.yaml").read_text(encoding="utf-8")
            )
        }

        assert channels and channels == imported
        assert set(chain.view("simulator/addresses.json")["channels"]) == channels
        variables = chain.view("simulator/variables.json")
        assert {str(channel["address"]) for channel in variables["channels"]} == channels
        assert chain.view("facility_facts.json")["channel_count"] == len(channels)

    def test_the_simulator_serves_both_models_from_their_own_decks(self, chain: Chain) -> None:
        """Each model is served, over the bytes of the deck its export filed."""
        variables = chain.view("simulator/variables.json")
        served = {str(model["name"]) for model in variables["models"] if model["served"]}

        assert set(MODELS) <= served
        assert set(MODELS) <= set(chain.view("simulator/served_models.json")["models"])
        rendered = chain.passes[-1].rendered
        for name, model in _physics(chain.document).items():
            deck = (chain.facility / str(model["deck"])).read_bytes()
            assert rendered[f"data/simulator/decks/{name}.json"] == deck, name

    def test_each_model_wires_channels_of_its_own(self, chain: Chain) -> None:
        """A model drives channels the view serves, and no channel has two models."""
        variables = chain.view("simulator/variables.json")
        owner = {str(channel["address"]): channel["owner"] for channel in variables["channels"]}
        wired = {
            str(model["name"]): {str(record["address"]) for record in model["wiring"]}
            for model in variables["models"]
            if model["name"] in MODELS
        }

        assert sorted(wired) == list(MODELS)
        for name, addresses in wired.items():
            assert addresses, f"{name} wires no channel"
            assert {owner.get(address) for address in addresses} == {name}
        assert not wired[MODELS[0]] & wired[MODELS[1]]

    def test_the_limits_view_holds_the_records_of_the_limits_file(self, chain: Chain) -> None:
        records = yaml.safe_load((chain.facility / "limits.yaml").read_text(encoding="utf-8"))
        limits = chain.view("channel_limits.json")

        addresses = {str(record["address"]) for record in records["records"]}
        assert addresses
        assert {key for key in limits if not key.startswith("_")} == addresses

    def test_the_middle_layer_index_holds_every_channel(
        self, chain: Chain, middle_layer: Path
    ) -> None:
        """The DuckDB copy lists each channel of the facility file, and no other.

        A channel is one row per place it is listed, so the count is the
        number of distinct channel names.
        """
        import duckdb

        index = middle_layer / "data" / "channel_finder" / "middle_layer.duckdb"
        connection = duckdb.connect(str(index), read_only=True)
        try:
            rows = connection.execute("SELECT DISTINCT channel_name FROM channels").fetchall()
        finally:
            connection.close()

        assert {name for (name,) in rows} == {
            str(channel["id"]) for channel in chain.document["channels"]
        }


# ===================================================================
# The deck an export is paired with
# ===================================================================

#: What every line about a deck that disagrees with its export ends in.
REMEDY = "import the lattice the export was sampled from, or export again over this one"


def _paired_import(root: Path, tree: str, edit: Any = None, deck: Path | None = None) -> Any:
    """Import one single-export tree from a scratch copy of its files.

    Args:
        root: The scratch deployment. Created if absent.
        tree: The fixture tree; its one export is copied beside its siblings.
        edit: Applied to the copied ``<stem>.va.json`` document before the import.
        deck: A deck filed in place of the tree's own, under the tree's name.

    Returns:
        The result of the one ``osprey facility import mml`` call.
    """
    from osprey.cli.main import cli
    from osprey.facility.layers.mml.mapping import MAPPING_FILE

    (ao,) = export_files(tree)
    stem = ao.name.removesuffix(".ao.json")
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text(f"name: {PROJECT}\ndata: data\n", encoding="utf-8")
    installed = root / "data" / "facility" / MAPPING_FILE
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, installed)
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    for source in sorted((FIXTURES / tree).glob(f"{stem}.*")):
        shutil.copyfile(source, exports / source.name)
    if deck is not None:
        shutil.copyfile(deck, exports / f"{stem}.lattice.mat")
    if edit is not None:
        sibling = exports / f"{stem}.va.json"
        document = json.loads(sibling.read_text(encoding="utf-8"))
        edit(document)
        sibling.write_text(json.dumps(document), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        ["facility", "import", "mml", str(exports / ao.name), "--repo", str(root)],
        catch_exceptions=False,
    )
    assert "Traceback" not in result.output
    return result


class TestTheDeckBesideAnExport:
    """The import holds each deck to the lattice fingerprint its export states."""

    def test_an_export_beside_the_deck_of_another_lattice_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        """The spear3 export beside the nsls2 deck stops on the element count, writing nothing."""
        from osprey.facility.layers.mml.mapping import MAPPING_FILE

        result = _paired_import(
            tmp_path, "spear3", deck=FIXTURES / "nsls2" / "nsls2.storagering.lattice.mat"
        )

        assert result.exit_code == 1, result.output
        assert result.stderr.splitlines() == [
            "import mml: export-invalid: StorageRing: the deck spear3.storagering.lattice.mat "
            f"holds elements 3510 and the export states 876; {REMEDY}"
        ]
        facility = tmp_path / "data" / "facility"
        assert sorted(_files(facility, facility)) == [MAPPING_FILE]

    def test_an_export_that_differs_in_energy_alone_imports_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        """A stated energy the deck does not hold is said, and the deck is still filed."""

        def restate(document: dict[str, Any]) -> None:
            document["lattice"]["energy_gev"] = 2.5

        result = _paired_import(tmp_path, "synthetic", edit=restate)

        assert result.exit_code == 0, result.output
        assert (
            "import mml: deck energy: SR: the deck quokka.sr.lattice.mat holds energy_gev 2.0 "
            f"and the export states 2.5; {REMEDY}"
        ) in result.output.splitlines()
        assert (tmp_path / "data/facility/imported/mml/decks/SR.json").is_file()

    def test_an_export_that_refused_its_fingerprint_is_filed_unchecked_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        """A fingerprint the export could not take is no fact to hold the deck to."""

        def refuse(document: dict[str, Any]) -> None:
            document["lattice"] = {"refused": "no Java runtime"}

        result = _paired_import(tmp_path, "synthetic", edit=refuse)

        assert result.exit_code == 0, result.output
        assert (
            "import mml: deck unchecked: SR: the export states no lattice fingerprint "
            "(no Java runtime); the deck quokka.sr.lattice.mat is filed unchecked"
        ) in result.output.splitlines()
        assert (tmp_path / "data/facility/imported/mml/decks/SR.json").is_file()

    def test_a_deck_that_is_the_exports_own_prints_no_line_about_it(self, tmp_path: Path) -> None:
        result = _paired_import(tmp_path, "synthetic")

        assert result.exit_code == 0, result.output
        assert not [line for line in result.output.splitlines() if ": deck " in line]
