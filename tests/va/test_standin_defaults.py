"""The shipped stand-in BPM perturbation, against the four things it must fit.

:data:`STANDIN_BPM_ERRORS_DEFAULT` is the demo machine's own ``machine.json``
entry -- read from the tree the demo serves rather than written in the module
-- and it has to satisfy four separate contracts at once, none of which that
module can check:

1. **The served tree's data.** A deployment's perturbation is the one its own
   ``machine.json`` states, the packaged demo's included, so the spec is
   written beside that machine's channels and no device name is spelled in the
   framework's own source. A tree stating none ships none: there is no
   framework-side fallback for a facility to inherit.
2. **The packaged manifest.** Every device it names must be a BPM the served
   lattice actually has, and every axis it perturbs must have a readback
   address in ``channel_manifest.json`` -- an offset on an axis nothing serves
   is a perturbation with nowhere to appear.
3. **The env-var grammar.** It is rendered verbatim into a compose
   ``VA_BPM_ERRORS`` value, and the container parses it with
   ``entrypoint._parse_bpm_errors``, which ``SystemExit``\\ s on anything it
   does not like. That parser is called here on the real constant, so the
   pairing is checked rather than assumed -- and its answer is compared with
   :func:`parse_standin_default`, which is the host side's copy of the same
   split.
4. **The compose render.** The stand-in's env line is where the constant is
   actually delivered, inside a ``${VA_STANDIN_BPM_ERRORS-...}`` fallback so
   an operator keeps the override -- ``-`` and not ``:-``, so an explicitly
   EMPTY override is an unperturbed stand-in rather than a fall back to this
   constant; and a single-instance render must not so much as mention it.

The offset-only rule earns its own test because it is the load-bearing one:
with everything else in ``bpm_read``'s keyword set at identity, a reading is
exactly ``x - offset``, which is what lets the archiver seed reproduce the
same systematic error by adding it to the values it synthesizes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator import entrypoint
from osprey.services.virtual_accelerator.manifest import standin_defaults
from osprey.services.virtual_accelerator.manifest.paths import MANIFEST_OUTPUT, PACKAGE_PATHS
from osprey.services.virtual_accelerator.manifest.standin_defaults import (
    STANDIN_BPM_ERRORS_DEFAULT,
    STANDIN_BPM_ERRORS_KEY,
    default_bpm_errors_for_lattice,
    parse_bpm_error_spec,
    parse_standin_default,
    read_standin_bpm_errors,
    served_data_root,
)
from osprey.utils.dotenv import VA_LATTICE_DEFAULT

# The helpers that render the packaged VA compose template the way the
# deployment does. Imported from the instance-axis suite that owns them rather
# than restated, so a render pinned here is the same render pinned there.
from tests.deployment.test_va_compose_instances import (
    _context,
    _instance_block,
    _render,
    _render_text,
)

#: The address a BPM readback is served at: ``SR:DIAG:BPM:<id>:POSITION:<axis>``.
#: The fam_name the fault grammar keys on is ``BPM`` + that ``<id>``, which is
#: how the physics bridge matches a seeded error to an element.
_BPM_ADDRESS = re.compile(r"^SR:DIAG:BPM:([^:]+):POSITION:([XY])$")

#: The only two fields the shipped default may use, and the axis each perturbs.
_ALLOWED_FIELD_AXES = {"offset_x": "X", "offset_y": "Y"}


def _manifest_bpm_axes() -> dict[str, set[str]]:
    """``{fam_name: {"X", "Y"}}`` for every BPM the packaged manifest serves."""
    manifest = json.loads(MANIFEST_OUTPUT.read_text(encoding="utf-8"))
    axes: dict[str, set[str]] = {}
    for channel in manifest["channels"]:
        match = _BPM_ADDRESS.match(channel["address"])
        if match:
            axes.setdefault(f"BPM{match.group(1)}", set()).add(match.group(2))
    return axes


def _tree_stating(root: Path, spec: str) -> Path:
    """A deployment root carrying a machine that states ``spec``, as data root."""
    data_root = root / "data"
    (data_root / "simulation").mkdir(parents=True, exist_ok=True)
    (data_root / "simulation" / "machine.json").write_text(
        json.dumps({STANDIN_BPM_ERRORS_KEY: spec, "channels": {}}), encoding="utf-8"
    )
    return data_root


class TestTheDefaultIsTheServedTreesOwnData:
    """Where the perturbation is written: beside the machine it displaces."""

    def test_the_demo_states_its_perturbation_in_its_own_machine(self) -> None:
        """The constant is a read of the demo tree, not a second spelling of it.

        Two copies -- one in the data the demo serves, one in the framework's
        source -- could disagree about which machine the stand-in is standing
        in for, and the copy that lost would still render into a compose file.
        """
        machine = json.loads(PACKAGE_PATHS.machine_json.read_text(encoding="utf-8"))
        assert machine[STANDIN_BPM_ERRORS_KEY].strip() == STANDIN_BPM_ERRORS_DEFAULT
        assert STANDIN_BPM_ERRORS_DEFAULT, "the demo machine states no perturbation"

    def test_no_device_of_the_default_is_spelled_in_the_module(self) -> None:
        """A device name in framework source would serve exactly one facility."""
        source = Path(standin_defaults.__file__).read_text(encoding="utf-8")
        assert STANDIN_BPM_ERRORS_DEFAULT not in source
        for device in parse_standin_default():
            assert device not in source, f"{device} is spelled in {standin_defaults.__name__}"

    def test_a_tree_inherits_no_perturbation_from_the_framework(self, tmp_path: Path) -> None:
        """A machine that states none ships none, whatever the demo states.

        One bindings-driven path: a facility's stand-in perturbs the devices
        its own machine names, and a framework-side fallback would hand it the
        demo's device names -- offsets on devices it has never served.
        """
        data_root = tmp_path / "data"
        (data_root / "simulation").mkdir(parents=True)
        (data_root / "simulation" / "machine.json").write_text(
            json.dumps({"name": "a facility of its own", "channels": {}}), encoding="utf-8"
        )

        assert default_bpm_errors_for_lattice(True, data_root) == ""

    def test_a_tree_gets_the_perturbation_it_states_itself(self, tmp_path: Path) -> None:
        """Its own devices, not the ones the packaged demo happens to serve."""
        data_root = tmp_path / "data"
        (data_root / "simulation").mkdir(parents=True)
        spec = "C-A-01:offset_x=2.5e-4;C-A-09:offset_y=-1.5e-4"
        (data_root / "simulation" / "machine.json").write_text(
            json.dumps({STANDIN_BPM_ERRORS_KEY: spec, "channels": {}}), encoding="utf-8"
        )

        assert default_bpm_errors_for_lattice(True, data_root) == spec
        assert STANDIN_BPM_ERRORS_DEFAULT not in default_bpm_errors_for_lattice(True, data_root)

    def test_a_machine_stating_no_perturbation_ships_none(self, tmp_path: Path) -> None:
        """A machine whose stand-in reads as it does asks for the empty set."""
        machine = tmp_path / "machine.json"
        machine.write_text(json.dumps({"name": "unperturbed", "channels": {}}), encoding="utf-8")

        assert read_standin_bpm_errors(machine) == ""

    def test_a_perturbation_that_is_not_grammar_text_is_refused(self, tmp_path: Path) -> None:
        """Refused by name rather than rendered.

        The value is interpolated into a compose line verbatim, so a mapping
        left where the grammar belongs would reach the container as its own
        repr and fail at a boot nobody is watching.
        """
        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps({STANDIN_BPM_ERRORS_KEY: {"BPM03": {"offset_x": 1.5e-4}}, "channels": {}}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=STANDIN_BPM_ERRORS_KEY):
            read_standin_bpm_errors(machine)


class TestStandinDefaultErrorsFitTheManifest:
    """The constant names devices and axes the built-in machine really has."""

    def test_standin_default_errors_name_only_manifest_devices(self) -> None:
        known = set(_manifest_bpm_axes())
        assert known, "packaged manifest served no BPM position addresses"
        assert set(parse_standin_default()) <= known

    def test_standin_default_errors_perturb_only_served_axes(self) -> None:
        """An offset on an axis with no readback would never show up anywhere."""
        axes = _manifest_bpm_axes()
        for device, fields in parse_standin_default().items():
            for field in fields:
                assert _ALLOWED_FIELD_AXES[field] in axes[device], (
                    f"{device} has no {field} readback in the packaged manifest"
                )


class TestStandinDefaultErrorsAreOffsetOnly:
    """Offsets alone, at magnitudes the parser carries and a reader can see."""

    def test_standin_default_errors_use_offset_fields_only(self) -> None:
        """Gain, roll, polarity and noise are all refused, by design.

        Only a pure additive offset gives ``reading == x - offset``, and only
        that arithmetic can be reproduced additively by the archiver seed.
        """
        for fields in parse_standin_default().values():
            assert set(fields) <= set(_ALLOWED_FIELD_AXES)

    def test_the_parser_narrows_no_shipped_offset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A displacement answers to the machine, never to a ceiling.

        How far the shipped offsets displace the demo machine is a question
        about that machine -- far enough to see, close enough to be a
        commissioning error -- and the parser holds no opinion about it: a
        displacement is the magnitude the simulator was asked to seed, in the
        unit the monitor publishes. So every shipped offset, inflated past any
        plausible one, still reaches the error model as written.
        """
        inflated = {
            device: {field: value * 1e4 for field, value in fields.items()}
            for device, fields in parse_standin_default().items()
        }
        spec = ";".join(
            device + ":" + ",".join(f"{field}={value!r}" for field, value in fields.items())
            for device, fields in inflated.items()
        )
        monkeypatch.setenv("VA_BPM_ERRORS", spec)

        assert entrypoint._parse_bpm_errors() == {
            device: {field: pytest.approx(value) for field, value in fields.items()}
            for device, fields in inflated.items()
        }

    def test_standin_default_errors_are_visible_against_the_machine(self) -> None:
        """Every offset is well clear of the BPM channels' own motion.

        ``machine.json`` gives the storage-ring BPMs a 0.0 m baseline with a
        30 um wander texture on top. FR-4 compares a ``live`` read against a
        ``va`` read and expects them to differ by at least half the seeded
        offset, so half of the smallest offset here has to beat that wander --
        otherwise a passing comparison could be the weather.
        """
        magnitudes = [
            abs(value) for fields in parse_standin_default().values() for value in fields.values()
        ]
        assert magnitudes
        assert min(magnitudes) / 2 > 3e-5


class TestStandinDefaultErrorsRoundTripThroughTheGrammar:
    """The container's own parser accepts the constant, and agrees about it."""

    def test_standin_default_errors_parse_as_the_container_parses_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real ``_parse_bpm_errors``, on the real constant.

        It reads ``os.environ`` itself, so the constant is delivered the way
        compose delivers it. A malformed entry, an unknown field or a value
        the parser refuses would ``SystemExit`` here rather than at a container
        boot nobody is watching.
        """
        monkeypatch.setenv("VA_BPM_ERRORS", STANDIN_BPM_ERRORS_DEFAULT)
        parsed = entrypoint._parse_bpm_errors()

        assert parsed == parse_standin_default()
        assert parsed, "the shipped default perturbs nothing"

    def test_a_device_spelled_as_an_address_is_one_token(self) -> None:
        """Both splits take the LAST colon, so an address stays one device.

        A device is as free to be spelled as the address its reading is
        published on as by the element's own name; that spelling is colon
        separated at every level, and a field list carries none, so the last
        colon is the one between a device and its fields.
        """
        spec = "SR:DIAG:BPM:12:POSITION:X:offset_x=50e-6"

        assert parse_bpm_error_spec(spec) == {
            "SR:DIAG:BPM:12:POSITION:X": {"offset_x": pytest.approx(50e-6)}
        }

    def test_an_address_spelled_device_still_carries_a_field_list(self) -> None:
        spec = "SR:DIAG:BPM:12:X:offset_x=50e-6,polarity_y=-1"

        assert parse_bpm_error_spec(spec) == {
            "SR:DIAG:BPM:12:X": {
                "offset_x": pytest.approx(50e-6),
                "polarity_y": pytest.approx(-1.0),
            }
        }


class TestLatticeConditionalDefault:
    """The one rule the build and the render both resolve the fallback from."""

    def test_the_host_side_spells_no_lattice_as_the_container_does(self) -> None:
        """The two sides of ``VA_LATTICE=none`` are the same string.

        The host resolves the chain against
        :data:`~osprey_connectors.dotenv.VA_LATTICE_DEFAULT` while the container
        reads the variable itself, and a stand-in is rendered by the first and
        booted by the second. Two spellings of "no lattice" would leave the
        render perturbing a machine the IOC serves clean.
        """
        assert VA_LATTICE_DEFAULT == entrypoint.LATTICE_NONE

    def test_a_served_lattice_gets_its_trees_perturbation(self, tmp_path: Path) -> None:
        """A lattice in the served tree is a model for that tree's offsets."""
        data_root = _tree_stating(tmp_path, STANDIN_BPM_ERRORS_DEFAULT)

        assert default_bpm_errors_for_lattice(True, data_root) == STANDIN_BPM_ERRORS_DEFAULT

    def test_no_lattice_gets_the_empty_set(self, tmp_path: Path) -> None:
        """No model to displace, so the stand-in serves its manifest clean."""
        data_root = _tree_stating(tmp_path, STANDIN_BPM_ERRORS_DEFAULT)

        assert default_bpm_errors_for_lattice(False, data_root) == ""

    def test_a_deployment_describing_no_machine_gets_the_empty_set(self) -> None:
        """Nothing to stand in for, so nothing to perturb."""
        assert default_bpm_errors_for_lattice(True, None) == ""

    def test_a_data_root_with_no_machine_gets_the_empty_set(self, tmp_path: Path) -> None:
        """A tree that describes no machine is read as stating no perturbation."""
        assert default_bpm_errors_for_lattice(True, tmp_path / "data") == ""


class TestWhichTreeADeploymentIsHanded:
    """``served_data_root``: the tree the containers actually mount."""

    def test_the_repo_tree_answers_before_a_render_exists(self, tmp_path: Path) -> None:
        """What the build is about to copy is the honest answer meanwhile."""
        repo = _tree_stating(tmp_path / "repo", "D1:offset_x=1e-4").parent

        assert served_data_root(repo, tmp_path / "repo" / "build") == repo / "data"

    def test_the_published_render_wins_over_the_repo_tree(self, tmp_path: Path) -> None:
        """The containers mount the render, so the render is what they serve.

        Same precedence, and for the same reason, as the env chain's: a key
        both roots set is the published one's
        (:func:`~osprey_connectors.dotenv.resolved_va_lattice`).
        """
        repo = _tree_stating(tmp_path / "repo", "D1:offset_x=1e-4").parent
        build = _tree_stating(tmp_path / "repo" / "build", "D2:offset_y=2e-4").parent

        assert served_data_root(repo, build) == build / "data"
        assert default_bpm_errors_for_lattice(True, served_data_root(repo, build)) == (
            "D2:offset_y=2e-4"
        )

    def test_a_deployment_describing_no_machine_resolves_to_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "data").mkdir()

        assert served_data_root(tmp_path, tmp_path / "build") is None


class TestStandinDefaultErrorsReachTheComposeRender:
    """Where the constant is actually delivered: the stand-in's env line."""

    def test_standin_default_errors_render_as_the_standin_fallback(self) -> None:
        """Rendered as the ``-`` fallback, so the host override still wins.

        ``-``, not ``:-``: the default is substituted only for an UNSET
        variable, so ``VA_STANDIN_BPM_ERRORS=`` reaches the container as the
        empty fault set an operator asked for instead of being rounded back up
        to this constant.
        """
        text = _render_text(
            _context(
                instances={
                    "virtual_accelerator": _instance_block(5064),
                    "live_standin": _instance_block(5074),
                },
                deployed_services=["virtual_accelerator", "live_standin"],
                standin_bpm_errors_default=STANDIN_BPM_ERRORS_DEFAULT,
            )
        )
        assert f"${{VA_STANDIN_BPM_ERRORS-{STANDIN_BPM_ERRORS_DEFAULT}}}" in text
        assert "${VA_STANDIN_BPM_ERRORS:-" not in text

    def test_standin_default_errors_land_on_the_standin_instance_alone(self) -> None:
        """The baseline instance keeps its own clean ``VA_BPM_ERRORS``.

        Sharing one variable would apply an operator's fault to both machines
        at once, and leave the two reading alike when neither is set.
        """
        rendered = _render(
            _context(
                instances={
                    "virtual_accelerator": _instance_block(5064),
                    "live_standin": _instance_block(5074),
                },
                deployed_services=["virtual_accelerator", "live_standin"],
                standin_bpm_errors_default=STANDIN_BPM_ERRORS_DEFAULT,
            )
        )
        services = rendered["services"]
        assert (
            STANDIN_BPM_ERRORS_DEFAULT in services["live-standin"]["environment"]["VA_BPM_ERRORS"]
        )
        assert (
            STANDIN_BPM_ERRORS_DEFAULT
            not in services["virtual-accelerator"]["environment"]["VA_BPM_ERRORS"]
        )

    def test_standin_default_errors_do_not_leak_into_a_single_instance_render(self) -> None:
        """A project with one instance renders as if the constant did not exist."""
        text = _render_text(
            _context(
                instances={"virtual_accelerator": _instance_block(5064)},
                deployed_services=["virtual_accelerator"],
                standin_bpm_errors_default=STANDIN_BPM_ERRORS_DEFAULT,
            )
        )
        assert STANDIN_BPM_ERRORS_DEFAULT not in text
        assert "VA_STANDIN_BPM_ERRORS" not in text


class TestStandinDefaultErrorsMatchTheBuildRefusal:
    """The build-time check and the shipped default must agree.

    The build refuses a non-offset field in a profile's stand-in fault set;
    this pins the framework's own default against that same check, so the
    thing OSPREY ships could itself be built.
    """

    def test_standin_default_errors_pass_the_build_offset_only_check(self) -> None:
        checker = _shipped_bpm_errors_field_errors()
        if checker is None:
            pytest.skip("build-side offset-only check not present in this tree")
        assert checker(STANDIN_BPM_ERRORS_DEFAULT) == []


def _shipped_bpm_errors_field_errors():
    """The build's offset-only checker, or ``None`` where it does not exist.

    Resolved by lookup rather than imported at module scope: the check lands in
    the build layer on its own schedule, and this file must collect either way.
    """
    for module_name in (
        "osprey.cli.build_profile_va_faults",
        "osprey.cli.build_profile_model",
    ):
        try:
            module = __import__(module_name, fromlist=["_"])
        except ImportError:
            continue
        checker = getattr(module, "shipped_bpm_errors_field_errors", None)
        if checker is not None:
            return checker
    return None
