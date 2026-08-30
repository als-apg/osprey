"""The shipped stand-in BPM perturbation, against the three things it must fit.

:data:`STANDIN_BPM_ERRORS_DEFAULT` is a string constant that has to satisfy
three separate contracts at once, none of which its own module can check:

1. **The packaged manifest.** Every device it names must be a BPM the built-in
   lattice actually has, and every axis it perturbs must have a readback
   address in ``channel_manifest.json`` -- an offset on an axis nothing serves
   is a perturbation with nowhere to appear.
2. **The env-var grammar.** It is rendered verbatim into a compose
   ``VA_BPM_ERRORS`` value, and the container parses it with
   ``entrypoint._parse_bpm_errors``, which ``SystemExit``\\ s on anything it
   does not like. That parser is called here on the real constant, so the
   pairing is checked rather than assumed -- and its answer is compared with
   :func:`parse_standin_default`, which is the host side's copy of the same
   split.
3. **The compose render.** The stand-in's env line is where the constant is
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

import pytest

from osprey.services.virtual_accelerator import entrypoint
from osprey.services.virtual_accelerator.manifest.paths import MANIFEST_OUTPUT
from osprey.services.virtual_accelerator.manifest.standin_defaults import (
    LATTICE_BUILTIN,
    STANDIN_BPM_ERRORS_DEFAULT,
    default_bpm_errors_for_lattice,
    parse_standin_default,
)

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
    """Offsets alone, at magnitudes the parser accepts and a reader can see."""

    def test_standin_default_errors_use_offset_fields_only(self) -> None:
        """Gain, roll, polarity and noise are all refused, by design.

        Only a pure additive offset gives ``reading == x - offset``, and only
        that arithmetic can be reproduced additively by the archiver seed.
        """
        for fields in parse_standin_default().values():
            assert set(fields) <= set(_ALLOWED_FIELD_AXES)

    def test_standin_default_errors_stay_inside_the_parse_bounds(self) -> None:
        """Bounds come from the parser that enforces them, not from a copy."""
        for device, fields in parse_standin_default().items():
            for field, value in fields.items():
                low, high = entrypoint._BPM_ERROR_FIELD_BOUNDS[field]
                assert low <= value <= high, f"{device}:{field}={value}"

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
        compose delivers it. A malformed entry, an unknown field or an
        out-of-bounds magnitude would ``SystemExit`` here rather than at a
        container boot nobody is watching.
        """
        monkeypatch.setenv("VA_BPM_ERRORS", STANDIN_BPM_ERRORS_DEFAULT)
        parsed = entrypoint._parse_bpm_errors()

        assert parsed == parse_standin_default()
        assert parsed, "the shipped default perturbs nothing"


class TestLatticeConditionalDefault:
    """The one rule the build and the render both resolve the fallback from."""

    def test_lattice_builtin_is_the_entrypoints_own_spelling(self) -> None:
        """The host-side respelling, pinned against the container's constant.

        ``entrypoint`` owns the value and cannot be imported by the build or the
        render, so it is respelled beside the default it conditions. A rename
        there without one here would leave every stand-in rendered unperturbed.
        """
        assert LATTICE_BUILTIN == entrypoint.LATTICE_BUILTIN

    def test_the_builtin_lattice_gets_the_shipped_perturbation(self) -> None:
        """Case- and whitespace-insensitively: a chain value is written by hand."""
        assert default_bpm_errors_for_lattice(LATTICE_BUILTIN) == STANDIN_BPM_ERRORS_DEFAULT
        assert default_bpm_errors_for_lattice("  BuiltIn ") == STANDIN_BPM_ERRORS_DEFAULT

    @pytest.mark.parametrize("lattice", ["none", "", "channels.json"])
    def test_every_other_lattice_gets_the_empty_set(self, lattice: str) -> None:
        """No PyAT model to displace, so the stand-in serves its manifest clean."""
        assert default_bpm_errors_for_lattice(lattice) == ""


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
