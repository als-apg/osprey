"""Every e2e stack built from demo pins the scenarios it starts in.

An e2e lane reads live values off the simulator, so the scenarios a stack
starts in are part of what its assertions stand on. The preset's default set
is a product decision that moves on its own; ``demo-history`` in particular
replays a whole fabricated past into the archiver and the logbook, which no
lane asserts against. So every site that runs ``osprey init --preset
demo`` and then ``up`` states the start set itself, through
:func:`tests.e2e.profile_edits.e2e_scenario_pin` -- the preset's own default
minus ``demo-history``, read off the preset file rather than copied here.

This reads the SOURCE of the test modules rather than importing them, for the
reason ``test_e2e_project_names`` gives: importing one runs its import-time
guards. The helper itself is imported, lazily, by the tests that check what it
returns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests._tree_scan import TESTS_ROOT, python_sources

#: The helper every such init calls, by the name the scan looks for.
HELPER = "e2e_scenario_pin"

#: The shared builders the deployed lanes init through. Each must pin, and the
#: scan must reach each, or a green result says nothing.
SHARED_BUILDERS = ("e2e/sdk_helpers.py", "e2e/_orm_stack.py")

#: Modules that init demo and name ``up`` without needing the pin,
#: each with why. A new module lands here only with a reason as good.
EXEMPT = {
    # The PTY suite renders hello-world; demo is only a parameter
    # value it never brings up.
    "pty/conftest.py": "renders hello-world, never ups demo",
    # CLI tests drive `up` against a stubbed compose; nothing reads a scenario.
    "cli/test_lifecycle_phases.py": "stubbed up, no simulator runs",
    "cli/test_init_verb.py": "stubbed up, no simulator runs",
    # The tutorial stack pins its own start set (task 5.2 owns it).
    "docs/screenshots/test_tutorial_stack.py": "tutorial stack pins its own start set",
    # Deploys hello-world; demo appears only in prose.
    "e2e/test_bluesky_deploy.py": "deploys hello-world",
}

#: A frozen copy of a past release's tree, not a test of this one.
FROZEN = "fixtures/previous_release_repo/"

_PRESET_FLAG = re.compile(r"""["']--preset["']""")
_DEMO = re.compile(r"\bdemo\b")
_UP_VERB = re.compile(r"""["']up["']""")


def _inits_demo_and_ups(text: str) -> bool:
    """Whether a module's source runs a demo init and names ``up``."""
    return bool(
        _PRESET_FLAG.search(text) and _DEMO.search(text) and _UP_VERB.search(text)
    )


def test_e2e_scenario_pin_every_demo_init_that_ups_pins_its_scenarios() -> None:
    matched: list[str] = []
    offenders: list[str] = []
    for relative, text in python_sources(exempt=Path(__file__)):
        if relative.startswith(FROZEN) or relative in EXEMPT:
            continue
        if relative not in SHARED_BUILDERS and not _inits_demo_and_ups(text):
            continue
        matched.append(relative)
        if HELPER not in text:
            offenders.append(relative)

    for builder in SHARED_BUILDERS:
        assert builder in matched, f"the scan must reach {builder}; it matched {matched}"
    assert "va/e2e/test_bluesky_lanes.py" in matched, (
        f"the scan must reach tests/va/e2e as well as tests/e2e; it matched {matched}"
    )
    assert offenders == [], (
        f"these modules init demo and bring it up without {HELPER}(), "
        f"so their stack starts in whatever the preset's default set is today: {offenders}"
    )


def test_e2e_scenario_pin_exemptions_still_name_existing_modules() -> None:
    missing = [path for path in EXEMPT if not (TESTS_ROOT / path).is_file()]
    assert missing == [], f"exemptions for modules that no longer exist: {missing}"


def test_e2e_scenario_pin_is_the_preset_default_minus_demo_history() -> None:
    from osprey.cli.build_profile_resolve import preset_authored_config
    from tests.e2e.profile_edits import WITHHELD_SCENARIO, e2e_scenario_pin

    authored = preset_authored_config("demo")["simulation.default_scenarios"]
    flag, pair = e2e_scenario_pin()
    assert flag == "--set"
    key, _, value = pair.partition("=")
    assert key == "config.simulation.default_scenarios"
    assert WITHHELD_SCENARIO == "demo-history"
    pinned = json.loads(value)
    assert pinned == [name for name in authored or [] if name != WITHHELD_SCENARIO]
    assert WITHHELD_SCENARIO not in pinned


def test_e2e_scenario_pin_resolves_to_the_literal_config_key_with_a_list() -> None:
    from osprey.cli.build_profile_resolve import cli_edit_layer
    from tests.e2e.profile_edits import e2e_scenario_pin

    _flag, pair = e2e_scenario_pin()
    layer = cli_edit_layer((pair,))
    assert list(layer) == ["config"]
    assert list(layer["config"]) == ["simulation.default_scenarios"]
    assert isinstance(layer["config"]["simulation.default_scenarios"], list)


def test_e2e_scenario_pin_is_empty_for_a_preset_that_states_no_start_set() -> None:
    from tests.e2e.profile_edits import e2e_scenario_pin

    assert e2e_scenario_pin("hello-world") == []


def test_e2e_scenario_pin_rides_in_the_shared_plan_stack_argv(tmp_path: Path) -> None:
    from tests.e2e._orm_stack import init_args
    from tests.e2e.profile_edits import e2e_scenario_pin

    _flag, pair = e2e_scenario_pin()
    assert pair in init_args("p", output_dir=tmp_path)


def test_e2e_scenario_pin_unaffected_init_renders_the_start_set_without_demo_history(
    tmp_path: Path,
) -> None:
    """A pinned demo init renders a start set with no ``demo-history``.

    The pin is checked where it lands, not only as an argument: ``osprey init``
    with it writes the profile, ``osprey build`` renders ``config.yml``, and the
    rendered ``simulation.default_scenarios`` is the preset's default minus
    ``demo-history`` -- so the stack an e2e lane brings up seeds no history.
    """
    import yaml
    from click.testing import CliRunner

    from osprey.cli.build_profile_resolve import preset_authored_config
    from osprey.cli.init_cmd import init
    from tests._builds import run_build
    from tests.e2e.profile_edits import WITHHELD_SCENARIO, e2e_scenario_pin

    repo = tmp_path / "pinned"
    inited = CliRunner().invoke(
        init, [str(repo), "--preset", "demo", "--no-git", *e2e_scenario_pin()]
    )
    assert inited.exit_code == 0, inited.output
    built = run_build(repo)
    assert built.exit_code == 0, built.output

    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    start_set = rendered["simulation"]["default_scenarios"]
    authored = preset_authored_config("demo")["simulation.default_scenarios"]
    assert WITHHELD_SCENARIO in authored, "the preset's own default seeds the history"
    assert WITHHELD_SCENARIO not in start_set
    assert start_set == [name for name in authored if name != WITHHELD_SCENARIO]
