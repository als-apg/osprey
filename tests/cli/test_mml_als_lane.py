"""The ALS lane: the whole install, over the one facility that never enters the repo.

Every other lane in this suite runs on a committed fixture. A fixture is a
machine somebody invented to exercise a rule, so a fixture that passes says the
rule is self-consistent, not that it survives contact with a real middle layer
whose families were named over thirty years by people who were not thinking
about this exporter. That contact is what this file buys, and it buys it
without bringing the facility into the repository: the export, the reviewed
mapping and everything derived from them stay on the machine that has them, and
the ruling for this feature is that no facility constant, family name, count or
address enters osprey or its committed tests. So nothing here is pinned to a
number. Every assertion is a shape, which is also why the same assertions can be
re-run over a stand-in tree to prove this file's own code path.

The lane is two deployments over one export, because that is how a facility is
installed:

* the chain --- ``import`` -> ``map --init`` -> the reviewed mapping ->
  ``map --check`` -> ``emit --duckdb`` -> ``verify``, twice --- driven by the
  same ``run_chain`` the fixtures use, and judged by the same cases
  :class:`~tests.cli.test_mml_chain.TestTheVirtualAcceleratorChain` judges them
  by. The report that chain ends in is read for the sections a reviewer reads.
* the install recipe --- ``init --preset control-assistant`` -> the harvest ->
  the refusals -> ``set`` -> ``validate`` -> ``build``, twice --- and judged by
  the cases :class:`~tests.cli.test_mml_build_recipes.TestServedFromATwoZeroExport`
  judges the fixture recipe by.

Both borrow their assertions whole rather than restating them. An invariant
that moves in either parent moves here in the same commit, and a real export
that breaks one breaks it under the name it already has, which is the only way
this lane is worth its skip.

The lane needs a 2.0 export. A 1.0 export harvested onto this preset raises the
open question of whose ring the deployment then serves, and that question is
not this file's to answer, so an export carrying no virtual accelerator is a
named skip rather than a lane that quietly asserts one of the answers.

Two ways in other than the facility's own checkout:

* ``OSPREY_ALS_LANE_STAND_IN=<dir>`` names a directory holding a 2.0 export and
  its reviewed ``mapping.yaml``, and the lane installs that instead. It is
  test-only. The same fixtures and the same borrowed cases run, so a green run
  proves this file's code path and the plumbing it drives; it proves nothing
  about the facility, whose export it never touched.
* ``OSPREY_ALS_MML_EXPORT`` and ``OSPREY_ALS_MML_MAPPING`` name one export and
  its mapping directly, and win ahead of both the checkout gate and the MATLAB
  gate --- deliberately, so a reviewer holding an export elsewhere names it
  rather than moving it. That pair is shared with the counts lane in
  ``test_mml_chain``, so setting it for one lane arms the other too, and this
  one is two full deployments of the named export.
"""

from __future__ import annotations

import inspect
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.services.mml.va.verify import REPORT_FILENAME

pytest.importorskip("linkml_runtime")
pytest.importorskip("duckdb")
pytest.importorskip("lume_pyat")

from tests.cli.test_mml_build_recipes import (  # noqa: E402
    TestServedFromATwoZeroExport as _BuildCases,
)
from tests.cli.test_mml_build_recipes import (  # noqa: E402
    drive_emit,
    env_values,
    invoke,
    published,
    served_settings,
)
from tests.cli.test_mml_chain import (  # noqa: E402
    ALS_EXPORT_ENV,
    ALS_MAPPING_ENV,
    Chain,
    run_chain,
)
from tests.cli.test_mml_chain import (  # noqa: E402
    TestTheVirtualAcceleratorChain as _ChainCases,
)
from tests.cli.test_mml_verify import SECTIONS  # noqa: E402

#: Where the facility's own profiles are checked out. The one path this file
#: knows: a discovery root on the machine that has the export, not anything
#: about the facility, and an installation that keeps it elsewhere says so
#: rather than editing this file.
ALS_PROFILES_ENV = "OSPREY_ALS_PROFILES"
DEFAULT_ALS_PROFILES = Path("/Users/thellert/code/als-profiles")

#: The MATLAB that produces the export. The lane does not run it --- the
#: re-export is a session a person sits through --- but a machine with no
#: MATLAB has no way to refresh what it reads, so a tree found there would be
#: evidence about a build nobody on that machine can reproduce.
MATLAB_ENV = "OSPREY_MML_MATLAB"

#: A test-only stand-in: a directory carrying a 2.0 export and its reviewed
#: mapping, run through the lane in place of the facility's. It exists so this
#: file's own resolution, fixtures and borrowed cases can be executed on a
#: machine that has no export at all --- proving the lane runs, never proving
#: anything about the facility. A run under it reports the tree it used.
STAND_IN_ENV = "OSPREY_ALS_LANE_STAND_IN"

AO_SUFFIX = ".ao.json"
VA_SUFFIX = ".va.json"


@dataclass(frozen=True)
class Lane:
    """The inputs one run of the lane installs.

    Attributes:
        label: What the tree is, for the failure to name.
        exports: The export paths handed to ``mml import``.
        mapping: The reviewed ``mapping.yaml`` the chain installs.
    """

    label: str
    exports: tuple[str, ...]
    mapping: Path


def _va_sibling(export: Path) -> Path:
    """The virtual accelerator ``mml import`` picks up beside *export*."""
    return export.with_name(export.name[: -len(AO_SUFFIX)] + VA_SUFFIX)


def _two_zero_export(root: Path) -> Path | None:
    """The first export under *root* that carries a machine, or nothing."""
    for export in sorted(root.rglob(f"*{AO_SUFFIX}")):
        if _va_sibling(export).is_file():
            return export
    return None


def _from_tree(tree: Path, label: str) -> Lane | str:
    """Read one directory as a lane, or say what it is missing."""
    exports = sorted(tree.glob(f"*{AO_SUFFIX}"))
    if not exports:
        return f"{label} holds no *{AO_SUFFIX} export"
    if not any(_va_sibling(export).is_file() for export in exports):
        return (
            f"{label} carries no *{VA_SUFFIX} sibling, so it is a 1.0 export; "
            "the lane asserts the 2.0 chain through build"
        )
    mapping = tree / "mapping.yaml"
    if not mapping.is_file():
        return f"{label} holds no reviewed mapping.yaml"
    return Lane(label=label, exports=tuple(str(path) for path in exports), mapping=mapping)


def _resolve() -> Lane | str:
    """The lane this machine can run, or the reason it can run none.

    The two explicit variables win over discovery, as they already do for the
    counts lane in ``test_mml_chain``: a reviewer holding an export somewhere
    else names it rather than moving it.
    """
    stand_in = os.environ.get(STAND_IN_ENV)
    if stand_in:
        tree = Path(stand_in).expanduser().resolve()
        if not tree.is_dir():
            return f"{STAND_IN_ENV} names {tree}, which is not a directory"
        return _from_tree(tree, f"stand-in tree {tree.name}")

    named_export = os.environ.get(ALS_EXPORT_ENV)
    named_mapping = os.environ.get(ALS_MAPPING_ENV)
    if named_export and named_mapping:
        export = Path(named_export).expanduser().resolve()
        mapping = Path(named_mapping).expanduser().resolve()
        if not export.is_file():
            return f"{ALS_EXPORT_ENV} names {export}, which is not a file"
        if not mapping.is_file():
            return f"{ALS_MAPPING_ENV} names {mapping}, which is not a file"
        if not _va_sibling(export).is_file():
            return (
                f"{ALS_EXPORT_ENV} names a 1.0 export ({_va_sibling(export).name} is "
                "not beside it); the lane asserts the 2.0 chain through build"
            )
        return Lane(
            label=f"the export named by {ALS_EXPORT_ENV}",
            exports=(str(export),),
            mapping=mapping,
        )

    root = Path(os.environ.get(ALS_PROFILES_ENV, DEFAULT_ALS_PROFILES)).expanduser()
    if not root.is_dir():
        return (
            f"{root} is not a checkout; the lane reads the facility's own profiles "
            f"there, or wherever {ALS_PROFILES_ENV} names"
        )
    if not (os.environ.get(MATLAB_ENV) or shutil.which("matlab")):
        return (
            f"no MATLAB on PATH and {MATLAB_ENV} names none; the 2.0 export the "
            "lane installs is produced there"
        )
    export = _two_zero_export(root)
    if export is None:
        return (
            f"no 2.0 export under {root}: nothing there carries a *{VA_SUFFIX} "
            "sibling, so the facility has not been re-exported with mml_export 2.0 "
            f"(or {ALS_EXPORT_ENV} and {ALS_MAPPING_ENV} name one elsewhere)"
        )
    return _from_tree(export.parent, f"the 2.0 export in {export.parent}")


_RESOLVED = _resolve()
LANE: Lane | None = _RESOLVED if isinstance(_RESOLVED, Lane) else None
SKIP_REASON = "" if LANE is not None else str(_RESOLVED)

pytestmark = [
    pytest.mark.requires_als_profiles,
    pytest.mark.skipif(LANE is None, reason=SKIP_REASON),
]


def _mapping_token() -> str:
    """The container-name prefix the reviewed mapping implies.

    The same value the recipe's ``facility_prefix`` computes, read off the
    mapping this lane installs: the token names the facility, and lowercase is
    what survives Docker object names.
    """
    assert LANE is not None
    document = yaml.safe_load(LANE.mapping.read_text(encoding="utf-8"))
    return str(document["facility"]["token"]).lower()


def _borrowed(cases: type, parameter: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The cases of *cases* this lane can feed, and the ones it cannot.

    Selected on the fixture each case asks for: one that takes *parameter*
    judges a single installed tree, which is what this lane has to hand it. The
    rest are returned too rather than dropped, because a case this lane cannot
    feed is a case that never runs over the facility, and the borrow guard
    turns that into a red test instead of a quiet fourteen green.

    Returns:
        The selected ``test_*`` names and the rejected ones.
    """
    selected: list[str] = []
    rejected: list[str] = []
    for name, function in vars(cases).items():
        if not name.startswith("test_"):
            continue
        if parameter in inspect.signature(function).parameters:
            selected.append(name)
        else:
            rejected.append(name)
    return tuple(selected), tuple(rejected)


#: The one parent case this lane is right not to run: it asks which fixtures
#: the repository commits, which is a question about this suite and not about a
#: facility, and it takes no tree to be handed. Every other rejected case is a
#: case the lane cannot feed, and the borrow guard names it.
EXCLUDED = frozenset({"test_a_two_zero_export_is_committed"})

#: The 2.0 chain's cases, and the install recipe's, run here over the
#: facility's own tree. Named rather than copied: an invariant that changes in
#: either parent changes here in the same commit.
CHAIN_CASES, CHAIN_REJECTED = _borrowed(_ChainCases, "two_zero_chain")
BUILD_CASES, BUILD_REJECTED = _borrowed(_BuildCases, "served_repo")


@pytest.fixture(scope="module")
def als_chain(tmp_path_factory: pytest.TempPathFactory) -> Chain:
    """The facility installed end to end, through ``verify``, twice over."""
    assert LANE is not None
    return run_chain(
        tmp_path_factory.mktemp("als-chain") / "deployment",
        LANE.exports,
        (),
        LANE.mapping,
        name=LANE.label,
        verify=True,
    )


@pytest.fixture(scope="module")
def als_build(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The install recipe over the facility's export, driven once.

    The verbs are the recipe's, in the recipe's order, on a second deployment:
    the chain builds in a scratch repo with no preset, and ``osprey build``
    publishes a tree only a real deployment has. Nothing is decided here ---
    each refusal is obeyed as written, and every claim about what came out is
    borrowed from the recipe's own cases.

    This deployment stands on its own: the container-name token is a property
    of the reviewed mapping, so it is read there rather than from the chain,
    and a chain that breaks leaves these cases free to fail or pass on their
    own evidence.
    """
    assert LANE is not None
    runner = CliRunner()
    repo = tmp_path_factory.mktemp("als-build") / "deployment"

    invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    invoke(runner, "mml", "import", *LANE.exports, "--repo", str(repo))
    shutil.copy(LANE.mapping, repo / "data" / "mml" / "mapping.yaml")

    emitted, rounds = drive_emit(runner, repo)
    invoke(runner, "set", "--repo", str(repo), *served_settings(_mapping_token()))

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")
    first = published(repo)
    first_env = env_values(repo)
    invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "fixture": LANE.label,
        "repo": repo,
        "rounds": rounds,
        "emit": emitted.output,
        "validate": validate.output,
        "build": build.output,
        "first": first,
        "first_env": first_env,
    }


def test_the_lane_borrows_cases_from_both_parents() -> None:
    # Both selections are computed, so a parent whose fixture is renamed would
    # empty its side silently and leave a lane that installs a facility and
    # asserts nothing about it. A parent that gains a case taking some other
    # fixture is the quieter version of the same failure: the lane keeps
    # reporting green over a facility it no longer judges by that case. Both
    # are read off the parents' own members, so a rename shows up here.
    assert CHAIN_CASES, "no 2.0 chain case takes a chain"
    assert BUILD_CASES, "no install-recipe case takes a built repo"

    members = set(vars(_ChainCases)) | set(vars(_BuildCases))
    assert EXCLUDED <= members, (
        f"{sorted(EXCLUDED - members)} is excluded from this lane but neither "
        "parent defines it; the exclusion outlived the case it names"
    )

    stranded = sorted((set(CHAIN_REJECTED) | set(BUILD_REJECTED)) - EXCLUDED)
    assert not stranded, (
        f"{stranded} would never run over {LANE.label if LANE else 'this lane'} "
        f"({LANE.exports if LANE else ()}): each takes a fixture this lane does "
        "not supply. Give the lane that fixture, or excuse the case in EXCLUDED"
    )


@pytest.mark.parametrize("case", CHAIN_CASES)
def test_the_facility_holds_every_two_zero_chain_invariant(als_chain: Chain, case: str) -> None:
    getattr(_ChainCases(), case)(als_chain)


@pytest.mark.parametrize("case", BUILD_CASES)
def test_the_facility_holds_every_served_build_invariant(
    als_build: dict[str, Any], case: str
) -> None:
    getattr(_BuildCases(), case)(als_build)


def test_the_report_carries_every_section_a_reviewer_reads(als_chain: Chain) -> None:
    """A reviewer reads the report, so the report has to be the whole one.

    The sections are compared as a sequence rather than as a set: the order is
    the order the argument is made in --- the verdict, then what was exported,
    then the comparison, then everything the comparison could not cover --- and
    a report whose blocks arrive in another order is a different document from
    the one the reviewer was taught to read. A section this file does not know
    is left alone; only the known ones are held to their order.
    """
    report = (als_chain.root / "data" / "mml" / REPORT_FILENAME).read_text(encoding="utf-8")
    headings = [line.rstrip() for line in report.splitlines() if line.startswith("## ")]

    missing = [section for section in SECTIONS if section not in headings]
    assert not missing, f"{LANE.label if LANE else ''} is missing {missing}"
    assert tuple(heading for heading in headings if heading in SECTIONS) == SECTIONS
