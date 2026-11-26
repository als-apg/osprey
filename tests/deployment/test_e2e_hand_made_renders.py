"""The deployments the container e2e lanes assemble by hand start like built ones.

Two e2e modules place a ``build/config.yml`` they wrote themselves under a
``profile.yml`` root: the multi-user lifecycle module (a committed facility
config stands in for a render) and the chain boot (the facility render gains
the agent's config). Neither runs ``osprey build``, so neither gets what a
build writes beside the config unless the fixture writes it too. A start verb
and the python executor both refuse a deployment whose render carries no
protected-path list, and those lanes need a container runtime, so a fixture
that lacks the list fails only in CI, minutes into a lane.

These hold the fixtures to the two readers without a container: the drift
check ``osprey up`` runs, and the list reader the executor refuses by.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.deployment.staleness import DriftState, check_drift
from osprey.utils.workspace import load_protected_paths
from tests.e2e import test_deploy_lifecycle as lifecycle


def _seeded(root: Path) -> Path:
    return lifecycle._seed_repo(
        root / "project",
        lifecycle.CONFIG_FIXTURE.read_text(encoding="utf-8"),
        name=lifecycle.PROJECT_NAME,
    )


def _isolated(root: Path) -> Path:
    ports = {
        family: 20000 + 100 * index
        for index, family in enumerate(
            ("nginx", "web", "artifact", "ariel", "channel_finder", "okf", "system_health")
        )
    }
    return lifecycle._make_isolation_repo(
        root / "project-a", project_name="osprey-e2e-hand-made", ports=ports, users=("alice",)
    )


@pytest.mark.parametrize(
    "make",
    [_seeded, _isolated, lifecycle._make_hetero_repo],
    ids=["committed config", "two-project isolation", "heterogeneous renders"],
)
def test_a_lifecycle_fixture_repo_passes_the_start_gate(make, tmp_path: Path) -> None:
    """``osprey up`` starts each lifecycle fixture: its drift check reads clean."""
    report = check_drift(make(tmp_path))

    assert report.state is DriftState.CLEAN, report.status_line
    assert not report.refuses


def test_the_chain_boot_deployment_carries_the_protected_path_list(tmp_path: Path) -> None:
    """The config the chain boot writes for the agent is one the executor runs under."""
    from tests.facility import test_chain as chain
    from tests.va.e2e import test_chain_boot as chain_boot

    built = chain.run_chain(tmp_path / chain.PROJECT, chain_boot.CHAIN_CONFIG, passes=1)
    served = SimpleNamespace(tree=SimpleNamespace(repo=built.root), port=5064)

    deployment = chain_boot._measuring(served)

    entries, finding = load_protected_paths(deployment.config_path)
    assert finding is None, finding
    assert entries == ()
