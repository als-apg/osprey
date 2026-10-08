"""The one rule that decides which containers on a host belong to a deployment.

A row labelled for this project is ours. A row labelled for another project is
never ours, whatever it is called. A row with no project label at all is ours
only when one of its names matches a service by whole name segments.

:data:`HOST_ROWS` is shared: the health-probe and status tests import it to pin
that every reader claims exactly the rows this rule claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from osprey.deployment.compose_generator import PROJECT_LABEL, REPO_ID_LABEL
from osprey.deployment.container_ownership import (
    COMPOSE_PROJECT_LABEL,
    COMPOSE_SERVICE_LABEL,
    START_REFUSAL,
    ClaimBasis,
    ForeignCheckoutError,
    Resource,
    claims_row,
    container_label,
    container_names,
    deployment_containers,
    host_claim,
    names_match_service,
    row_project,
)

PROJECT = "mine"


def _row(name: str, state: str, **labels: str) -> dict[str, Any]:
    """One podman-shaped ``ps`` record."""
    return {"Names": [name], "State": state, "Labels": dict(labels)}


HOST_ROWS: list[dict[str, Any]] = [
    _row("other-openobserve", "running", **{PROJECT_LABEL: "other"}),
    _row("mine-openobserve", "exited", **{PROJECT_LABEL: PROJECT}),
    _row(
        "mine-ariel-postgres",
        "running",
        **{PROJECT_LABEL: PROJECT, COMPOSE_SERVICE_LABEL: "postgresql"},
    ),
    _row("mine-archiver-mongodb", "exited", **{PROJECT_LABEL: PROJECT}),
    _row("mine-archive", "running", **{PROJECT_LABEL: PROJECT}),
    _row("stack-openobserve", "running", **{COMPOSE_PROJECT_LABEL: "stack"}),
    _row("openobserve-scratch", "running"),
    {
        "Names": "mine-archiver-recorder",
        "State": "running",
        "Labels": f"{PROJECT_LABEL}={PROJECT},{COMPOSE_SERVICE_LABEL}=archiver_recorder",
    },
]


def _names(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {container_names(row)[0] for row in rows}


def _ours(rows: list[dict[str, Any]], services: tuple[str, ...], **kw: Any) -> set[str]:
    owned = deployment_containers(rows, project_name=PROJECT, services=services, **kw)
    return {container_names(c.row)[0] for c in owned.ours}


def test_a_row_labelled_for_this_project_is_ours() -> None:
    owned = deployment_containers(HOST_ROWS, project_name=PROJECT, services=())
    bases = {container_names(c.row)[0]: c.basis for c in owned.ours}
    assert bases == {
        "mine-openobserve": ClaimBasis.PROJECT,
        "mine-ariel-postgres": ClaimBasis.PROJECT,
        "mine-archiver-mongodb": ClaimBasis.PROJECT,
        "mine-archive": ClaimBasis.PROJECT,
        "mine-archiver-recorder": ClaimBasis.PROJECT,
    }


def test_a_row_labelled_for_another_project_is_never_ours_even_by_name() -> None:
    ours = _ours(HOST_ROWS, ("openobserve",))
    assert "other-openobserve" not in ours
    assert "stack-openobserve" not in ours


def test_the_compose_project_label_counts_when_the_osprey_label_is_absent() -> None:
    row = _row("mine-event-dispatcher", "running", **{COMPOSE_PROJECT_LABEL: PROJECT})
    owned = deployment_containers([row], project_name=PROJECT, services=())
    assert [c.basis for c in owned.ours] == [ClaimBasis.PROJECT]
    assert row_project(row) == PROJECT


def test_the_osprey_label_wins_over_the_compose_label() -> None:
    row = _row(
        "x-openobserve", "running", **{PROJECT_LABEL: "other", COMPOSE_PROJECT_LABEL: PROJECT}
    )
    assert row_project(row) == "other"
    owned = deployment_containers([row], project_name=PROJECT, services=("openobserve",))
    assert owned.ours == ()
    assert owned.other_projects == (row,)


def test_an_unlabelled_row_is_ours_only_when_a_name_matches_a_service() -> None:
    assert "openobserve-scratch" in _ours(HOST_ROWS, ("openobserve",))
    assert "openobserve-scratch" not in _ours(HOST_ROWS, ("postgresql",))
    owned = deployment_containers(HOST_ROWS, project_name=PROJECT, services=("openobserve",))
    by_name = [c for c in owned.ours if c.basis is ClaimBasis.NAME]
    assert _names(c.row for c in by_name) == {"openobserve-scratch"}


def test_this_checkouts_repo_id_claims_a_row_whatever_its_project_label() -> None:
    row = _row("elsewhere-openobserve", "running", **{PROJECT_LABEL: "other", REPO_ID_LABEL: "id1"})
    owned = deployment_containers([row], project_name=PROJECT, services=(), identity="id1")
    assert [c.basis for c in owned.ours] == [ClaimBasis.CHECKOUT]
    assert owned.other_projects == ()


def test_other_projects_lists_only_osprey_labelled_rows() -> None:
    owned = deployment_containers(HOST_ROWS, project_name=PROJECT, services=("openobserve",))
    assert _names(owned.other_projects) == {"other-openobserve"}


def test_docker_string_labels_and_podman_object_labels_read_alike() -> None:
    podman = _row("/p-x", "running", **{PROJECT_LABEL: "p", COMPOSE_SERVICE_LABEL: "x"})
    docker = {
        "Names": "/p-x",
        "State": "running",
        "Labels": f"{PROJECT_LABEL}=p, {COMPOSE_SERVICE_LABEL}=x",
    }
    for row in (podman, docker):
        assert container_label(row, PROJECT_LABEL) == "p"
        assert container_label(row, COMPOSE_SERVICE_LABEL) == "x"
        assert container_label(row, REPO_ID_LABEL) is None
        assert container_names(row) == ["p-x"]
    docker_owned = deployment_containers(HOST_ROWS, project_name=PROJECT, services=())
    assert "mine-archiver-recorder" in {container_names(c.row)[0] for c in docker_owned.ours}


@pytest.mark.parametrize(
    ("service", "row", "project", "expected"),
    [
        ("dispatch_worker", _row("p-dispatch-worker-2", "running"), "p", True),
        ("mongodb", _row("p-archiver-mongodb", "running"), "p", True),
        ("archive", _row("p-archiver-recorder", "running"), "p", False),
        (
            "postgresql",
            _row("p-ariel-postgres", "running", **{COMPOSE_SERVICE_LABEL: "postgresql"}),
            "p",
            True,
        ),
        ("graphdb", _row("graphdb-demo-qmd", "running"), "graphdb-demo", False),
    ],
)
def test_a_service_matches_whole_name_segments_only(
    service: str, row: dict[str, Any], project: str, expected: bool
) -> None:
    assert names_match_service(row, service, project) is expected


def test_for_service_reads_the_compose_service_label() -> None:
    owned = deployment_containers(HOST_ROWS, project_name=PROJECT, services=("postgresql",))
    assert _names(owned.for_service("postgresql")) == {"mine-ariel-postgres"}
    assert _names(owned.for_service("archive")) == {"mine-archive"}


def test_status_partition_claims_exactly_what_the_rule_claims() -> None:
    from osprey.deployment import status_display

    services = ["openobserve", "postgresql", "archive"]
    mine, unlabelled, foreign, by_name, others = status_display._partition_by_checkout(
        HOST_ROWS, "nobody-000000", PROJECT, services
    )
    listed = _names([*mine, *unlabelled, *foreign, *by_name])
    owned = deployment_containers(
        HOST_ROWS, project_name=PROJECT, services=services, identity="nobody-000000"
    )
    assert listed == _names(c.row for c in owned.ours)
    assert _names(others) == _names(owned.other_projects)


# ---------------------------------------------------------------------------
# Which checkout holds this project's name on the host
# ---------------------------------------------------------------------------

THIS_ID = "aaaaaaaaaaaa"
OTHER_ID = "bbbbbbbbbbbb"
OTHER_PATH = "/home/x/code/mine"


class _ListingProbe:
    """The two read-only listings :func:`host_claim` asks of a runtime probe."""

    runtime = "docker"

    def __init__(self, containers=(), volumes=()) -> None:
        self.containers = list(containers)
        self.volumes = list(volumes)
        self.asked: list[tuple[str, str]] = []

    def containers_for_project(self, project, *, include_stopped=True):
        self.asked.append(("containers", project, include_stopped))
        return list(self.containers)

    def volumes_for_project(self, project):
        self.asked.append(("volumes", project))
        return list(self.volumes)


def _container(name: str, repo_id: str | None = None, path: str | None = None) -> Resource:
    labels = {COMPOSE_PROJECT_LABEL: PROJECT}
    if repo_id is not None:
        labels[REPO_ID_LABEL] = repo_id
    if path is not None:
        labels["com.docker.compose.project.working_dir"] = path
    return Resource("container", name, labels)


@pytest.mark.parametrize(
    ("row_repo_id", "row_project", "repo_id", "expected"),
    [
        (THIS_ID, PROJECT, THIS_ID, True),
        (OTHER_ID, PROJECT, THIS_ID, False),
        (THIS_ID, "other", THIS_ID, True),
        ("", PROJECT, THIS_ID, True),
        (None, PROJECT, THIS_ID, True),
        ("", "other", THIS_ID, False),
        (OTHER_ID, PROJECT, "", True),
        ("", "", THIS_ID, False),
    ],
)
def test_claims_row_is_the_one_ownership_rule(row_repo_id, row_project, repo_id, expected):
    """Both ids present decide outright; otherwise the project name decides."""
    assert (
        claims_row(
            row_repo_id=row_repo_id, row_project=row_project, project=PROJECT, repo_id=repo_id
        )
        is expected
    )


def test_host_claim_partitions_containers_by_checkout_identity():
    ours = _container("mine-a", THIS_ID)
    theirs = _container("mine-b", OTHER_ID, OTHER_PATH)
    unlabelled = _container("mine-c")
    probe = _ListingProbe([ours, theirs, unlabelled])

    claim = host_claim(PROJECT, THIS_ID, probe=probe)

    assert claim.containers.ours == [ours]
    assert claim.foreign == [theirs]
    assert claim.containers.unidentified == [unlabelled]
    assert claim.held_elsewhere


def test_host_claim_never_partitions_volumes_whatever_label_they_carry():
    """Volumes belong to the project by name; an old repo-id label decides nothing."""
    old = Resource("volume", "mine_data", {COMPOSE_PROJECT_LABEL: PROJECT, REPO_ID_LABEL: OTHER_ID})
    probe = _ListingProbe([_container("mine-a", THIS_ID)], [old])

    claim = host_claim(PROJECT, THIS_ID, probe=probe)

    assert claim.volumes == (old,)
    assert claim.foreign == []
    assert not claim.held_elsewhere


def test_an_unlabelled_container_does_not_count_as_another_checkouts():
    claim = host_claim(PROJECT, THIS_ID, probe=_ListingProbe([_container("mine-c")]))
    assert not claim.held_elsewhere


def test_host_claim_can_ask_for_running_containers_only():
    probe = _ListingProbe()
    host_claim(PROJECT, THIS_ID, probe=probe, include_stopped=False)
    assert ("containers", PROJECT, False) in probe.asked


def test_the_start_refusal_names_the_other_copy_and_the_way_to_stop_it(tmp_path):
    other = tmp_path / "mine"
    other.mkdir()
    probe = _ListingProbe(
        [_container("mine-a", OTHER_ID, str(other)), _container("mine-b", OTHER_ID, str(other))],
        [Resource("volume", "mine_data", {COMPOSE_PROJECT_LABEL: PROJECT})],
    )
    claim = host_claim(PROJECT, THIS_ID, probe=probe)

    error = claim.refusal(START_REFUSAL, extra_remedy="give this copy its own name")

    assert isinstance(error, ForeignCheckoutError)
    assert error.summary("osprey up") == (
        "osprey up will not start over containers from another copy of this repo"
    )
    flowed = " ".join(error.cause.split())
    assert f"2 containers are named {PROJECT!r}" in flowed
    assert f"{other} (still on disk)" in flowed
    assert error.remedy == f"stop that deployment where it lives: `osprey down --repo {other}`"
    assert error.extra_remedy == "give this copy its own name"
    assert [r.name for r in error.resources] == ["mine-a", "mine-b"]
    # The volumes are evidence of sharing, never refused on.
    assert "mine_data" not in [r.name for r in error.resources]


def test_the_start_refusal_after_a_move_names_the_one_command_that_clears_it():
    probe = _ListingProbe([_container("mine-a", OTHER_ID, "/nowhere/mine")])
    error = host_claim(PROJECT, THIS_ID, probe=probe).refusal(START_REFUSAL)

    assert claim_gone(probe)
    assert "no such directory on this host now" in error.cause
    assert error.remedy == (
        "the copy they came from is no longer on this host, so remove its containers: "
        "`docker rm -f mine-a`. The project's volumes are untouched and keep their data."
    )


def claim_gone(probe) -> bool:
    return host_claim(PROJECT, THIS_ID, probe=probe).other_copy_gone


def test_the_other_copy_counts_as_gone_only_when_every_recorded_path_is(tmp_path):
    live = tmp_path / "mine"
    live.mkdir()
    gone = _container("mine-a", OTHER_ID, "/nowhere/mine")
    assert not claim_gone(_ListingProbe([gone, _container("mine-b", OTHER_ID, str(live))]))
    assert not claim_gone(_ListingProbe([_container("mine-c", OTHER_ID)]))
    assert not claim_gone(_ListingProbe([_container("mine-d", THIS_ID, "/nowhere/mine")]))
    assert claim_gone(_ListingProbe([gone]))
