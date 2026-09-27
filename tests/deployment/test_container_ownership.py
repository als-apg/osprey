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
    ClaimBasis,
    container_label,
    container_names,
    deployment_containers,
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
