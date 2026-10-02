"""The Teams bridge's Graph leg: upload a file into the library, share a folder.

Every request is answered by an :class:`httpx.MockTransport`; no socket is opened.
The token leg is a stub :class:`TokenSource` stand-in answering one fixed bearer, so
each test asserts only what reaches Graph.
"""

from __future__ import annotations

import json
from urllib.parse import unquote

import httpx
import pytest

from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.graph import (
    MAX_INVITE_RECIPIENTS,
    GraphError,
    GraphFiles,
    UploadedFile,
    run_folder,
)

DRIVE = "b!drive-id_1"
BEARER = "graph-bearer-value"


class StubTokens:
    """Answers one fixed Graph bearer."""

    def token(self) -> str:
        return BEARER


def make_config(cloud: str = "commercial", folder: str = "osprey/answers") -> TeamsBridgeConfig:
    return TeamsBridgeConfig(
        app_id="app", tenant_id="tenant", cloud=cloud, files_drive_id=DRIVE, files_folder=folder
    )


def drive_item(name: str = "table.csv", **overrides) -> dict:
    item = {
        "id": "item-1",
        "name": name,
        "parentReference": {"id": "folder-1", "driveId": DRIVE},
        "webUrl": f"https://tenant.sharepoint.com/sites/osprey/files/{name}",
    }
    item.update(overrides)
    return item


class GraphEndpoint:
    """A recording stand-in for the two Graph routes the bridge calls."""

    def __init__(self, *, upload=None, invite=None, raises: Exception | None = None) -> None:
        self.upload = upload or (lambda request: httpx.Response(201, json=drive_item()))
        self.invite = invite or (
            lambda request: httpx.Response(200, json={"value": [{"id": "perm-1"}]})
        )
        self.raises = raises
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if request.method == "PUT":
            return self.upload(request)
        return self.invite(request)


def make_files(endpoint: GraphEndpoint, cfg: TeamsBridgeConfig | None = None) -> GraphFiles:
    http = httpx.Client(transport=httpx.MockTransport(endpoint))
    return GraphFiles(cfg or make_config(), StubTokens(), http)  # type: ignore[arg-type]


# --- upload ----------------------------------------------------------------


def test_an_upload_puts_the_bytes_at_the_runs_path_in_the_library():
    endpoint = GraphEndpoint()
    files = make_files(endpoint)

    uploaded = files.upload(("osprey", "answers", "R1"), "table.csv", b"a,b\n1,2\n", "text/csv")

    request = endpoint.requests[0]
    assert request.method == "PUT"
    assert str(request.url) == (
        f"https://graph.microsoft.com/v1.0/drives/{DRIVE}/root:/osprey/answers/R1/table.csv:/content"
    )
    assert request.content == b"a,b\n1,2\n"
    assert uploaded == UploadedFile(
        name="table.csv",
        item_id="item-1",
        folder_id="folder-1",
        web_url="https://tenant.sharepoint.com/sites/osprey/files/table.csv",
    )


@pytest.mark.parametrize(
    ("cloud", "host"), [("commercial", "graph.microsoft.com"), ("gcchigh", "graph.microsoft.us")]
)
def test_the_upload_url_uses_the_clouds_graph_host(cloud, host):
    endpoint = GraphEndpoint()
    make_files(endpoint, make_config(cloud)).upload(("R1",), "a.pdf", b"%PDF", "application/pdf")
    assert endpoint.requests[0].url.host == host


def test_path_segments_and_the_name_are_percent_encoded():
    endpoint = GraphEndpoint()
    make_files(endpoint).upload(("Osprey files", "R#1"), "a b?.csv", b"x", "text/csv")

    raw = endpoint.requests[0].url.raw_path.decode()
    assert "/root:/Osprey%20files/R%231/a%20b%3F.csv:/content" in raw
    assert unquote(raw).endswith("/root:/Osprey files/R#1/a b?.csv:/content")


def test_an_upload_carries_the_graph_bearer():
    endpoint = GraphEndpoint()
    make_files(endpoint).upload(("R1",), "a.csv", b"x", "text/csv")
    assert endpoint.requests[0].headers["Authorization"] == f"Bearer {BEARER}"


@pytest.mark.parametrize(
    ("served", "sent"),
    [
        ("text/csv", "text/csv"),
        ("application/pdf", "application/pdf"),
        (None, "application/octet-stream"),
    ],
)
def test_an_upload_sends_the_served_content_type_or_octet_stream(served, sent):
    endpoint = GraphEndpoint()
    make_files(endpoint).upload(("R1",), "a.bin", b"x", served)
    assert endpoint.requests[0].headers["Content-Type"] == sent


@pytest.mark.parametrize(
    "item",
    [
        drive_item(webUrl="http://tenant.sharepoint.com/a.csv"),
        drive_item(webUrl=None),
        {k: v for k, v in drive_item().items() if k != "webUrl"},
        {k: v for k, v in drive_item().items() if k != "id"},
        {k: v for k, v in drive_item().items() if k != "parentReference"},
    ],
)
def test_an_upload_answer_without_an_https_web_url_is_an_error(item):
    endpoint = GraphEndpoint(upload=lambda request: httpx.Response(201, json=item))
    with pytest.raises(GraphError):
        make_files(endpoint).upload(("R1",), "a.csv", b"x", "text/csv")


def test_a_non_2xx_upload_raises_graph_error_with_a_bounded_body():
    endpoint = GraphEndpoint(upload=lambda request: httpx.Response(403, text="E" * 5000))
    with pytest.raises(GraphError) as excinfo:
        make_files(endpoint).upload(("R1",), "a.csv", b"x", "text/csv")
    message = str(excinfo.value)
    assert "403" in message
    assert message.count("E") <= 300


def test_a_transport_failure_raises_graph_error():
    endpoint = GraphEndpoint(raises=httpx.ConnectError("no route to host"))
    with pytest.raises(GraphError, match="no route to host"):
        make_files(endpoint).upload(("R1",), "a.csv", b"x", "text/csv")


# --- share -----------------------------------------------------------------


def test_a_share_invites_by_directory_id_read_only_sign_in_required_and_mails_no_one():
    endpoint = GraphEndpoint()
    make_files(endpoint).share("folder-1", ["oid-a", "oid-b"])

    (request,) = endpoint.requests
    assert request.method == "POST"
    assert str(request.url) == (
        f"https://graph.microsoft.com/v1.0/drives/{DRIVE}/items/folder-1/invite"
    )
    assert request.headers["Authorization"] == f"Bearer {BEARER}"
    assert json.loads(request.content) == {
        "recipients": [{"objectId": "oid-a"}, {"objectId": "oid-b"}],
        "requireSignIn": True,
        "sendInvitation": False,
        "roles": ["read"],
    }
    assert "createLink" not in str(request.url)


def test_a_share_splits_its_recipients_at_the_request_limit():
    endpoint = GraphEndpoint()
    ids = [f"oid-{n}" for n in range(MAX_INVITE_RECIPIENTS * 2 + 1)]
    make_files(endpoint).share("folder-1", ids)

    batches = [json.loads(request.content)["recipients"] for request in endpoint.requests]
    assert [len(batch) for batch in batches] == [MAX_INVITE_RECIPIENTS, MAX_INVITE_RECIPIENTS, 1]
    assert [r["objectId"] for batch in batches for r in batch] == ids


def test_a_partial_share_is_an_error():
    body = {"value": [{"id": "perm-1"}, {"error": {"code": "notAllowed", "message": "guest"}}]}
    endpoint = GraphEndpoint(invite=lambda request: httpx.Response(207, json=body))
    with pytest.raises(GraphError, match="notAllowed"):
        make_files(endpoint).share("folder-1", ["oid-a", "oid-b"])


@pytest.mark.parametrize(
    "response",
    [
        lambda request: httpx.Response(403, text="accessDenied"),
        lambda request: httpx.Response(200, text="<html>"),
    ],
)
def test_a_failed_share_request_is_an_error(response):
    endpoint = GraphEndpoint(invite=response)
    with pytest.raises(GraphError):
        make_files(endpoint).share("folder-1", ["oid-a"])


def test_a_graph_error_never_carries_the_bearer():
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad token {request.headers['Authorization']}")

    endpoint = GraphEndpoint(upload=echo, invite=echo)
    files = make_files(endpoint)
    for call in (
        lambda: files.upload(("R1",), "a.csv", b"x", "text/csv"),
        lambda: files.share("folder-1", ["oid-a"]),
    ):
        with pytest.raises(GraphError) as excinfo:
            call()
        assert BEARER not in str(excinfo.value)


# --- the run's folder ------------------------------------------------------


def test_run_folder_appends_the_run_to_the_configured_folder():
    assert run_folder(make_config(folder="osprey/answers"), "R-1.a_b") == (
        "osprey",
        "answers",
        "R-1.a_b",
    )
    assert run_folder(make_config(folder=""), "R1") == ("R1",)


@pytest.mark.parametrize("run_id", ["a/b", "..", ".", ""])
def test_run_folder_refuses_a_run_id_that_is_not_one_segment(run_id):
    with pytest.raises(ValueError):
        run_folder(make_config(), run_id)
