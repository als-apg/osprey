"""``GET /api/audit/recent`` — the gated tail reader over this container's ledgers.

The route is the read half of the unified audit ledger: :mod:`osprey.audit.writer`
appends, this serves back the newest lines. Three properties are what the suite
exists to hold, and each one has been a real bug class somewhere:

* **It is a tier boundary, not a convenience.** An audit ledger names every
  refusal a deployment produced, the subjects they were about and the sessions
  they happened in. A tier that may not open the Config panel may not read the
  ledger either, so the route shares the Config panel's gate and its refusal
  verbatim — one switch, one wording, no second spelling to drift.
* **The directory is the process's own, never the request's.** Each container
  binds only ``var/audit/<its own identity>/`` and the deployment-wide view is
  host-side. So the reader resolves its directory exactly where the writer does
  and takes nothing path-shaped from a client: the ``surface`` filter is
  matched against the stems this container actually has, so even a guard bug
  cannot name a file outside the zone.
* **The read is bounded.** Ledgers grow without limit; a tail reader that
  loaded one to answer ``limit=10`` would be a memory fault waiting for a busy
  deployment. The window is derived from the requested limit and seeked to from
  the end, which makes a torn first line normal rather than exceptional — so a
  line that does not parse is skipped, never fatal.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from osprey.audit import writer
from osprey.audit.writer import ledger_path
from osprey.interfaces.web_terminal.routes import audit as audit_routes
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX
from osprey.interfaces.web_terminal.routes.config import _require_config_panel
from osprey.utils.identity import acting_identity

RECENT = "/api/audit/recent"

#: The identity this container writes (and therefore reads) under.
OWN_IDENTITY = "alice"

BASE_CONFIG = {
    "project_name": "audit-routes",
    "control_system": {"writes_enabled": False},
    "claude_code": {"default_model": "sonnet"},
}


def _record(ts: str, *, surface: str = "mcp", subject: str = "s", **extra) -> str:
    """One JSONL line in the shape :meth:`AuditEnvelope.to_dict` emits."""
    line = {
        "ts": ts,
        "surface": surface,
        "actor": OWN_IDENTITY,
        "posture": "readonly",
        "posture_source": "config",
        "session": None,
        "subject": subject,
        "decision": "refused",
        "reason": "denied",
    }
    line.update(extra)
    return json.dumps(line, separators=(",", ":"))


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """A project directory carrying a parseable ``config.yml``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False))
    return project


@pytest.fixture
def zone(tmp_path, monkeypatch) -> Path:
    """The audit zone, with this process's identity pinned to ``OWN_IDENTITY``.

    Redirects the two seams — the writer's ``audit_dir`` and the route's
    ``acting_identity`` — rather than standing up a rendered repo, which is
    exactly why the writer keeps ``audit_dir()`` as its own function.

    ``writer.audit_dir`` and not ``audit_routes.audit_dir``: the reader calls it
    through the writer module precisely so that this one patch moves reader and
    writer together. Patched on the route instead, a suite that also fires a
    recorder would write into the developer's live ``var/audit``.
    """
    root = tmp_path / "var" / "audit"
    root.mkdir(parents=True)
    monkeypatch.setattr(writer, "audit_dir", lambda: root)
    monkeypatch.setattr(audit_routes, "acting_identity", lambda: OWN_IDENTITY)
    return root


@pytest.fixture
def own_dir(zone) -> Path:
    """This container's own ledger directory, created."""
    own = zone / OWN_IDENTITY
    own.mkdir()
    return own


def _app(project_dir, *, config_panel_enabled: bool | None) -> FastAPI:
    """A routes-only app; ``None`` leaves the flag OFF ``app.state`` entirely."""
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = project_dir / "config.yml"
    app.state.project_cwd = str(project_dir)
    app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    if config_panel_enabled is not None:
        app.state.config_panel_enabled = config_panel_enabled
    return app


@pytest.fixture
def client(project_dir):
    """Config panel enabled — the tier that may read the ledger."""
    with TestClient(_app(project_dir, config_panel_enabled=True)) as client:
        yield client


@pytest.fixture
def disabled_client(project_dir):
    with TestClient(_app(project_dir, config_panel_enabled=False)) as client:
        yield client


@pytest.fixture
def default_client(project_dir):
    """No flag on state at all — the absent-key posture every plain app has."""
    with TestClient(_app(project_dir, config_panel_enabled=None)) as client:
        yield client


def _subjects(response) -> list[str]:
    return [record["subject"] for record in response.json()["records"]]


# ---- the gate ---- #


class TestTierGate:
    """The route is behind the Config panel tier, with that tier's refusal."""

    def test_a_disabled_panel_refuses_with_403(self, disabled_client, own_dir):
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z") + "\n")
        response = disabled_client.get(RECENT)
        assert response.status_code == 403

    def test_the_refusal_names_the_config_panel_switch(self, disabled_client, own_dir):
        detail = disabled_client.get(RECENT).json()["detail"]
        assert "web.config_panel.enabled: false" in detail

    def test_the_refusal_is_the_config_panels_own_wording(self, disabled_client):
        """Not a second spelling: the SAME gate produced it.

        A copy of the message would drift the first time the panel's wording is
        edited, and an operator would meet two stories about one switch.
        """
        with pytest.raises(HTTPException) as excinfo:
            _require_config_panel(_FakeRequest(config_panel_enabled=False))
        assert disabled_client.get(RECENT).json()["detail"] == excinfo.value.detail

    def test_the_gate_runs_before_the_read(self, disabled_client, zone):
        """No ledger directory at all — still a 403, never a 500 or a 200."""
        assert disabled_client.get(RECENT).status_code == 403

    def test_an_enabled_panel_serves_the_ledger(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z") + "\n")
        assert client.get(RECENT).status_code == 200

    def test_an_absent_flag_means_enabled(self, default_client, own_dir):
        """Matches every other Config-panel route: absent is not disabled."""
        assert default_client.get(RECENT).status_code == 200

    @pytest.mark.parametrize(
        "params",
        [{"surface": "../bob"}, {"surface": ""}, {"limit": "abc"}, {"limit": "1e9"}],
        ids=["traversal-surface", "empty-surface", "non-numeric-limit", "float-limit"],
    )
    def test_the_gate_answers_before_any_parameter_check(self, disabled_client, own_dir, params):
        """403, never 400 or 422 — the gate is the FIRST thing that answers.

        Every one of these values is refusable on its own merits, and a lower
        tier must not be able to tell them apart from a valid request: a 422
        for ``limit=abc`` would confirm the route exists and that it takes an
        integer ``limit``, to a caller who may not read the ledger at all.
        That is why ``limit`` arrives as a string and is parsed in the handler
        rather than annotated ``int``.
        """
        response = disabled_client.get(RECENT, params=params)
        assert response.status_code == 403, (params, response.text)

    def test_a_non_numeric_limit_is_refused_once_past_the_gate(self, client, own_dir):
        """Past the gate it is an ordinary bad request, with an honest message.

        Out of range clamps; not-a-number cannot, so it refuses — and says
        which of the two it is doing.
        """
        response = client.get(RECENT, params={"limit": "abc"})
        assert response.status_code == 400
        assert "limit" in response.json()["detail"]

    def test_a_numeric_limit_still_arrives_as_a_number(self, client, own_dir):
        """The string annotation is a gate-ordering device, not a behaviour change."""
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z") + "\n")
        assert client.get(RECENT, params={"limit": "7"}).json()["limit"] == 7


class _FakeRequest:
    """Minimal stand-in carrying only ``app.state``, for the gate comparison."""

    def __init__(self, *, config_panel_enabled: bool) -> None:
        state = type("State", (), {"config_panel_enabled": config_panel_enabled})()
        self.app = type("App", (), {"state": state})()


# ---- tail semantics ---- #


class TestTailSemantics:
    """Newest first, bounded, and never fatal on a bad line."""

    def test_records_come_back_newest_first(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(
            "\n".join(_record(f"2026-01-01T00:00:0{n}Z", subject=f"s{n}") for n in range(4)) + "\n"
        )
        assert _subjects(client.get(RECENT)) == ["s3", "s2", "s1", "s0"]

    def test_records_are_served_verbatim(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(
            _record("2026-01-01T00:00:00Z", detail="why", role="operator") + "\n"
        )
        [record] = client.get(RECENT).json()["records"]
        assert record["detail"] == "why"
        assert record["role"] == "operator"
        assert record["session"] is None

    def test_limit_caps_the_number_returned(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(
            "\n".join(_record(f"2026-01-01T00:00:{n:02d}Z", subject=f"s{n}") for n in range(10))
            + "\n"
        )
        assert _subjects(client.get(RECENT, params={"limit": 3})) == ["s9", "s8", "s7"]

    def test_the_default_limit_applies_without_a_param(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(
            "\n".join(
                _record("2026-01-01T00:00:00Z", subject=f"s{n}")
                for n in range(audit_routes.DEFAULT_LIMIT + 10)
            )
            + "\n"
        )
        assert len(client.get(RECENT).json()["records"]) == audit_routes.DEFAULT_LIMIT

    def test_an_oversized_limit_is_clamped_not_refused(self, client, own_dir):
        """Clamped inside the handler, so the GATE stays the first thing that runs.

        A ``Query(le=...)`` bound would answer a lower tier with 422 before the
        gate ever saw the request, which tells an unauthorized caller the route
        exists and what it takes.
        """
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z") + "\n")
        response = client.get(RECENT, params={"limit": 10_000})
        assert response.status_code == 200
        assert response.json()["limit"] == audit_routes.MAX_LIMIT

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_clamped_to_one(self, client, own_dir, limit):
        (own_dir / "mcp.jsonl").write_text(
            "\n".join(_record(f"2026-01-01T00:00:0{n}Z", subject=f"s{n}") for n in range(3)) + "\n"
        )
        response = client.get(RECENT, params={"limit": limit})
        assert response.json()["limit"] == 1
        assert _subjects(response) == ["s2"]

    def test_a_malformed_line_is_skipped_not_fatal(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(
            _record("2026-01-01T00:00:00Z", subject="good-0")
            + "\n{ this is not json\n"
            + _record("2026-01-01T00:00:02Z", subject="good-1")
            + "\n"
        )
        response = client.get(RECENT)
        assert response.status_code == 200
        assert _subjects(response) == ["good-1", "good-0"]

    def test_a_json_line_that_is_not_an_object_is_skipped(self, client, own_dir):
        """``[1,2]`` and ``"x"`` parse fine and are not records."""
        (own_dir / "mcp.jsonl").write_text(
            '[1,2]\n"x"\n' + _record("2026-01-01T00:00:00Z", subject="only") + "\n"
        )
        assert _subjects(client.get(RECENT)) == ["only"]

    def test_a_blank_line_is_skipped(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(
            "\n\n" + _record("2026-01-01T00:00:00Z", subject="only") + "\n\n"
        )
        assert _subjects(client.get(RECENT)) == ["only"]

    def test_a_final_line_without_a_newline_is_still_read(self, client, own_dir):
        """The writer's torn-write path can leave one; it is a whole record."""
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="tail"))
        assert _subjects(client.get(RECENT)) == ["tail"]

    def test_a_non_string_ts_sorts_oldest_not_lexicographically(self, client, own_dir):
        """A foreign line with e.g. an integer epoch ``ts`` must not float to
        the top: stringified and compared as text, an epoch like ``1735689600``
        would sort newer than every real ISO-8601 timestamp (``'1' > '2'`` is
        false, but the writer's records all start with ``'2'`` for the 2020s,
        and a leading digit is not the only way to sort out of place)."""
        (own_dir / "mcp.jsonl").write_text(
            _record("2026-01-01T00:00:00Z", subject="real").replace(
                '"ts":"2026-01-01T00:00:00Z"', '"ts":9999999999'
            )
            + "\n"
            + _record("2020-01-01T00:00:00Z", subject="older-real")
            + "\n"
        )
        assert _subjects(client.get(RECENT)) == ["older-real", "real"]

    def test_a_missing_ledger_directory_is_an_empty_list(self, client, zone):
        response = client.get(RECENT)
        assert response.status_code == 200
        assert response.json()["records"] == []

    def test_an_empty_ledger_directory_is_an_empty_list(self, client, own_dir):
        assert client.get(RECENT).json()["records"] == []

    def test_an_unresolvable_audit_zone_is_empty_not_a_500(self, client, monkeypatch):
        """A deployment with no resolvable project root downgrades to nothing.

        The reader must not be the thing that takes the web terminal down; the
        writer swallows its own errors for the same reason.
        """

        def boom():
            raise RuntimeError("no project root")

        monkeypatch.setattr(writer, "audit_dir", boom)
        monkeypatch.setattr(audit_routes, "acting_identity", lambda: OWN_IDENTITY)
        response = client.get(RECENT)
        assert response.status_code == 200
        assert response.json()["records"] == []


class TestTheReadIsBounded:
    """A ledger is unbounded; the read is not."""

    def test_only_the_tail_window_is_read(self, client, own_dir, monkeypatch):
        """The newest records come back without the whole file being loaded."""
        ledger = own_dir / "mcp.jsonl"
        ledger.write_text(
            "\n".join(
                _record("2026-01-01T00:00:00Z", subject=f"s{n:05d}", detail="x" * 400)
                for n in range(4000)
            )
            + "\n"
        )
        assert ledger.stat().st_size > 1_000_000

        reads: list[int] = []
        real_read = audit_routes._tail_bytes

        def spy(path, budget):
            data, truncated = real_read(path, budget)
            reads.append(len(data))
            return data, truncated

        monkeypatch.setattr(audit_routes, "_tail_bytes", spy)
        response = client.get(RECENT, params={"limit": 5})

        assert _subjects(response) == [f"s{n:05d}" for n in (3999, 3998, 3997, 3996, 3995)]
        assert reads and max(reads) < ledger.stat().st_size

    def test_a_partial_first_line_is_dropped_rather_than_parsed(self, client, own_dir):
        """Seeking into the middle of a line must not manufacture a record."""
        ledger = own_dir / "mcp.jsonl"
        ledger.write_text(
            _record("2026-01-01T00:00:00Z", subject="old", detail="y" * 900)
            + "\n"
            + _record("2026-01-01T00:00:01Z", subject="new")
            + "\n"
        )
        response = client.get(RECENT, params={"limit": 1})
        assert response.status_code == 200
        assert _subjects(response) == ["new"]

    def test_a_fragment_that_parses_is_still_not_served_as_a_record(self, client, own_dir):
        """The case the explicit partial-line drop exists for.

        Usually a fragment is self-evidently junk — the tail of a JSON object
        is not a JSON object, so the parse skip catches it. The exception is a
        seek that lands EXACTLY on a record boundary with junk ahead of it on
        the same line, which a torn append whose terminator did not land can
        leave behind. Then the fragment parses cleanly into a dict, and only
        dropping the first line keeps the reader from manufacturing a record it
        never saw the start of.

        The offsets are computed, not guessed: the record is padded so the
        window boundary falls on its opening brace.
        """
        budget = 1 * audit_routes.MAX_RECORD_BYTES + audit_routes.TAIL_SLACK_BYTES
        bare = _record("2026-01-01T00:00:00Z", subject="forged", detail="")
        # Pad ``detail`` until the encoded record is exactly ``budget - 1``
        # bytes, so that ``size - budget`` lands on its first byte.
        forged = _record(
            "2026-01-01T00:00:00Z", subject="forged", detail="p" * (budget - 1 - len(bare))
        )
        assert len(forged) == budget - 1

        ledger = own_dir / "mcp.jsonl"
        ledger.write_text("G" * 5000 + forged + "\n")
        assert ledger.stat().st_size - budget == 5000

        response = client.get(RECENT, params={"limit": 1})
        assert response.status_code == 200
        assert _subjects(response) == [], "a fragment was served as a record"

    def test_a_file_exactly_one_window_wide_keeps_its_first_line(self, client, own_dir):
        """The seek lands on byte 0 here, so nothing was torn off to read it.

        Regression for the off-by-one at ``len(raw) == budget``: that length
        is reached BOTH when the file is exactly one window wide (seek
        position 0, whole file read, first line is real) and when a larger
        file is read from a non-zero offset (first line is a fragment).
        ``_tail_bytes`` disambiguates via the seek position, not the byte
        count, so this whole-file case must not drop its first record.
        """
        budget = 1 * audit_routes.MAX_RECORD_BYTES + audit_routes.TAIL_SLACK_BYTES
        bare = _record("2026-01-01T00:00:00Z", subject="whole", detail="")
        # Pad so the file (including its trailing newline) is exactly `budget`
        # bytes: the seek then lands on byte 0, not mid-file.
        record = _record(
            "2026-01-01T00:00:00Z", subject="whole", detail="p" * (budget - 1 - len(bare))
        )
        assert len(record) == budget - 1

        ledger = own_dir / "mcp.jsonl"
        ledger.write_text(record + "\n")
        assert ledger.stat().st_size == budget

        response = client.get(RECENT, params={"limit": 1})
        assert response.status_code == 200
        assert _subjects(response) == ["whole"], "a whole record at the window boundary was dropped"

    def test_the_number_of_ledgers_read_is_capped(self, client, own_dir):
        for n in range(audit_routes.MAX_LEDGERS + 20):
            (own_dir / f"surface{n:03d}.jsonl").write_text(
                _record("2026-01-01T00:00:00Z", surface=f"surface{n:03d}") + "\n"
            )
        assert len(client.get(RECENT).json()["ledgers"]) == audit_routes.MAX_LEDGERS


# ---- the directory is the container's own ---- #


class TestReadsOnlyItsOwnSubdir:
    """Nothing path-shaped reaches the filesystem from a request."""

    def test_the_reader_and_the_writer_resolve_the_same_directory(self):
        """Pinned by construction, not by two literals that happen to agree.

        Deliberately UNPATCHED: the seams are redirected everywhere else in
        this file, and a pin that compares two redirected values proves only
        that the redirect worked. Here both sides run the real resolvers — the
        writer's ``audit_dir()`` and the shared ``acting_identity()`` ladder —
        so the assertion is that the reader derives its directory the same way
        the writer does, whatever those resolve to on this machine.
        """
        assert audit_routes.identity_dir() == ledger_path("mcp").parent
        assert audit_routes.identity_dir() == ledger_path("sidecar").parent
        assert audit_routes.identity_dir().name == acting_identity()

    def test_the_writers_own_seam_redirects_the_reader_too(self, tmp_path, monkeypatch):
        """One ``writer.audit_dir`` patch moves both halves, which is the point.

        ``osprey.audit.writer`` documents that function as THE seam, and every
        other suite in this epic redirects it by name. Bound into this module
        by a from-import, the route would keep resolving the real ``var/audit``
        while the writer wrote into ``tmp_path`` — a reader that silently
        answers from the developer's or the CI runner's live ledger.
        """
        root = tmp_path / "zone"
        root.mkdir()
        monkeypatch.setattr(writer, "audit_dir", lambda: root)

        assert audit_routes.identity_dir().parent == root
        assert audit_routes.identity_dir() == ledger_path("mcp").parent

    def test_another_identitys_ledger_is_not_read(self, client, own_dir, zone):
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="mine") + "\n")
        other = zone / "bob"
        other.mkdir()
        (other / "mcp.jsonl").write_text(_record("2026-01-01T00:00:09Z", subject="theirs") + "\n")
        assert _subjects(client.get(RECENT)) == ["mine"]

    def test_the_response_names_the_identity_it_read(self, client, own_dir):
        assert client.get(RECENT).json()["identity"] == OWN_IDENTITY

    def test_nested_directories_are_not_walked(self, client, own_dir):
        nested = own_dir / "nested"
        nested.mkdir()
        (nested / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="deep") + "\n")
        assert client.get(RECENT).json()["records"] == []

    def test_non_ledger_files_are_ignored(self, client, own_dir):
        (own_dir / "notes.txt").write_text("not a ledger\n")
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="only") + "\n")
        assert _subjects(client.get(RECENT)) == ["only"]

    def test_a_directory_named_like_a_ledger_is_not_a_ledger(self, client, own_dir):
        """A directory that happens to end in ``.jsonl`` is not a ledger.

        Mutation-tested: dropping the ``is_file()`` guard in ``_ledgers`` left
        the rest of the suite green, because the decoy would fail to open and
        degrade to an empty read for it — but its phantom stem still leaked
        into the response's ``ledgers`` list, which this pins against.
        """
        (own_dir / "decoy.jsonl").mkdir()
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="only") + "\n")
        assert client.get(RECENT).json()["ledgers"] == ["mcp"]

    def test_a_symlink_named_like_a_ledger_is_not_a_ledger(self, client, own_dir, zone):
        """The other half of the same guard, and the one that leaks records.

        ``is_file()`` FOLLOWS symlinks, so it alone enumerates a link named
        ``<anything>.jsonl`` and serves whatever it resolves to — here another
        identity's ledger, which the whole route exists not to read. A decoy
        directory only leaks a phantom stem; a symlink leaks the records.
        """
        other = zone / "bob"
        other.mkdir()
        (other / "secret.jsonl").write_text(
            _record("2026-01-01T00:00:09Z", subject="theirs") + "\n"
        )
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="mine") + "\n")
        (own_dir / "linked.jsonl").symlink_to(other / "secret.jsonl")

        response = client.get(RECENT)
        assert response.json()["ledgers"] == ["mcp"]
        assert _subjects(response) == ["mine"]

    def test_a_symlink_to_a_ledger_in_this_very_directory_is_still_not_one(self, client, own_dir):
        """No exception for a link that happens to resolve inside the zone.

        The rule is "a ledger is a real file that lives here", not "a ledger is
        anything that resolves somewhere allowed" — a target-based rule is the
        one that has to be right about resolution, and this one does not.
        """
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="mine") + "\n")
        (own_dir / "alias.jsonl").symlink_to(own_dir / "mcp.jsonl")

        response = client.get(RECENT)
        assert response.json()["ledgers"] == ["mcp"]
        assert _subjects(response) == ["mine"], "the record was served twice"


class TestSurfaceFilter:
    """The one client-supplied string, and it never names a path."""

    def test_a_surface_filter_selects_one_ledger(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="m") + "\n")
        (own_dir / "sidecar.jsonl").write_text(_record("2026-01-01T00:00:09Z", subject="s") + "\n")
        assert _subjects(client.get(RECENT, params={"surface": "mcp"})) == ["m"]

    def test_an_unknown_surface_is_an_empty_list(self, client, own_dir):
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z") + "\n")
        response = client.get(RECENT, params={"surface": "nosuch"})
        assert response.status_code == 200
        assert response.json()["records"] == []

    @pytest.mark.parametrize(
        "value",
        [
            "../bob/mcp",
            "..",
            ".",
            "/etc/passwd",
            "mcp/../../bob/mcp",
            "a\\b",
            "mcp\x00",
            "",
            " ",
            "x" * 300,
        ],
    )
    def test_a_path_ish_surface_is_refused(self, client, own_dir, value):
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z") + "\n")
        response = client.get(RECENT, params={"surface": value})
        assert response.status_code == 400, value

    def test_traversal_cannot_reach_another_identity_even_if_the_guard_slipped(
        self, client, own_dir, zone
    ):
        """Belt and braces: the filter matches STEMS this container enumerated.

        The guard above refuses the value outright; this asserts the second
        line of defence — the path is never built from the parameter, so a
        guard that let something through still names nothing outside the zone.
        """
        other = zone / "bob"
        other.mkdir()
        (other / "mcp.jsonl").write_text(_record("2026-01-01T00:00:09Z", subject="theirs") + "\n")
        (own_dir / "mcp.jsonl").write_text(_record("2026-01-01T00:00:00Z", subject="mine") + "\n")

        selected = audit_routes._select(own_dir, "../bob/mcp")
        assert selected == []

    def test_the_refusal_explains_the_rule(self, client, own_dir):
        detail = client.get(RECENT, params={"surface": "../bob"}).json()["detail"]
        assert "surface" in detail


# ---- the tier walk sees it ---- #


class TestTheTierWalkCoversTheRoute:
    """The completeness walk must actually govern this prefix."""

    def test_the_walk_includes_the_audit_prefix(self):
        from tests.interfaces.web_terminal.test_tier_gate_completeness import (
            ALL_VERB_PREFIXES,
        )

        assert "/api/audit" in ALL_VERB_PREFIXES

    def test_the_route_floor_counts_the_new_route(self):
        from tests.interfaces.web_terminal.test_tier_gate_completeness import (
            MIN_GATED_ROUTES,
        )

        assert MIN_GATED_ROUTES >= 13
