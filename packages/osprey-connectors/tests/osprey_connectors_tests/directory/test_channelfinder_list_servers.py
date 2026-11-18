"""ChannelFinder lists every IOC, past the service's result window, or says it cannot."""

from __future__ import annotations

import time
import urllib.parse
from typing import Any

from osprey_connectors._service_http import ReadOnlyTransport
from osprey_connectors.directory import ServerRow
from osprey_connectors.directory._channelfinder import ChannelFinderDirectory
from osprey_connectors.service_answers import BackendSpec, Unavailable

DEADLINE = 6.0
CHANNELS = "/resources/channels"
SCROLL = "/resources/scroll"
#: Elasticsearch's default ``index.max_result_window``: ``~from`` + ``~size`` may not pass it.
WINDOW = 10_000
REFUSED_PAST_WINDOW = Unavailable("answered HTTP 500")


def channel(name: str, ioc: str | None, *, host: str | None = None, status: str = "Active"):
    properties = [{"name": "pvStatus", "value": status}]
    if ioc is not None:
        properties.append({"name": "iocName", "value": ioc})
    if host is not None:
        properties.append({"name": "hostName", "value": host})
    return {"name": name, "properties": properties, "tags": []}


def big_directory(iocs: int = 30, per_server: int = 450) -> list[dict[str, Any]]:
    """More channels than the result window: *iocs* IOCs of *per_server* PVs, every third Inactive."""
    rows = []
    for i in range(iocs):
        for j in range(per_server):
            rows.append(
                channel(
                    f"SR{i:02d}:PV{j:04d}",
                    f"ioc-{i:02d}",
                    host=f"host-{i % 7}",
                    status="Inactive" if j % 3 == 0 else "Active",
                )
            )
    return rows


class ChannelFinderFake:
    """A ChannelFinder over *channels* with Elasticsearch's result window.

    ``channels`` pages by ``~size``/``~from`` and refuses a page reaching past the
    window, as the service does. ``scroll`` answers ``{id, channels}`` in name
    order, the id being the last name, and ``scroll/{id}`` continues after it.

    ``clamp`` serves at most that many channels per scroll page whatever ``~size``
    asks for. Every request the real transport would refuse for the backend's
    ``ALLOWED_REQUESTS`` is refused here too, by asking the real transport's check;
    ``real_paths=False`` turns that off for a test about something else.
    """

    def __init__(
        self,
        channels: list[dict[str, Any]],
        *,
        window: int = WINDOW,
        clamp: int | None = None,
        real_paths: bool = True,
    ) -> None:
        self.channels = sorted(channels, key=lambda c: c["name"])
        self.window = window
        self.clamp = clamp
        self.path_check = (
            ReadOnlyTransport(
                {"url": "https://cf.demo.lab/ChannelFinder"},
                ChannelFinderDirectory.ALLOWED_REQUESTS,
                where="directory.channelfinder",
            )
            if real_paths
            else None
        )
        self.sent: list[tuple[str, str, list[tuple[str, Any]]]] = []
        self.deadlines: list[float] = []
        self.fail_at: int | None = None
        self.answer_override: dict[int, Any] = {}
        self.delay_s = 0.0

    async def request(self, method, path, *, params=None, deadline, **_body):
        pairs = list(params or [])
        index = len(self.sent)
        self.sent.append((method, path, pairs))
        self.deadlines.append(deadline)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.path_check is not None and not self.path_check._allows(method, path):
            return Unavailable(f"refused: {method} {path} is not an allowed request")
        if index in self.answer_override:
            return self.answer_override[index]
        if self.fail_at == index:
            return Unavailable("no answer within 6 s")
        size = int(dict(pairs).get("~size", 10_000))
        if size > self.window:
            return REFUSED_PAST_WINDOW
        if path == CHANNELS:
            start = int(dict(pairs).get("~from", 0))
            if start + size > self.window:
                return REFUSED_PAST_WINDOW
            return self.channels[start : start + size]
        if path == SCROLL or path.startswith(SCROLL + "/"):
            after = urllib.parse.unquote(path[len(SCROLL) + 1 :]) if path != SCROLL else None
            rest = [c for c in self.channels if after is None or c["name"] > after]
            page = rest[: min(size, self.clamp or size)]
            return {"id": page[-1]["name"] if page else None, "channels": page}
        return Unavailable("answered HTTP 404")


def backend(fake: ChannelFinderFake) -> ChannelFinderDirectory:
    spec = BackendSpec(
        kind="directory",
        type="channelfinder",
        target="live",
        settings={"url": "https://cf.demo.lab/ChannelFinder"},
        rehearsal=False,
    )
    return ChannelFinderDirectory(spec, transport=fake)


async def naive_offset_pager(fake: ChannelFinderFake, size: int = 1000) -> list | Unavailable:
    """What a pager over ``~from`` gets from a directory past the window."""
    rows: list = []
    while True:
        page = await fake.request(
            "GET",
            CHANNELS,
            params=[("~name", "*"), ("~size", size), ("~from", len(rows))],
            deadline=DEADLINE,
        )
        if isinstance(page, Unavailable):
            return page
        rows.extend(page)
        if len(page) < size:
            return rows


class TestTheFakeHasTheCeiling:
    async def test_a_naive_offset_pager_is_refused_past_the_window(self):
        fake = ChannelFinderFake(big_directory())

        assert isinstance(await naive_offset_pager(fake), Unavailable)

    async def test_a_directory_inside_the_window_pages_by_offset(self):
        fake = ChannelFinderFake(big_directory(iocs=2, per_server=100))

        assert len(await naive_offset_pager(fake)) == 200


class TestListServers:
    async def test_every_server_of_a_directory_past_the_window_is_listed(self):
        fake = ChannelFinderFake(big_directory())

        rows = await backend(fake).list_servers(DEADLINE)

        assert not isinstance(rows, Unavailable), rows
        assert len(fake.channels) > WINDOW
        assert [row.name for row in rows] == [f"ioc-{i:02d}" for i in range(30)]
        assert sum(row.channel_count for row in rows) == len(fake.channels)
        assert rows[0] == ServerRow(
            name="ioc-00", host="host-0", channel_count=450, inactive_count=150
        )
        assert all(row.inactive_count == 150 for row in rows)

    async def test_it_pages_over_the_scroll_route_never_by_offset(self):
        fake = ChannelFinderFake(big_directory())

        await backend(fake).list_servers(DEADLINE)

        paths = [path for _, path, _ in fake.sent]
        assert paths[0] == SCROLL
        assert all(path.startswith(SCROLL + "/") for path in paths[1:])
        assert len(paths) > 1
        assert not any(name == "~from" for _, _, pairs in fake.sent for name, _ in pairs)
        assert all(method == "GET" for method, _, _ in fake.sent)

    async def test_the_scroll_id_is_one_path_segment_even_for_a_channel_name(self):
        fake = ChannelFinderFake(
            [channel(f"SR:A?B#C%{i:05d}.X", "odd", host="h") for i in range(6000)]
        )

        rows = await backend(fake).list_servers(DEADLINE)

        assert rows == [ServerRow(name="odd", host="h", channel_count=6000, inactive_count=0)]
        continued = [path for _, path, _ in fake.sent[1:]]
        assert continued and all("/" not in path[len(SCROLL) + 1 :] for path in continued)
        assert all("?" not in path and "#" not in path for path in continued)

    async def test_a_scroll_id_the_transport_refuses_gives_no_list_rather_than_part(self):
        # The transport never sends a segment holding '/' once decoded, so a page
        # ending on such a name cannot be continued: the answer says so.
        fake = ChannelFinderFake([channel(f"SR/A{i:05d}", "odd", host="h") for i in range(6000)])

        rows = await backend(fake).list_servers(DEADLINE)

        assert isinstance(rows, Unavailable)
        assert "refused" in rows.reason and "5000 PVs" in rows.reason

    async def test_the_real_transport_lets_the_scroll_continue_after_a_channel_name(self):
        fake = ChannelFinderFake(
            [channel(f"SR01:BPM{i:05d}.X", "bpm", host="h") for i in range(6000)],
            real_paths=True,
        )

        rows = await backend(fake).list_servers(DEADLINE)

        assert rows == [ServerRow(name="bpm", host="h", channel_count=6000, inactive_count=0)]

    async def test_the_scroll_routes_are_allowed_requests(self):
        allowed = ChannelFinderDirectory.ALLOWED_REQUESTS

        assert ("GET", SCROLL) in allowed
        assert ("GET", SCROLL + "/*") in allowed

    async def test_rows_are_unique_by_name_and_sorted(self):
        fake = ChannelFinderFake(
            [
                channel("B:1", "zeta", host="h1"),
                channel("A:1", "alpha", host="h2"),
                channel("C:1", "zeta", host="h1", status="Inactive"),
            ]
        )

        rows = await backend(fake).list_servers(DEADLINE)

        assert rows == [
            ServerRow(name="alpha", host="h2", channel_count=1, inactive_count=0),
            ServerRow(name="zeta", host="h1", channel_count=2, inactive_count=1),
        ]

    async def test_a_channel_with_no_server_name_belongs_to_no_row(self):
        fake = ChannelFinderFake(
            [channel("A:1", None, host="h"), channel("B:1", "server", host="h")]
        )

        rows = await backend(fake).list_servers(DEADLINE)

        assert rows == [ServerRow(name="server", host="h", channel_count=1, inactive_count=0)]

    async def test_a_server_with_no_host_reports_none(self):
        fake = ChannelFinderFake([channel("A:1", "server")])

        rows = await backend(fake).list_servers(DEADLINE)

        assert rows == [ServerRow(name="server", host=None, channel_count=1, inactive_count=0)]

    async def test_the_host_of_its_active_channels_wins_over_a_stale_one(self):
        fake = ChannelFinderFake(
            [
                channel("A:1", "server", host="old-host", status="Inactive"),
                channel("A:2", "server", host="old-host", status="Inactive"),
                channel("A:3", "server", host="new-host"),
            ]
        )

        rows = await backend(fake).list_servers(DEADLINE)

        assert rows[0].host == "new-host"

    async def test_a_service_serving_short_pages_is_still_listed_whole(self):
        fake = ChannelFinderFake(big_directory(), clamp=1000)

        rows = await backend(fake).list_servers(DEADLINE)

        assert not isinstance(rows, Unavailable), rows
        assert sum(row.channel_count for row in rows) == len(fake.channels)
        assert len(rows) == 30

    async def test_a_short_page_with_no_id_ends_the_listing(self):
        fake = ChannelFinderFake(big_directory(iocs=2, per_server=10))
        fake.answer_override[0] = {"id": None, "channels": fake.channels}

        rows = await backend(fake).list_servers(DEADLINE)

        assert not isinstance(rows, Unavailable), rows
        assert sum(row.channel_count for row in rows) == 20
        assert len(fake.sent) == 1

    async def test_an_empty_directory_lists_no_servers(self):
        rows = await backend(ChannelFinderFake([])).list_servers(DEADLINE)

        assert rows == []


class TestCompleteOrNotGiven:
    async def test_a_page_failing_mid_scroll_answers_unavailable_not_part(self):
        fake = ChannelFinderFake(big_directory())
        fake.fail_at = 1

        answer = await backend(fake).list_servers(DEADLINE)

        assert isinstance(answer, Unavailable)
        assert "no answer within 6 s" in answer.reason

    async def test_a_malformed_page_answers_unavailable(self):
        fake = ChannelFinderFake(big_directory())
        fake.answer_override[1] = {"id": "x", "channels": "not a list"}

        answer = await backend(fake).list_servers(DEADLINE)

        assert isinstance(answer, Unavailable)

    async def test_a_body_that_is_not_a_scroll_answers_unavailable(self):
        fake = ChannelFinderFake(big_directory())
        fake.answer_override[0] = []

        assert isinstance(await backend(fake).list_servers(DEADLINE), Unavailable)

    async def test_a_scroll_that_does_not_advance_answers_unavailable(self):
        fake = ChannelFinderFake(big_directory())
        first = fake.channels[:5000]
        for index in range(1, 50):
            fake.answer_override[index] = {"id": first[-1]["name"], "channels": first}
        fake.answer_override[0] = {"id": first[-1]["name"], "channels": first}

        answer = await backend(fake).list_servers(DEADLINE)

        assert isinstance(answer, Unavailable)
        assert len(fake.sent) <= 3

    async def test_a_full_page_with_no_id_answers_unavailable(self):
        fake = ChannelFinderFake(big_directory())
        fake.answer_override[0] = {"id": None, "channels": fake.channels[:5000]}

        assert isinstance(await backend(fake).list_servers(DEADLINE), Unavailable)

    async def test_a_deadline_miss_answers_unavailable(self):
        fake = ChannelFinderFake(big_directory())
        fake.delay_s = 0.02

        answer = await backend(fake).list_servers(0.03)

        assert isinstance(answer, Unavailable)
        assert "0.03 s" in answer.reason

    async def test_each_page_spends_only_what_the_earlier_ones_left(self):
        fake = ChannelFinderFake(big_directory())
        fake.delay_s = 0.01

        await backend(fake).list_servers(DEADLINE)

        assert fake.deadlines[0] <= DEADLINE
        assert all(
            later < earlier
            for earlier, later in zip(fake.deadlines, fake.deadlines[1:], strict=False)
        )
        assert fake.deadlines[-1] <= DEADLINE - 0.01 * (len(fake.deadlines) - 1)
