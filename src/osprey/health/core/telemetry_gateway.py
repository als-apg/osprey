"""Core ``telemetry_gateway`` health category.

Rows **only when ``telemetry_gateway`` is in ``deployed_services``**; the
category stays a valid ``--category`` name at all times. It reads the gateway's
own counters (the Collector's Prometheus page) and checks them in two tiers.

Stateless tier — everywhere the category runs
---------------------------------------------
Including inside agent containers, over the network alias. Nothing is written.

* ``telemetry_gateway_metrics`` — the counters page answers (``warning`` when it
  does not; the counter rows below are then absent).
* ``telemetry_gateway_queue`` — ``warning`` when an exporter's queue holds more
  than 80 % of its capacity. A stop with a full ledger queue can lose the batch
  a reader is holding, so the operator fixes the store before restarting.
* ``telemetry_gateway_refusals`` — an **information** row, always ``ok``: "N
  refusals since gateway start". The enqueue-failed and receiver-refused series
  are monotonic counters, so one transient refusal would otherwise hold the
  check at ``warning`` until the gateway restarts; whether refusals are still
  happening is the host tier's question.
* ``telemetry_gateway_audit_files`` — where ``var/audit`` is visible: ``warning``
  naming every ``*.jsonl`` (dotfiles included) that is not a regular file. A FIFO
  blocks its receiver's open; a symlink to a device stalls its feed.
* ``telemetry_gateway_unfed`` — where ``var/audit`` and the rendered Collector
  file are both visible: the audit directories no ledger receiver reads, listed
  as "not fed until the next build". Information, ``ok``.

Host tier — stateful, on the host only
--------------------------------------
Where the repo root is writable and the container runtime answers, the category
keeps the previous sample in ``var/health/telemetry_gateway.json``
(``{counters, ts, sent_since}``, written atomically) and adds
``telemetry_gateway_trend``, ``warning`` when, against a sample at least 30 s
old:

* an exporter's send failures rose while its sent count did not;
* either refusal counter rose;

or, whatever the sample's age, an exporter's queue is non-empty while its sent
count has not moved for 300 s or more, or the gateway's log shows a storage
file renamed to ``.backup`` (the state store was found corrupt and recreated).

A counter below its previous value means the gateway restarted: the baseline is
re-taken. No previous sample: ``ok``, "baseline taken". A container never takes
this tier and never touches ``var/health``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import yaml

from osprey.deployment import telemetry_config as tc
from osprey.deployment.telemetry_gateway import (
    COLLECTOR_CONFIG_FILE,
    ledger_include_identity,
    non_regular_ledgers,
)
from osprey.health.core.telemetry_store import on_compose_network
from osprey.health.derive import _in_container
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "telemetry_gateway"

#: The host tier's previous sample, relative to the repo root.
BASELINE_FILE = Path("var") / "health" / "telemetry_gateway.json"
#: The audit ledger tree and the rendered Collector file, relative to the repo root.
AUDIT_DIR = Path("var") / "audit"
COLLECTOR_FILE = Path("build") / "services" / tc.GATEWAY_SERVICE / COLLECTOR_CONFIG_FILE

#: A comparison needs a previous sample at least this old.
MIN_SAMPLE_AGE_S = 30.0
#: A non-empty queue whose sent count has not moved for this long is stalled.
STALL_S = 300.0
#: Queue fill above which the stateless tier degrades.
QUEUE_DEGRADE_RATIO = 0.8

_METRICS_TIMEOUT_S = 5.0
_LOG_TIMEOUT_S = 10.0
_FALLBACK_QUEUE_CAPACITY = 1024 * 1024 * 1024

_SENT = "otelcol_exporter_sent_"
_SEND_FAILED = "otelcol_exporter_send_failed_"
_ENQUEUE_FAILED = "otelcol_exporter_enqueue_failed_"
_REFUSED = "otelcol_receiver_refused_"
_QUEUE_SIZE = "otelcol_exporter_queue_size"
_QUEUE_CAPACITY = "otelcol_exporter_queue_capacity"
_COUNTER_PREFIXES = (_SENT, _SEND_FAILED, _ENQUEUE_FAILED, _REFUSED)

_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)(?:\s+\S+)?$"
)
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_BACKUP_RE = re.compile(r"\.backup\b")

#: Reads the gateway's log since a moment (``None``: the runtime did not answer).
LogReader = Callable[[datetime], Awaitable[str | None]]


@dataclass
class Sample:
    """One reading of the gateway's counters page."""

    counters: dict[str, float] = field(default_factory=dict)
    #: Bytes held per exporter, summed over its per-signal queues.
    queue_size: dict[str, float] = field(default_factory=dict)
    queue_capacity: dict[str, float] = field(default_factory=dict)
    #: Bytes held per ``(exporter, data_type)`` queue; each fills on its own.
    queue_series: dict[tuple[str, str], float] = field(default_factory=dict)

    def per_exporter(self, prefix: str) -> dict[str, float]:
        """Sum the counters of one family by their ``exporter`` label."""
        sums: dict[str, float] = {}
        for key, value in self.counters.items():
            name, labels = _split_key(key)
            if name.startswith(prefix):
                exporter = labels.get("exporter", "")
                sums[exporter] = sums.get(exporter, 0.0) + value
        return sums

    def total(self, *prefixes: str) -> float:
        """Sum every counter of the given families."""
        return sum(
            value for key, value in self.counters.items() if _split_key(key)[0].startswith(prefixes)
        )


def parse_metrics(text: str) -> Sample:
    """Read the counters this category uses from a Prometheus text page.

    Counter names are taken with and without the ``_total`` suffix the
    Prometheus exporter adds, so the reading does not depend on the
    Collector's naming mode. Lines that are not samples are ignored.
    """
    sample = Sample()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        if math.isnan(value):
            continue
        name = match["name"].removesuffix("_total")
        labels = dict(_LABEL_RE.findall(match["labels"] or ""))
        if name == _QUEUE_SIZE:
            exporter = labels.get("exporter", "")
            series = (exporter, labels.get("data_type", ""))
            sample.queue_series[series] = sample.queue_series.get(series, 0.0) + value
            sample.queue_size[exporter] = sample.queue_size.get(exporter, 0.0) + value
        elif name == _QUEUE_CAPACITY:
            sample.queue_capacity[labels.get("exporter", "")] = value
        elif name.startswith(_COUNTER_PREFIXES):
            key = _series_key(name, labels)
            sample.counters[key] = sample.counters.get(key, 0.0) + value
    return sample


def _series_key(name: str, labels: Mapping[str, str]) -> str:
    inner = ",".join(f"{k}={labels[k]}" for k in sorted(labels) if k != "service_instance_id")
    return f"{name}{{{inner}}}"


def _split_key(key: str) -> tuple[str, dict[str, str]]:
    name, _, rest = key.partition("{")
    labels: dict[str, str] = {}
    for part in rest.rstrip("}").split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            labels[k] = v
    return name, labels


# --------------------------------------------------------------------------- #
# Stateless tier
# --------------------------------------------------------------------------- #


def stateless_rows(sample: Sample) -> list[CheckResult]:
    """The queue and refusal rows a single counters reading supports."""
    return [_queue_row(sample), _refusals_row(sample)]


def _queue_row(sample: Sample) -> CheckResult:
    full: list[str] = []
    worst = 0.0
    for (exporter, data_type), size in sorted(sample.queue_series.items()):
        capacity = sample.queue_capacity.get(exporter) or _FALLBACK_QUEUE_CAPACITY
        ratio = size / capacity
        worst = max(worst, ratio)
        if size > QUEUE_DEGRADE_RATIO * capacity:
            label = f"{exporter or '?'} {data_type}" if data_type else exporter or "?"
            full.append(f"{label} {ratio:.0%}")
    if full:
        return CheckResult(
            "telemetry_gateway_queue",
            CATEGORY,
            Status.WARNING,
            f"Export queue over {QUEUE_DEGRADE_RATIO:.0%} full: {', '.join(full)}",
            details=(
                "The store is not taking writes fast enough. Fix the store before "
                "restarting the gateway: a stop with a full ledger queue can lose a batch."
            ),
        )
    return CheckResult(
        "telemetry_gateway_queue",
        CATEGORY,
        Status.OK,
        f"Export queues at most {worst:.0%} full",
        value=f"{worst:.0%}",
    )


def _refusals_row(sample: Sample) -> CheckResult:
    count = int(sample.total(_ENQUEUE_FAILED, _REFUSED))
    return CheckResult(
        "telemetry_gateway_refusals",
        CATEGORY,
        Status.OK,
        f"{count} refusal{'s' if count != 1 else ''} since gateway start",
        value=str(count),
    )


def fed_identities(collector_config: Mapping[str, Any]) -> set[str]:
    """The identity directories the rendered Collector file has a ledger receiver for."""
    receivers = collector_config.get("receivers")
    if not isinstance(receivers, Mapping):
        return set()
    fed: set[str] = set()
    for receiver_id, body in receivers.items():
        if not str(receiver_id).startswith("file_log/") or not isinstance(body, Mapping):
            continue
        for pattern in body.get("include") or []:
            identity = ledger_include_identity(str(pattern))
            if identity is not None:
                fed.add(identity)
    return fed


def unfed_directories(audit_root: Path, collector_config: Mapping[str, Any]) -> list[str]:
    """The identity directories under *audit_root* no ledger receiver reads."""
    fed = fed_identities(collector_config)
    names: list[str] = []
    with contextlib.suppress(OSError), os.scandir(audit_root) as identities:
        for identity in identities:
            if identity.is_dir(follow_symlinks=False) and identity.name not in fed:
                names.append(identity.name)
    return sorted(names)


def _audit_rows(root: Path) -> list[CheckResult]:
    audit_root = root / AUDIT_DIR
    if not audit_root.is_dir():
        return []
    rows: list[CheckResult] = []
    bad = non_regular_ledgers(audit_root)
    if bad:
        rows.append(
            CheckResult(
                "telemetry_gateway_audit_files",
                CATEGORY,
                Status.WARNING,
                f"{len(bad)} ledger path{'s are' if len(bad) != 1 else ' is'} not a regular file",
                details="\n".join(str(path) for path in bad)
                + "\n`osprey up` moves each aside as `<name>.refused-<ts>`.",
            )
        )
    else:
        rows.append(
            CheckResult(
                "telemetry_gateway_audit_files",
                CATEGORY,
                Status.OK,
                "Every ledger file is a regular file",
            )
        )
    collector = _load_collector_config(root / COLLECTOR_FILE)
    if collector is not None:
        unfed = unfed_directories(audit_root, collector)
        if unfed:
            rows.append(
                CheckResult(
                    "telemetry_gateway_unfed",
                    CATEGORY,
                    Status.OK,
                    f"Not fed until the next build: {', '.join(unfed)}",
                    value=str(len(unfed)),
                    details="`osprey build` then `osprey up` gives each its own receiver.",
                )
            )
        else:
            rows.append(
                CheckResult(
                    "telemetry_gateway_unfed",
                    CATEGORY,
                    Status.OK,
                    "Every audit directory is fed",
                    value="0",
                )
            )
    return rows


def _load_collector_config(path: Path) -> Mapping[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, Mapping) else None


# --------------------------------------------------------------------------- #
# Host tier
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Baseline:
    """The host tier's previous sample."""

    counters: dict[str, float]
    ts: datetime
    sent_since: dict[str, datetime]

    def to_json(self) -> dict[str, Any]:
        return {
            "counters": self.counters,
            "ts": self.ts.isoformat(),
            "sent_since": {k: v.isoformat() for k, v in self.sent_since.items()},
        }

    @classmethod
    def from_json(cls, data: Any) -> Baseline | None:
        if not isinstance(data, Mapping):
            return None
        try:
            counters = {str(k): float(v) for k, v in dict(data["counters"]).items()}
            ts = _aware(datetime.fromisoformat(str(data["ts"])))
            since = {
                str(k): _aware(datetime.fromisoformat(str(v)))
                for k, v in dict(data.get("sent_since") or {}).items()
            }
        except (KeyError, TypeError, ValueError):
            return None
        return cls(counters, ts, since)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def read_baseline(path: Path) -> Baseline | None:
    """The stored sample, or ``None`` when there is none or it is unreadable."""
    try:
        return Baseline.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def write_baseline(path: Path, baseline: Baseline) -> None:
    """Write *baseline* atomically: a temporary file in the same directory, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(baseline.to_json(), handle, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def evaluate_trend(
    sample: Sample,
    previous: Baseline | None,
    now: datetime,
    *,
    backup_in_log: bool = False,
) -> tuple[CheckResult, Baseline | None]:
    """Judge *sample* against *previous*; return the row and the baseline to store.

    The returned baseline is ``None`` when the stored one must be kept: a
    sample younger than :data:`MIN_SAMPLE_AGE_S` is not a comparison point.
    """
    sent = sample.per_exporter(_SENT)
    fresh = Baseline(dict(sample.counters), now, dict.fromkeys(sent, now))

    if previous is None:
        return _trend(Status.OK, "Baseline taken"), fresh
    if any(sample.counters.get(key, value) < value for key, value in previous.counters.items()):
        return _trend(Status.OK, "Gateway restarted; baseline re-taken"), fresh

    prev_sample = Sample(counters=dict(previous.counters))
    prev_sent = prev_sample.per_exporter(_SENT)
    reasons: list[str] = []

    for exporter, size in sorted(sample.queue_size.items()):
        if size <= 0 or sent.get(exporter) != prev_sent.get(exporter):
            continue
        since = previous.sent_since.get(exporter, previous.ts)
        stalled = (now - since).total_seconds()
        if stalled >= STALL_S:
            reasons.append(
                f"{exporter or '?'} queue holds data and has sent nothing for {int(stalled)} s"
            )
    if backup_in_log:
        reasons.append("the gateway log shows its state file renamed to .backup")

    age = (now - previous.ts).total_seconds()
    keep = age < MIN_SAMPLE_AGE_S
    if not keep:
        failed = sample.per_exporter(_SEND_FAILED)
        prev_failed = prev_sample.per_exporter(_SEND_FAILED)
        for exporter in sorted(failed):
            rose = failed[exporter] > prev_failed.get(exporter, 0.0)
            if rose and sent.get(exporter, 0.0) <= prev_sent.get(exporter, 0.0):
                reasons.append(f"{exporter or '?'} send failures rose with nothing sent")
        refused_now = sample.total(_ENQUEUE_FAILED, _REFUSED)
        refused_then = prev_sample.total(_ENQUEUE_FAILED, _REFUSED)
        if refused_now > refused_then:
            reasons.append(f"{int(refused_now - refused_then)} new refusal(s) in {int(age)} s")

    if keep:
        stored = None
    else:
        since_map = {
            exporter: (
                previous.sent_since.get(exporter, previous.ts)
                if value == prev_sent.get(exporter)
                else now
            )
            for exporter, value in sent.items()
        }
        stored = Baseline(dict(sample.counters), now, since_map)

    if reasons:
        return (
            _trend(
                Status.WARNING,
                "; ".join(reasons),
                details="`osprey logs telemetry_gateway` and `osprey health --category telemetry_store`.",
            ),
            stored,
        )
    if keep:
        return _trend(
            Status.OK, f"Previous sample {int(age)} s old; compared at {int(MIN_SAMPLE_AGE_S)} s"
        ), None
    return _trend(Status.OK, f"Counters steady over {int(age)} s"), stored


def _trend(status: Status, message: str, *, details: str = "") -> CheckResult:
    return CheckResult("telemetry_gateway_trend", CATEGORY, status, message, details=details)


def _default_log_reader(config: Mapping[str, Any] | None) -> LogReader:
    """Read the gateway container's log through the deployment's container runtime."""

    async def _read(since: datetime) -> str | None:
        from osprey.deployment.compose_generator import resolve_project_name
        from osprey.deployment.runtime_helper import get_runtime_command

        try:
            runtime = (await asyncio.to_thread(get_runtime_command, config))[0]
        except (RuntimeError, FileNotFoundError, IndexError):
            return None
        container = f"{resolve_project_name(dict(config or {}))}-telemetry-gateway"
        try:
            proc = await asyncio.create_subprocess_exec(
                runtime,
                "logs",
                "--since",
                since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                container,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_LOG_TIMEOUT_S)
        except (OSError, TimeoutError):
            return None
        if proc.returncode != 0:
            return None
        return out.decode("utf-8", errors="replace")

    return _read


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def metrics_url(config: Mapping[str, Any] | None, environ: Mapping[str, str]) -> str:
    """The gateway's counters page from where the check runs."""
    if on_compose_network(environ):
        return f"http://{tc.GATEWAY_SERVICE}:{tc.GATEWAY_METRICS_CONTAINER_PORT}/metrics"
    return f"http://127.0.0.1:{tc.metrics_port(config)}/metrics"


def telemetry_gateway(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    cwd: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
    now: Callable[[], datetime] | None = None,
    in_container: Callable[[], bool] | None = None,
    log_reader: LogReader | None = None,
) -> CategoryCallable:
    """Build the ``telemetry_gateway`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Health runtime. Unused.
        cwd: The repo root ``var/`` and ``build/`` hang off; defaults to
            :func:`Path.cwd`, resolved when the callable runs.
        transport: Optional httpx transport for tests.
        environ: The process environment the network shape is read from;
            defaults to :data:`os.environ`.
        now: Clock, for tests; defaults to the current UTC time.
        in_container: Whether this process runs in a container; defaults to
            the health package's own detection.
        log_reader: Reads the gateway's log since a moment, ``None`` when the
            container runtime does not answer (which also keeps the host tier
            off); defaults to the deployment's runtime.

    Returns:
        A no-argument async callable returning the category's check results.
    """

    async def _run() -> list[CheckResult]:
        if not tc.gateway_deployed(config):
            return []
        root = cwd if cwd is not None else Path.cwd()
        env = os.environ if environ is None else environ
        current = (now or (lambda: datetime.now(UTC)))()
        contained = (in_container or _in_container)()

        rows: list[CheckResult] = []
        sample = await _fetch(metrics_url(config, env), transport, rows)
        if sample is not None:
            rows.extend(stateless_rows(sample))
        rows.extend(_audit_rows(root))

        if sample is not None and not contained and os.access(root, os.W_OK):
            trend = await _host_tier(
                root, sample, current, log_reader or _default_log_reader(config)
            )
            if trend is not None:
                rows.append(trend)
        return rows

    return _run


async def _fetch(
    url: str, transport: httpx.AsyncBaseTransport | None, rows: list[CheckResult]
) -> Sample | None:
    try:
        async with httpx.AsyncClient(timeout=_METRICS_TIMEOUT_S, transport=transport) as client:
            resp = await client.get(url)
    except (httpx.HTTPError, OSError) as exc:
        rows.append(
            CheckResult(
                "telemetry_gateway_metrics",
                CATEGORY,
                Status.WARNING,
                f"Gateway counters unreachable at {url}: {exc}",
                details="Start it with `osprey up`; `osprey logs telemetry_gateway` shows why it stopped.",
            )
        )
        return None
    if resp.status_code != 200:
        rows.append(
            CheckResult(
                "telemetry_gateway_metrics",
                CATEGORY,
                Status.WARNING,
                f"Gateway counters returned HTTP {resp.status_code} ({url})",
            )
        )
        return None
    rows.append(
        CheckResult("telemetry_gateway_metrics", CATEGORY, Status.OK, f"Gateway answers ({url})")
    )
    return parse_metrics(resp.text)


async def _host_tier(
    root: Path, sample: Sample, current: datetime, log_reader: LogReader
) -> CheckResult | None:
    path = root / BASELINE_FILE
    previous = read_baseline(path)
    since = previous.ts if previous is not None else current
    log = await log_reader(since)
    if log is None:
        return None
    backup = previous is not None and bool(_BACKUP_RE.search(log))
    row, stored = evaluate_trend(sample, previous, current, backup_in_log=backup)
    if stored is not None:
        try:
            write_baseline(path, stored)
        except OSError as exc:
            return _trend(Status.WARNING, f"Cannot keep the gateway sample at {path}: {exc}")
    return row
