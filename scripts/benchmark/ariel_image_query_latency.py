#!/usr/bin/env python3
"""Query-embed latency of ARIEL picture search while pictures are embedded in bulk.

Picture search embeds the operator's query on the same llama-server that the
``image_embedding`` enhancement module feeds pictures to. This script measures
what a query waits for while that module is busy, on the documented server
command, through the adapter OSPREY itself calls
(:class:`osprey.models.providers.llama_cpp.LlamaCppProviderAdapter`):

1. ``/v1/models`` — the advertised id and capabilities.
2. Idle queries — one text embedding at a time, nothing else running.
3. Solo pictures — one picture embedding at a time; this is seconds per picture.
4. Queries under bulk — a background thread embeds pictures back to back, one
   in flight, exactly as the enhancement module does, while the foreground
   embeds queries one at a time. The p95 of these is the gated value.

The pictures are synthetic plots at the rendition size OSPREY stores (long edge
1024 px), drawn deterministically, so two runs send the same bytes.

The gate is ``p95 <= --gate-seconds`` (2 s). It only counts on a native x86_64
host: on any other machine the result is printed as informative and never
passes or fails the gate.

Usage::

    python scripts/benchmark/ariel_image_query_latency.py \\
        --base-url http://127.0.0.1:8080 --json latency.json

Exit status: 0 when the run completed (pass or fail); with ``--enforce``, 1 when
the gate failed or the host does not count; 2 when the server could not be
measured at all.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from typing import Any

GATE_SECONDS = 2.0
TIMEOUT_FLOOR_SECONDS = 120

_SUBJECTS = (
    "orbit kick near BPM 7",
    "horizontal orbit distortion after fill",
    "tunnel air temperature drift",
    "RF cavity trip strip chart",
    "vacuum pressure spike in sector 4",
    "beam current decay during top-off",
    "injection efficiency drop",
    "bunch-by-bunch feedback saturation",
    "quadrupole power supply ripple",
    "beam size growth on the pinhole camera",
    "insertion device gap scan",
    "storage ring lifetime after the vent",
)
_PHRASINGS = (
    "{}",
    "plot showing {}",
    "screenshot of {}",
    "when did we last see {}",
    "{} last week",
    "trend of {} overnight",
    "who logged {}",
    "{} with a picture attached",
    "archiver view of {}",
)


def queries(count: int) -> list[str]:
    """``count`` distinct operator-style queries, in a fixed order."""
    texts = [phrasing.format(subject) for phrasing in _PHRASINGS for subject in _SUBJECTS]
    return [texts[i % len(texts)] + ("" if i < len(texts) else f" ({i})") for i in range(count)]


def synthetic_picture(index: int, width: int, height: int) -> bytes:
    """One deterministic plot-like PNG of ``width`` x ``height`` pixels."""
    from PIL import Image, ImageDraw

    rng = random.Random(1000 + index)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 40, width - 30, height - 60
    draw.rectangle((left, top, right, bottom), outline="black", width=2)
    for step in range(1, 8):
        x = left + (right - left) * step // 8
        y = top + (bottom - top) * step // 8
        draw.line((x, top, x, bottom), fill=(220, 220, 220))
        draw.line((left, y, right, y), fill=(220, 220, 220))
        draw.text((x - 8, bottom + 8), str(step * 12), fill="black")
        draw.text((left - 40, y - 6), f"{rng.uniform(-2, 2):.1f}", fill="black")
    for colour in ((31, 119, 180), (214, 39, 40), (44, 160, 44)):
        value = rng.uniform(0.3, 0.7)
        points = []
        for x in range(left, right, 4):
            value = min(0.95, max(0.05, value + rng.uniform(-0.03, 0.03)))
            points.append((x, top + (bottom - top) * value))
        draw.line(points, fill=colour, width=2)
    draw.text((left, 12), f"SR:C{index + 1:02d} BPM orbit, fill {4100 + index}", fill="black")
    draw.text((left, height - 28), "time [h]", fill="black")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value with ``fraction`` of the sample at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summary(values: list[float]) -> dict[str, float | int]:
    """Count, mean, median, p95 and max of ``values``, in seconds."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean_s": round(statistics.fmean(values), 4),
        "p50_s": round(percentile(values, 0.50), 4),
        "p95_s": round(percentile(values, 0.95), 4),
        "max_s": round(max(values), 4),
    }


def host_facts() -> dict[str, Any]:
    """The machine this script runs on. Run it on the server's host for the gate."""
    model_name = None
    mem_total_kb = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    mem_total_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    machine = platform.machine()
    emulated = model_name is None or any(
        marker in model_name for marker in ("VirtualApple", "QEMU", "Apple")
    )
    native = machine == "x86_64" and platform.system() == "Linux" and not emulated
    return {
        "uname_m": machine,
        "system": platform.system(),
        "cpu_model_name": model_name,
        "logical_cpus": os.cpu_count(),
        "mem_total_gib": round(mem_total_kb / 1024 / 1024, 1) if mem_total_kb else None,
        "native_x86_64": native,
        "label": "amd64 CPU, gated" if native else f"{machine}, informative",
    }


def main(argv: list[str] | None = None) -> int:
    from osprey.models.providers.llama_cpp import (
        LLAMA_CPP_DEFAULT_MODEL,
        LlamaCppProviderAdapter,
    )
    from osprey.services.ariel_search.search.image_lane import IMAGE_QUERY_TIMEOUT_S

    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server root")
    parser.add_argument("--model", default=LLAMA_CPP_DEFAULT_MODEL, help="the server's --alias")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=100, help="queries timed under bulk")
    parser.add_argument("--idle-queries", type=int, default=20)
    parser.add_argument("--solo-pictures", type=int, default=6)
    parser.add_argument("--pictures", type=int, default=6, help="distinct pictures in the bulk")
    parser.add_argument("--picture-size", default="1024x768", help="WIDTHxHEIGHT in pixels")
    parser.add_argument("--pause", type=float, default=0.25, help="seconds between queries")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per call")
    parser.add_argument("--gate-seconds", type=float, default=GATE_SECONDS)
    parser.add_argument("--json", dest="json_path", help="also write the result here")
    parser.add_argument("--enforce", action="store_true", help="exit 1 unless the gate passed")
    args = parser.parse_args(argv)

    width, height = (int(part) for part in args.picture_size.lower().split("x"))
    adapter = LlamaCppProviderAdapter()

    def embed(item: Any) -> float:
        started = time.perf_counter()
        adapter.execute_image_embedding(
            [item],
            args.model,
            base_url=args.base_url,
            dimensions=args.dimensions,
            timeout=args.timeout,
        )
        return time.perf_counter() - started

    result: dict[str, Any] = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": host_facts(),
        "base_url": args.base_url,
        "model": args.model,
        "dimensions": args.dimensions,
        "picture_size": f"{width}x{height}",
        "gate_seconds": args.gate_seconds,
    }

    try:
        import requests

        models = requests.get(args.base_url.rstrip("/") + "/v1/models", timeout=10).json()
        entry = (models.get("data") or [{}])[0]
        result["v1_models"] = {
            "id": entry.get("id"),
            "capabilities": (models.get("models") or [{}])[0].get("capabilities"),
        }
        pictures = [
            (synthetic_picture(i, width, height), "image/png") for i in range(args.pictures)
        ]
        result["picture_bytes"] = [len(data) for data, _ in pictures]
        embed("warm-up query")
        embed(pictures[0])

        texts = queries(args.idle_queries + args.queries)
        idle = [embed(text) for text in texts[: args.idle_queries]]
        solo = [embed(pictures[i % len(pictures)]) for i in range(args.solo_pictures)]

        stop = threading.Event()
        bulk: list[float] = []
        bulk_errors: list[str] = []

        def bulk_loop() -> None:
            index = 0
            while not stop.is_set():
                try:
                    bulk.append(embed(pictures[index % len(pictures)]))
                except Exception as exc:  # the foreground reports it
                    bulk_errors.append(f"{type(exc).__name__}: {exc}")
                    return
                index += 1

        worker = threading.Thread(target=bulk_loop, daemon=True)
        worker.start()
        time.sleep(1.0)  # the first bulk picture is in flight before the first query
        under_bulk: list[float] = []
        for text in texts[args.idle_queries :]:
            under_bulk.append(embed(text))
            time.sleep(args.pause)
        stop.set()
        worker.join(timeout=args.timeout)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2))
        return 2

    seconds_per_picture = statistics.fmean(solo)
    p95 = percentile(under_bulk, 0.95)
    passed = p95 <= args.gate_seconds
    counts = result["host"]["native_x86_64"]
    result.update(
        {
            "idle_query": summary(idle),
            "solo_picture": summary(solo),
            "query_under_bulk": summary(under_bulk),
            "bulk_picture_during_queries": summary(bulk),
            "bulk_errors": bulk_errors,
            "queries_over_gate": sum(1 for value in under_bulk if value > args.gate_seconds),
            "queries_over_lane_timeout": sum(
                1 for value in under_bulk if value > IMAGE_QUERY_TIMEOUT_S
            ),
            "lane_timeout_s": IMAGE_QUERY_TIMEOUT_S,
            "gate": ("pass" if passed else "fail") if counts else "informative only",
            "fallback_lane_timeout_s": round(max(5.0, 2 * p95), 2),
            "image_embedding_timeout_seconds": max(
                TIMEOUT_FLOOR_SECONDS, math.ceil(10 * seconds_per_picture / 10) * 10
            ),
        }
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if args.enforce and not (counts and passed and not bulk_errors):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
