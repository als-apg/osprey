#!/usr/bin/env python
"""Curate CBORG's full model list into a canonical, deduplicated, chat-capable set
for the e2e model matrix (issue #259).

Fetches https://api.cborg.lbl.gov/v1/models via `ssh macstudio` (which is on-network;
the endpoint 403s off-VPN), then applies deterministic, AUDITABLE rules:

DROP (non-chat): embeddings, OCR, vision, rerankers, safety/privacy filters.
CLASSIFY: a family backed by an `lbl/*` id => self_hosted_open; else commercial_proxy.
COLLAPSE (same underlying model, dedup grouping only — we never fabricate an id):
  - routing prefixes:  amazon/ google/ anthropic/ meta/ xai/ lbl/
  - inference-effort / priority knobs: -low -medium -high -max -xhigh -priority -high-priority
  Distinct tiers are KEPT (-fast -mini -nano -lite -thinking -chat -codex -pro -reasoning ...)
  and distinct version numbers are KEPT (claude-opus-4-5 != claude-opus-4-8).
PROTOCOL: claude-* => anthropic (CBORG Anthropic route, no proxy); else openai (via proxy).

For each dedup group we pick the most-canonical id that ACTUALLY EXISTS in the list
(prefer no-prefix > lbl/ > other; prefer no-effort-suffix; then shorter), so every
emitted api_id is one CBORG already advertises.

Emits scripts/benchmark/canonical_models.json and prints a summary of what collapsed/dropped
(no silent caps — every drop/collapse is reported).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "canonical_models.json"

# Substrings that mark a model as NOT a text chat/tool model -> dropped.
NON_CHAT = (
    "embed",
    "nomic",
    "titan",
    "cohere-embed",
    "text-embedding",
    "gemini-embedding",
    "ocr",
    "vision",
    "-vl",
    "safeguard",
    "privacy-filter",
    "rerank",
)

ROUTING_PREFIXES = ("amazon/", "google/", "anthropic/", "meta/", "xai/", "lbl/")
# Pure inference knobs (same weights, different decode) -> collapsed for grouping.
EFFORT_SUFFIXES = ("-high-priority", "-priority", "-xhigh", "-max", "-high", "-medium", "-low")


def fetch_ids() -> list[str]:
    cmd = (
        'curl -s https://api.cborg.lbl.gov/v1/models -H "Authorization: Bearer $(cat ~/.cborg_key)"'
    )
    out = subprocess.run(["ssh", "macstudio", cmd], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        sys.exit(f"fetch failed: {out.stderr}")
    data = json.loads(out.stdout)
    return sorted({m["id"] for m in data["data"]})


def strip_prefix(mid: str) -> str:
    for p in ROUTING_PREFIXES:
        if mid.startswith(p):
            return mid[len(p) :]
    return mid


def strip_effort(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for s in EFFORT_SUFFIXES:
            if name.endswith(s):
                name = name[: -len(s)]
                changed = True
    return name


def family_key(mid: str) -> str:
    """Normalized key: routing prefix + effort/priority knob removed, lowercased."""
    return strip_effort(strip_prefix(mid)).lower()


def is_self_hosted(group_ids: list[str]) -> bool:
    return any(g.startswith("lbl/") for g in group_ids)


def canonical_id(group_ids: list[str]) -> str:
    """Pick the most canonical EXISTING id for the API call."""

    def rank(mid: str):
        has_slash = "/" in mid
        # prefer no-slash; among slashed, prefer lbl/
        slash_rank = 0 if not has_slash else (1 if mid.startswith("lbl/") else 2)
        has_effort = strip_effort(strip_prefix(mid)) != strip_prefix(mid)
        return (slash_rank, has_effort, len(mid), mid)

    return min(group_ids, key=rank)


def main() -> int:
    ids = fetch_ids()
    dropped: dict[str, list[str]] = {}
    kept: list[str] = []
    for mid in ids:
        low = mid.lower()
        if any(tok in low for tok in NON_CHAT):
            dropped.setdefault("non_chat", []).append(mid)
        else:
            kept.append(mid)

    groups: dict[str, list[str]] = {}
    for mid in kept:
        groups.setdefault(family_key(mid), []).append(mid)

    models = []
    for key, members in sorted(groups.items()):
        cid = canonical_id(members)
        self_hosted = is_self_hosted(members)
        protocol = "anthropic" if key.startswith("claude-") or key in ("claude",) else "openai"
        models.append(
            {
                "api_id": cid,
                "family_key": key,
                "category": "self_hosted_open" if self_hosted else "commercial_proxy",
                "protocol": protocol,
                "collapsed_from": sorted(members),
            }
        )

    models.sort(key=lambda m: (m["category"] != "self_hosted_open", m["family_key"]))
    n_open = sum(m["category"] == "self_hosted_open" for m in models)
    n_comm = len(models) - n_open

    payload = {
        "source": "https://api.cborg.lbl.gov/v1/models",
        "total_ids": len(ids),
        "dropped_non_chat": sorted(dropped.get("non_chat", [])),
        "n_canonical": len(models),
        "n_self_hosted_open": n_open,
        "n_commercial_proxy": n_comm,
        "models": models,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"raw ids:            {len(ids)}")
    print(f"dropped (non-chat): {len(payload['dropped_non_chat'])}")
    print(f"canonical models:   {len(models)}  (open={n_open}, commercial={n_comm})")
    print(f"\nwrote {OUT}\n")
    print("=== SELF-HOSTED OPEN ===")
    for m in models:
        if m["category"] == "self_hosted_open":
            extra = f"   <- {len(m['collapsed_from'])} ids" if len(m["collapsed_from"]) > 1 else ""
            print(f"  [{m['protocol']:9}] {m['api_id']}{extra}")
    print("=== COMMERCIAL PROXY ===")
    for m in models:
        if m["category"] == "commercial_proxy":
            extra = f"   <- {len(m['collapsed_from'])} ids" if len(m["collapsed_from"]) > 1 else ""
            print(f"  [{m['protocol']:9}] {m['api_id']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
