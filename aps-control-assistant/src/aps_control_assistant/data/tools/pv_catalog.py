"""
Utility for loading PV aliases from the PVs.mon SDDS file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


QUERY_STOPWORDS = {
    "list",
    "lists",
    "pv",
    "pvs",
    "name",
    "names",
    "of",
    "the",
    "for",
    "and",
    "or",
    "what",
    "show",
    "find",
    "get",
    "tell",
    "me",
    "please",
    "value",
    "values",
    "history",
    "plot",
    "monitor",
    "data",
    "info",
    "information",
    "about",
    "is",
    "are",
    "to",
    "in",
    "on",
    "at",
    "by",
    "be",
    "do",
    "go",
    "we",
    "us",
    "it",
    "its",
    "sr",
}

TIME_KEYWORDS = {
    "last",
    "past",
    "recent",
    "previous",
    "month",
    "months",
    "week",
    "weeks",
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "today",
    "yesterday",
    "tomorrow",
    "tonight",
    "ago",
}

# ---------------------------------------------------------------------------
# SR corrector/sextupole/etc. generation from srcorr.json
# ---------------------------------------------------------------------------

_SRCORR_CACHE = None

# Map human-friendly type keywords to section names in srcorr.json
SRCORR_TYPE_MAP = {
    "fast": ("SR Fast Correctors", "Sector ${sector} fast correctors"),
    "corrector": ("SR Correctors", "Sector ${sector} correctors"),
    "skew": ("SR Skew Quadrupoles", "Sector ${sector} skew quadrupoles"),
    "quadrupole": ("SR Quadrupoles", "Sector ${sector} quadrupoles"),
    "sextupole": ("SR Sextupoles", "Sector ${sector} sextupoles"),
    "dipole": ("SR Dipoles", "Sector ${sector} dipoles"),
    "dipole_trim": ("SR Dipole Trim", "Sector ${sector} dipole trim"),
}

# Simple keyword hints for type detection
SRCORR_TYPE_KEYWORDS = [
    ("fast", ("fast", "fh", "fv")),
    ("dipole_trim", ("trim", "trimmed", "trimmer")),
    ("dipole", ("dipole", "bending")),
    ("skew", ("skew", "sq")),
    ("sextupole", ("sext", "sextupole", "sxt")),
    ("quadrupole", ("quad", "quadrupole", "qf", "qd", "aq", "bq")),
    ("corrector", ("corrector", "corr", "h corrector", "v corrector", "hcor", "vcor")),
]

# Map suffix keyword to exact suffix label in srcorr.json
SRCORR_SUFFIX_KEYWORDS = {
    "setpoint": "setpoint",
    "set": "setpoint",
    "command": "setpoint",
    "readback": "setpoint readback",
    "sp_rb": "setpoint readback",
    "meas": "current",
    "current": "current",
    "dcct": "DCCT output",
    "voltage": "Output voltage",
    "heatsink": "Heatsink Temperature",
    "heat": "Heatsink Temperature",
    "temp": "Heatsink Temperature",
    "temperature": "Heatsink Temperature",
    "capacitor": "Capacitor Temperature",
    "caps": "Capacitor Temperature",
    "cap": "Capacitor Temperature",
    "damp": "Damping Resistor Temperature",
    "resistor": "Damping Resistor Temperature",
    "power": "PSID",
}

def _norm(text: str) -> str:
    return text.replace(" ", "").replace("_", "").lower()


def is_srcorr_query(query: str) -> bool:
    """Return True if query clearly targets SR magnets/correctors covered by srcorr.json."""
    q = query.lower()
    # Check for explicit SR mentions OR sector-based queries
    has_sr_context = ("sr" in q or "storage ring" in q or "sector" in q or
                      re.search(r'\bs\d{1,2}[abc]?:', q))  # Matches patterns like S01A:, S5:

    keywords = (
        "corrector",
        "fast",
        "quadrupole",
        "sextupole",
        "skew",
        "dipole",
        "trim",
        "fh",
        "fv",
        "h corrector",
        "v corrector",
        "qf",
        "qd",
    )
    has_device_keyword = any(k in q for k in keywords)

    # Return true if we have device keywords AND (SR context OR it's a specific device pattern)
    # This allows "sector 5 fast corrector" or "fast horizontal correctors" with sector context
    return has_device_keyword and (has_sr_context or
                                    re.search(r'sector\s+\d+', q) or
                                    ("fast" in q and ("corrector" in q or "fh" in q or "fv" in q)))


def _load_srcorr_json() -> Dict:
    global _SRCORR_CACHE
    if _SRCORR_CACHE is not None:
        return _SRCORR_CACHE
    path = Path(__file__).resolve().parent.parent / "channel_databases" / "srcorr.json"
    try:
        with path.open() as f:
            _SRCORR_CACHE = json.load(f)
    except Exception:
        _SRCORR_CACHE = {}
    return _SRCORR_CACHE


def _parse_sectors_from_query(query: str) -> List[int]:
    numbers = re.findall(r"(?:sector\s*)?(\d+)(?:\s*-\s*(\d+))?", query)
    sectors: List[int] = []
    for start_str, end_str in numbers:
        start = int(start_str)
        if end_str:
            end = int(end_str)
            step = 1 if start <= end else -1
            sectors.extend(range(start, end + step, step))
        else:
            sectors.append(start)
    if not sectors:
        sectors = list(range(1, 41))
    # Deduplicate preserving order
    dedup = []
    seen = set()
    for s in sectors:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def _detect_types(query: str) -> List[str]:
    q = query.lower()
    detected: List[str] = []
    for type_key, hints in SRCORR_TYPE_KEYWORDS:
        if any(h in q for h in hints):
            detected.append(type_key)
    # Prioritize specific families to avoid mixing in generic correctors
    if "fast" in detected:
        detected = ["fast"]
    elif "dipole_trim" in detected:
        detected = ["dipole_trim"]
    elif "dipole" in detected:
        detected = ["dipole"]
    if not detected:
        detected = ["corrector"]
    # Deduplicate while preserving order
    out = []
    seen = set()
    for t in detected:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _detect_suffixes(query: str, data_suffixes: Dict[str, str]) -> List[str]:
    q = query.lower()
    wanted: List[str] = []
    # Broad temperature requests: include all temp-like suffixes
    if "temp" in q or "temperature" in q:
        for name in data_suffixes:
            n = _norm(name)
            if "temp" in n or "temperature" in n:
                wanted.append(name)

    for key, label in SRCORR_SUFFIX_KEYWORDS.items():
        if key in q and label in data_suffixes:
            wanted.append(label)
    if not wanted:
        return list(data_suffixes.keys())
    # Deduplicate preserving order
    dedup = []
    seen = set()
    for w in wanted:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return dedup


def _device_matches(device: Dict, filters: List[str]) -> bool:
    if not filters:
        return True
    candidates = [_norm(device.get("channel", ""))]
    for tag in device.get("tags", []) or []:
        candidates.append(_norm(tag))
    for f in filters:
        nf = _norm(f)
        for cand in candidates:
            if nf == cand or nf in cand or cand in nf:
                return True
    return False


def generate_srcorr_pvs_from_query(query: str) -> List[str]:
    """
    Lightweight generator for SR corrector/quad/sextupole/dipole PVs based on srcorr.json.
    This avoids an LLM call when the query clearly maps to the structured srcorr database.
    """
    data = _load_srcorr_json()
    if not data:
        return []

    q_lower = query.lower()
    sectors = _parse_sectors_from_query(query)
    types = _detect_types(query)
    # Include suffixes if the query mentions:
    # 1. PV/channel/address/suffix concepts, OR
    # 2. Any specific suffix keyword (setpoint, current, temp, etc.)
    suffix_request_terms = ("pv", "pvs", "channel", "channels", "address", "addresses", "suffix")
    has_suffix_keyword = any(keyword in q_lower for keyword in SRCORR_SUFFIX_KEYWORDS.keys())
    include_suffixes = any(term in q_lower for term in suffix_request_terms) or has_suffix_keyword
    # Only include base devices if no specific suffix is requested
    # If user asks for "setpoint", don't include base addresses
    include_base = not has_suffix_keyword

    all_pvs: List[str] = []
    # Apply device/tag filters when the query includes device-like tokens or axis hints
    extra_filters: List[str] = []
    if "horizontal" in q_lower:
        extra_filters.extend(["h", "fh"])
    if "vertical" in q_lower:
        extra_filters.extend(["v", "fv"])
    filters = [w for w in re.findall(r"[A-Za-z]+\d+[:A-Za-z\d]*", query) if w] + extra_filters

    for t in types:
        section = SRCORR_TYPE_MAP.get(t)
        if not section:
            continue
        section_key, devices_key = section
        section_data = data.get(section_key, {})
        suffix_map = section_data.get("suffix", {})
        suffix_labels = _detect_suffixes(query, suffix_map) if include_suffixes else []
        devices = section_data.get(devices_key, [])

        for sector in sectors:
            sec_str = f"{sector:02d}"
            for device in devices:
                if not _device_matches(device, filters):
                    continue
                allowed_sectors = device.get("sectors")
                if allowed_sectors is not None and sector not in allowed_sectors:
                    continue
                base = device.get("address_template", "").replace("${sector}", sec_str)
                if not base:
                    continue
                if include_base:
                    all_pvs.append(base)
                for suffix_name in suffix_labels:
                    suffix_val = suffix_map.get(suffix_name)
                    if suffix_val:
                        all_pvs.append(base + suffix_val)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for pv in all_pvs:
        if pv not in seen:
            seen.add(pv)
            deduped.append(pv)
    return deduped

@dataclass(frozen=True)
class PVEntry:
    """Represents a Process Variable and its aliases."""

    pv_name: str
    label: str
    aliases: List[str]


class PVCatalog:
    """
    Parses the PVs.mon SDDS file and exposes alias lookups.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or self.default_path()
        self.entries: List[PVEntry] = []
        self.alias_map: Dict[str, PVEntry] = {}
        self._load()

    @staticmethod
    def default_path() -> str:
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(utils_dir)
        return os.path.join(project_root, "PVs.mon")

    def _load(self) -> None:
        if not os.path.exists(self.filepath):
            return

        ascii_mode = False
        try:
            with open(self.filepath, "rb") as sniff:
                header = sniff.read(4096)
                ascii_mode = b"mode=ascii" in header.lower()
        except OSError:
            return

        if ascii_mode:
            self._load_ascii()
            return

        parameter_types: List[str] = []
        column_defs: List[Tuple[str, str]] = []
        try:
            with open(self.filepath, "rb") as f:
                # Skip header lines, tracking parameter definitions
                while True:
                    line = f.readline()
                    if not line:
                        return

                    stripped = line.strip()
                    if not stripped:
                        continue

                    lower_line = stripped.lower()
                    if lower_line.startswith(b"&parameter"):
                        decoded = stripped.decode("utf-8", errors="ignore")
                        param_type = self._extract_parameter_type(decoded)
                        if param_type:
                            parameter_types.append(param_type)

                    elif lower_line.startswith(b"&column"):
                        decoded = stripped.decode("utf-8", errors="ignore")
                        column_def = self._extract_column_definition(decoded)
                        if column_def:
                            column_defs.append(column_def)

                    if lower_line.startswith(b"&data"):
                        break

                # After header, binary section begins
                row_count = self._read_int(f)
                if parameter_types:
                    self._skip_parameter_data(f, parameter_types)

                if not column_defs:
                    logging.warning("No column definitions found in PV catalog %s.", self.filepath)
                    return

                for index in range(row_count):
                    pv_name = None
                    description = None
                    try:
                        for col_name, col_type in column_defs:
                            value = self._read_column_value(f, col_type)
                            if value is None:
                                continue
                            if col_name == "controlname":
                                pv_name = value
                            elif col_name == "description":
                                description = value
                    except EOFError as exc:
                        logging.warning(
                            "PV catalog %s ended unexpectedly after %d/%d rows: %s",
                            self.filepath,
                            index,
                            row_count,
                            exc,
                        )
                        break

                    if not pv_name:
                        continue
                    entry = self._build_entry(pv_name, description or "")
                    self.entries.append(entry)
                    for alias in entry.aliases:
                        self.alias_map.setdefault(alias, entry)
        except (OSError, EOFError) as exc:
            logging.warning("Failed to load PV catalog %s: %s", self.filepath, exc)

    def _load_ascii(self) -> None:
        try:
            with open(self.filepath, "r", encoding="utf-8", errors="ignore") as fh:
                data_section = False
                column_names: List[str] = []
                control_idx: Optional[int] = None
                description_idx: Optional[int] = None
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    lower_line = line.lower()
                    if not data_section:
                        if lower_line.startswith("&column"):
                            column_def = self._extract_column_definition(line)
                            if column_def:
                                column_names.append(column_def[0])
                        if lower_line.startswith("&data"):
                            data_section = True
                            control_idx = self._column_index(column_names, "controlname")
                            description_idx = self._column_index(column_names, "description")
                        continue

                    if line.startswith(("!", "#", "/")):
                        continue
                    try:
                        parts = shlex.split(line)
                    except ValueError:
                        continue
                    if control_idx is None or description_idx is None:
                        continue
                    max_index = max(control_idx, description_idx)
                    if max_index >= len(parts):
                        continue
                    pv_name = parts[control_idx]
                    description = parts[description_idx] if description_idx < len(parts) else ""
                    entry = self._build_entry(pv_name, description)
                    self.entries.append(entry)
                    for alias in entry.aliases:
                        self.alias_map.setdefault(alias, entry)
        except OSError:
            return

    def _read_int(self, fh) -> int:
        data = fh.read(4)
        if len(data) != 4:
            raise EOFError("Unexpected end of PV catalog.")
        return struct.unpack(">I", data)[0]

    def _read_string(self, fh) -> str:
        length = self._read_int(fh)
        if length == 0:
            return ""
        data = fh.read(length)
        if len(data) != length:
            raise EOFError("Unexpected end of PV catalog string.")
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_parameter_type(definition: str) -> Optional[str]:
        match = re.search(r"type\s*=\s*([a-zA-Z]+)", definition)
        if match:
            return match.group(1).lower()
        return None

    @staticmethod
    def _extract_column_definition(definition: str) -> Optional[Tuple[str, str]]:
        name_match = re.search(r"name\s*=\s*([^,]+)", definition, re.IGNORECASE)
        type_match = re.search(r"type\s*=\s*([^,]+)", definition, re.IGNORECASE)
        if not name_match or not type_match:
            return None
        name = name_match.group(1).strip().lower()
        col_type = type_match.group(1).strip().lower()
        return name, col_type

    @staticmethod
    def _column_index(column_names: List[str], target: str) -> Optional[int]:
        target = target.lower()
        for idx, name in enumerate(column_names):
            if name == target:
                return idx
        return None

    def _skip_parameter_data(self, fh, parameter_types: List[str]) -> None:
        for param_type in parameter_types:
            if param_type == "string":
                length = self._read_int(fh)
                if length > 0:
                    self._consume_bytes(fh, length)
            elif param_type == "character":
                self._consume_bytes(fh, 1)
            elif param_type in {"short", "ushort"}:
                self._consume_bytes(fh, 2)
            elif param_type in {"long", "ulong", "int"}:
                self._consume_bytes(fh, 4)
            elif param_type in {"float"}:
                self._consume_bytes(fh, 4)
            elif param_type in {"double"}:
                self._consume_bytes(fh, 8)
            else:
                logging.warning("Unknown parameter type '%s' in PV catalog.", param_type)

    @staticmethod
    def _consume_bytes(fh, count: int) -> None:
        if count <= 0:
            return
        data = fh.read(count)
        if len(data) != count:
            raise EOFError("Unexpected end of PV catalog parameter data.")

    def _read_column_value(self, fh, column_type: str) -> Optional[str]:
        column_type = column_type.lower()
        if column_type == "string":
            return self._read_string(fh)
        if column_type == "character":
            data = fh.read(1)
            if len(data) != 1:
                raise EOFError("Unexpected end of PV catalog column data.")
            return data.decode("utf-8", errors="ignore")
        if column_type in {"short", "ushort"}:
            self._consume_bytes(fh, 2)
            return None
        if column_type in {"long", "ulong", "int"}:
            self._consume_bytes(fh, 4)
            return None
        if column_type == "float":
            self._consume_bytes(fh, 4)
            return None
        if column_type == "double":
            self._consume_bytes(fh, 8)
            return None
        logging.warning("Unknown column type '%s' in PV catalog.", column_type)
        return None

    def _build_entry(self, pv_name: str, description: str) -> PVEntry:
        alias_parts = re.split(r"[;,]", description)
        aliases = [pv_name.lower()]
        friendly_aliases: List[str] = []

        for part in alias_parts:
            raw = part.strip().lower()
            if not raw:
                continue

            cleaned = re.sub(r"^[^a-z0-9]+", "", raw).strip()
            if not cleaned:
                continue

            normalized = re.sub(r"^[0-9]+\s*", "", cleaned).strip()
            if not normalized:
                normalized = cleaned

            aliases.append(normalized)
            friendly_aliases.append(normalized)

            if normalized != cleaned:
                aliases.append(cleaned)
            if cleaned != raw:
                aliases.append(raw)

        label = friendly_aliases[0] if friendly_aliases else pv_name

        lower_pv = pv_name.lower()
        if lower_pv == "s-dcct:currentm":
            aliases.extend([
                "beam current",
                "storage ring current",
                "sr current",
                "sr beam current",
                "aps storage ring current",
                "beam curren",
            ])
        elif lower_pv == "bts:besocm:a:data:beam:qm":
            aliases.extend([
                "bts charge",
                "injector charge",
            ])

        # Add extra context-aware aliases for common PVs
        lower_pv = pv_name.lower()
        if lower_pv == "s-dcct:currentm":
            aliases.extend([
                "beam current",
                "storage ring current",
                "sr current",
                "sr beam current",
            ])
        elif lower_pv == "bts:besocm:a:data:beam:qm":
            aliases.extend([
                "bts charge",
                "injector charge",
            ])

        # Provide variants without punctuation
        additional: Iterable[str] = []
        for alias in list(aliases):
            stripped = alias.replace("-", " ").strip()
            if stripped and stripped not in aliases:
                aliases.append(stripped)
            colon_split = alias.replace(":", " ").strip()
            if colon_split and colon_split not in aliases:
                aliases.append(colon_split)
        aliases = list(dict.fromkeys(aliases))  # deduplicate preserving order

        return PVEntry(pv_name=pv_name, label=label, aliases=aliases)

    @staticmethod
    def _extend_query_tokens(tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        extended: List[str] = []
        index = 0
        length = len(tokens)
        while index < length:
            current = tokens[index]
            next_token = tokens[index + 1] if index + 1 < length else ""
            if (
                current
                and next_token
                and current[-1].isdigit()
                and next_token[0].isalpha()
                and any(ch.isdigit() for ch in next_token)
            ):
                extended.append(current + next_token)
                index += 2
            else:
                extended.append(current)
                index += 1
        return extended

    @classmethod
    def _build_alias_tokens(cls, alias_lower: str) -> Set[str]:
        tokens = set(re.findall(r"[a-z0-9:_-]+", alias_lower))
        for part in alias_lower.split(":"):
            part = part.strip()
            if part:
                tokens.add(part)
        enriched = set(tokens)
        for token in list(tokens):
            normalized = cls._normalize_token(token)
            if not normalized:
                continue
            enriched.add(normalized)
            enriched.update(cls._split_transition_fragments(normalized))
        return enriched

    @staticmethod
    def _split_transition_fragments(token: str) -> Set[str]:
        fragments: Set[str] = set()
        for pattern in (r"(?<=\d)(?=[a-z])", r"(?<=[a-z])(?=\d)"):
            for fragment in re.split(pattern, token):
                if fragment:
                    fragments.add(fragment)
        return fragments

    @staticmethod
    def _normalize_token(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value)

    @classmethod
    def _collect_matching_tokens(cls, tokens: List[str], alias_tokens: Set[str]) -> Set[str]:
        matched: Set[str] = set()
        for token in tokens:
            candidates = {token}
            normalized = cls._normalize_token(token)
            if normalized:
                candidates.add(normalized)
            if any(candidate in alias_tokens for candidate in candidates if candidate):
                matched.add(token)
        return matched

    @staticmethod
    def _filter_entries_by_context(entries: List[PVEntry], query_lower: str) -> List[PVEntry]:
        negative_keywords = ("setpoint", "set point", "command", "cmd", "trim")
        if any(keyword in query_lower for keyword in negative_keywords):
            return entries

        filtered = [
            entry
            for entry in entries
            if not any(
                keyword in entry.label.lower() or keyword in entry.pv_name.lower()
                for keyword in negative_keywords
            )
        ]
        return filtered

    def match_query(self, text: str) -> Optional[PVEntry]:
        """
        Match a query string against known aliases. Returns the best match.
        """
        lowered = text.lower()
        best_entry = None
        best_alias_length = 0
        for alias, entry in self.alias_map.items():
            if alias and alias in lowered:
                if len(alias) > best_alias_length:
                    best_alias_length = len(alias)
                    best_entry = entry
        return best_entry

    def match_all(self, text: str) -> List[PVEntry]:
        """
        Return all catalog entries whose aliases appear in the provided text.
        """
        lowered = text.lower()

        tokens = [
            token
            for token in re.findall(r"[a-z0-9:_-]+", lowered)
            if token and len(token) >= 2 and token not in QUERY_STOPWORDS
        ]
        filtered_tokens = [token for token in tokens if token not in TIME_KEYWORDS]
        effective_tokens = self._extend_query_tokens(filtered_tokens)
        required_token_set = set(effective_tokens)

        direct_matches: Dict[str, PVEntry] = {}
        token_matches: Dict[str, PVEntry] = {}
        token_hits: Dict[str, Set[str]] = {}
        for alias, entry in self.alias_map.items():
            alias_lower = alias.lower()
            if alias_lower and alias_lower in lowered:
                direct_matches.setdefault(entry.pv_name, entry)
                continue

            if not effective_tokens:
                continue

            alias_tokens = self._build_alias_tokens(alias_lower)
            matched_tokens = self._collect_matching_tokens(effective_tokens, alias_tokens)
            if not matched_tokens:
                continue

            hits = token_hits.setdefault(entry.pv_name, set())
            hits.update(matched_tokens)
            if hits.issuperset(required_token_set):
                token_matches.setdefault(entry.pv_name, entry)

        if direct_matches:
            return list(direct_matches.values())

        if token_matches:
            filtered = self._filter_entries_by_context(list(token_matches.values()), lowered)
            if filtered:
                return filtered
            return list(token_matches.values())

        return []

    def find_sr_corrector_pvs(self, query: str) -> List[PVEntry]:
        """
        Find SR corrector PVs using srcorr.json metadata.

        This method checks if the query is for SR correctors and uses the
        generate_srcorr_pvs_from_query function to get matching PVs.

        Returns:
            List of PVEntry objects for matching SR corrector PVs, or empty list if not applicable.
        """
        # Check if this is an SR corrector query
        is_srcorr = is_srcorr_query(query)
        logging.info(f"🔍 PV_Catalog: Checking if SR corrector query: {is_srcorr}")

        if not is_srcorr:
            logging.debug(f"   Not an SR corrector query, skipping srcorr.json")
            return []

        # Generate PVs from srcorr.json using fast-path
        logging.info(f"✓ PV_Catalog: Using srcorr.json fast-path for SR devices")
        pv_names = generate_srcorr_pvs_from_query(query)
        logging.info(f"📊 PV_Catalog: Generated {len(pv_names)} PV names from srcorr.json")

        if not pv_names:
            logging.warning(f"⚠️  PV_Catalog: SR query detected but no PVs generated")
            return []

        # Convert to PVEntry objects
        entries = []
        for pv_name in pv_names:
            entry = PVEntry(
                pv_name=pv_name,
                label=pv_name,
                aliases=[pv_name.lower()]
            )
            entries.append(entry)

        logging.info(f"✅ PV_Catalog: Returning {len(entries)} SR corrector PV entries")
        if entries:
            logging.info(f"   First 5 PVs: {[e.pv_name for e in entries[:5]]}")

        return entries

    def list_all_sr_corrector_pvs(self, include_suffixes: bool = False) -> Dict[str, Any]:
        """
        List all SR corrector PV names from srcorr.json.

        Args:
            include_suffixes: If True, include all suffix variations; if False, only base addresses

        Returns:
            Dictionary containing:
                - base_pvs: List of base PV addresses
                - suffixes: Dictionary of available suffixes (if include_suffixes=True)
                - total_count: Total number of PVs (base count or base × suffix count)
        """
        data = _load_srcorr_json()
        if not data:
            return {"base_pvs": [], "suffixes": {}, "total_count": 0}

        base_pvs = []
        all_suffixes = {}

        # Process all SR device types
        for type_key, (section_key, devices_key) in SRCORR_TYPE_MAP.items():
            section_data = data.get(section_key, {})
            devices = section_data.get(devices_key, [])

            # Collect suffixes from this section
            suffix_map = section_data.get("suffix", {})
            all_suffixes.update(suffix_map)

            # Generate base PVs for all sectors (1-40)
            for sector in range(1, 41):
                sec_str = f"{sector:02d}"
                for device in devices:
                    # Check sector constraints
                    allowed_sectors = device.get("sectors")
                    if allowed_sectors is not None and sector not in allowed_sectors:
                        continue

                    base = device.get("address_template", "").replace("${sector}", sec_str)
                    if base and base not in base_pvs:
                        base_pvs.append(base)

        # Sort for consistent ordering
        base_pvs.sort()

        result = {
            "base_pvs": base_pvs,
            "base_count": len(base_pvs),
        }

        if include_suffixes:
            result["suffixes"] = all_suffixes
            result["suffix_count"] = len(all_suffixes)
            result["total_count"] = len(base_pvs) * len(all_suffixes)
        else:
            result["total_count"] = len(base_pvs)

        return result

    def display_sr_corrector_pvs(self, include_suffixes: bool = False, max_display: int = 50):
        """
        Display SR corrector PV names to the console.

        Args:
            include_suffixes: If True, show suffix information
            max_display: Maximum number of base PVs to display (0 for all)
        """
        result = self.list_all_sr_corrector_pvs(include_suffixes=include_suffixes)

        print("\n" + "=" * 70)
        print("SR CORRECTOR PV NAMES FROM srcorr.json")
        print("=" * 70)

        base_pvs = result["base_pvs"]
        base_count = result["base_count"]

        if not base_pvs:
            print("No SR corrector PVs found in srcorr.json")
            return

        print(f"\nTotal base PV addresses: {base_count}")

        if include_suffixes:
            suffixes = result["suffixes"]
            suffix_count = result["suffix_count"]
            total_count = result["total_count"]
            print(f"Available suffixes: {suffix_count}")
            print(f"Total PVs (base × suffixes): {total_count}")

        # Display base PVs
        display_count = len(base_pvs) if max_display == 0 else min(max_display, len(base_pvs))
        print(f"\n{'All' if display_count == len(base_pvs) else f'First {display_count}'} Base PV Addresses:")
        print("-" * 70)

        for i, pv in enumerate(base_pvs[:display_count], 1):
            print(f"{i:4d}. {pv}")

        if display_count < len(base_pvs):
            print(f"\n... and {len(base_pvs) - display_count} more")

        # Display suffixes if requested
        if include_suffixes and result.get("suffixes"):
            print(f"\nAvailable Suffixes ({suffix_count} total):")
            print("-" * 70)
            for i, (name, suffix) in enumerate(sorted(result["suffixes"].items()), 1):
                print(f"{i:2d}. {name:30s} -> {suffix}")

            print(f"\nExample full PV names (first base PV with each suffix):")
            print("-" * 70)
            first_base = base_pvs[0]
            for i, (name, suffix) in enumerate(sorted(result["suffixes"].items())[:5], 1):
                full_pv = f"{first_base}{suffix}"
                print(f"{i}. {full_pv:50s} ({name})")

        print("\n" + "=" * 70)

    def get_default_entry(self) -> Optional[PVEntry]:
        """
        Returns the first entry in the catalog as a reasonable default.
        """
        return self.entries[0] if self.entries else None

    def get_entry_by_pv(self, pv_name: str) -> Optional[PVEntry]:
        pv_lower = pv_name.lower()
        for entry in self.entries:
            if entry.pv_name.lower() == pv_lower:
                return entry
        return None


_CATALOG_CACHE: Dict[str, Tuple[Optional[float], PVCatalog]] = {}
_CATALOG_LOCK = RLock()


def get_pv_catalog(filepath: Optional[str] = None, force_reload: bool = False) -> PVCatalog:
    """
    Returns a cached PVCatalog instance, refreshing when the underlying file changes.
    """
    path = filepath or PVCatalog.default_path()
    path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    with _CATALOG_LOCK:
        cached = _CATALOG_CACHE.get(path)
        if force_reload or cached is None or cached[0] != mtime:
            catalog = PVCatalog(filepath=path)
            _CATALOG_CACHE[path] = (mtime, catalog)
            return catalog
        return cached[1]


def clear_pv_catalog_cache(filepath: Optional[str] = None) -> None:
    """
    Clears the cached catalog (optionally for a specific file).
    """
    with _CATALOG_LOCK:
        if filepath is not None:
            path = filepath
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            _CATALOG_CACHE.pop(path, None)
        else:
            _CATALOG_CACHE.clear()
