"""Files an app template's ``data/`` tree takes from another template's.

Two app templates can need the same packaged content — the ARIEL-only template
documents the same demo machine and the same logbook incidents the
control-assistant template simulates. Keeping a copy in each lets them drift
apart, so the source holds one: the template that owns the content keeps it,
and the other declares what it takes in a ``shared_data.yml`` beside its
``data/`` directory::

    - from: control_assistant          # the template that owns the files
      source: simulation/scenarios     # a path in that template's data/
      target: logbook_seed             # where it lands in this template's data/
      include: ["*/logbook.json"]      # optional: globs under a directory source

Everything that reads a template's packaged ``data/`` — ``osprey init``
materializing a profile and ``osprey scaffold pull`` — reads it through
:func:`shared_data_files` as well, so a shared file is indistinguishable from
one the template ships itself.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import yaml

#: The declaration file, beside an app template's ``data/`` directory.
SHARED_DATA_FILENAME = "shared_data.yml"

_ENTRY_KEYS = frozenset({"from", "source", "target", "include"})


def shared_data_files(app_root: Path) -> dict[str, Path]:
    """The files ``app_root``'s ``data/`` tree takes from other templates.

    Args:
        app_root: The app template directory (``templates/apps/<name>``).

    Returns:
        Data-relative POSIX path -> the packaged file it is copied from, sorted
        by path. Empty when the template declares nothing.

    Raises:
        ValueError: If the declaration is malformed, names a template or path
            that does not exist, or lands a file where the template ships one of
            its own -- the duplicate this file exists to prevent.
    """
    manifest = app_root / SHARED_DATA_FILENAME
    if not manifest.is_file():
        return {}
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{manifest}: must be a list of entries")

    files: dict[str, Path] = {}
    for entry in raw:
        if not isinstance(entry, dict) or not {"from", "source", "target"} <= set(entry):
            raise ValueError(f"{manifest}: each entry needs 'from', 'source' and 'target'")
        unknown = sorted(set(entry) - _ENTRY_KEYS)
        if unknown:
            raise ValueError(f"{manifest}: unknown keys {unknown}")
        donor = app_root.parent / str(entry["from"]) / "data"
        source = donor / str(entry["source"])
        target = PurePosixPath(str(entry["target"]))
        if source.is_file():
            files[target.as_posix()] = source
        elif source.is_dir():
            patterns = entry.get("include") or ["**/*"]
            for pattern in patterns:
                for path in sorted(source.glob(str(pattern))):
                    if path.is_file():
                        relative = path.relative_to(source).as_posix()
                        files[(target / relative).as_posix()] = path
        else:
            raise ValueError(f"{manifest}: {source} does not exist")

    own = app_root / "data"
    clashes = sorted(rel for rel in files if (own / rel).exists())
    if clashes:
        raise ValueError(f"{manifest}: {clashes} are shipped by this template too; keep one copy")
    return dict(sorted(files.items()))
