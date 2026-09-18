"""Loader for MATLAB ``.mat`` MML exports.

A ``.mat`` export carries the Accelerator Objects in a variable named ``AO``
(or ``ao``) and, optionally, the Accelerator Data in ``AD`` (or ``ad``). The
loader decodes scipy's MATLAB containers into plain JSON-serialisable Python
values and returns them raw; family normalisation runs afterwards.

A ``.mat`` saving the lattice carries the ring in ``THERING`` and no families.
It is a different kind of input: :func:`load_mat` recognises it and hands back
the path in ``lattice``, and :func:`load_lattice` builds the pyAT ring from it.

Only MAT-file versions 5 and 7 are readable. Version 7.3 files are HDF5
containers that scipy cannot open, so they are refused with a message naming
the MATLAB call that re-saves the file in a readable format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatlabFunction, mat_struct

from osprey.services.mml.loaders import LoadedInput

if TYPE_CHECKING:
    import at

__all__ = ["decode", "load_lattice", "load_mat"]

_HEADER_LENGTH = 128
_VERSION_OFFSET = 124
_ENDIAN_OFFSET = 126
_HDF5_MAJOR_VERSION = 2

_AO_NAMES = ("AO", "ao")
_AD_NAMES = ("AD", "ad")

#: The variable a lattice deck is saved under, by the exporter and by pyAT.
LATTICE_VARIABLE = "THERING"


def load_mat(path: Path | str) -> LoadedInput:
    """Read a MATLAB ``.mat`` MML export, or recognise a lattice deck.

    Args:
        path: The ``.mat`` file to read.

    Returns:
        The decoded input, with raw family bodies in ``ao`` and the decoded
        ``AD`` variable (or ``None``) in ``ad``. A deck instead yields an
        input whose ``lattice`` is the file and whose ``ao`` is empty; its
        ring is read on demand by :func:`load_lattice`.

    Raises:
        click.ClickException: The file is a v7.3 (HDF5) MAT-file, cannot be
            read, carries both a ring and an ``AO``, or carries neither.
    """
    path = Path(path)
    _refuse_hdf5(path)
    names = _variable_names(path)
    if LATTICE_VARIABLE in names:
        if any(name in names for name in _AO_NAMES):
            raise click.ClickException(
                f"MAT-file {path} carries both {LATTICE_VARIABLE} and an AO variable; "
                "save the lattice and the Accelerator Objects as separate files."
            )
        return LoadedInput(
            ao={}, ad=None, export=None, system_keyed=False, source=path, lattice=path
        )
    try:
        variables = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    except Exception as exc:  # scipy raises several unrelated types on bad input
        raise click.ClickException(f"Cannot read MAT-file {path}: {exc}") from exc

    ao_raw = _first_present(variables, _AO_NAMES)
    if ao_raw is None:
        raise click.ClickException(f"MAT-file {path} has no AO or ao variable.")
    ao = decode(ao_raw)
    if not isinstance(ao, dict):
        raise click.ClickException(f"The AO variable in MAT-file {path} is not a struct.")

    ad_raw = _first_present(variables, _AD_NAMES)
    ad = None if ad_raw is None else decode(ad_raw)
    if ad is not None and not isinstance(ad, dict):
        raise click.ClickException(f"The AD variable in MAT-file {path} is not a struct.")

    return LoadedInput(ao=ao, ad=ad, export=None, system_keyed=False, source=path)


def load_lattice(path: Path | str) -> at.Lattice:
    """Build the pyAT ring held in a lattice deck.

    Every element of the saved ring is kept, so a ``RingParam`` stays in place
    as a tagged marker and the MATLAB index of each element is its position in
    the returned ring.

    Args:
        path: The ``.mat`` file whose ``THERING`` holds the ring.

    Returns:
        The ring, with its elements in saved order.

    Raises:
        click.ClickException: The file carries no readable ring; the message
            names the file.
    """
    import at

    path = Path(path)
    try:
        return at.load_mat(str(path), use=LATTICE_VARIABLE, keep_all=True)
    except Exception as exc:  # pyAT raises several unrelated types on a bad ring
        raise click.ClickException(f"Cannot read the lattice in {path}: {exc}") from exc


def decode(value: Any) -> Any:
    """Convert a value returned by ``loadmat`` into plain Python values.

    Structs become dicts, function handles decode through their wrapped
    struct, cell arrays become (nested) lists, char arrays become strings,
    numeric arrays become lists, and numpy scalars become Python scalars.

    Args:
        value: A value from ``loadmat(struct_as_record=False, squeeze_me=True)``.

    Returns:
        A JSON-serialisable combination of dicts, lists, strings and numbers.
    """
    if isinstance(value, mat_struct):
        return {name: decode(getattr(value, name)) for name in value._fieldnames}
    # MatlabFunction subclasses ndarray, so it must be matched before the array arms.
    if isinstance(value, MatlabFunction):
        return decode(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            items = [decode(item) for item in value.ravel()]
            return _reshape(items, value.shape)
        if value.dtype.kind == "U":
            if value.size == 0:
                return ""
            if value.ndim == 0:
                return str(value.item()).rstrip()
            return [str(row).rstrip() for row in value.ravel()]
        if value.size == 0:
            return []
        return value.tolist()
    if isinstance(value, (np.str_, np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _reshape(items: list, shape: tuple[int, ...]) -> Any:
    """Arrange a flat, C-ordered list into nested lists of ``shape``."""
    if not shape:
        return items[0]
    if len(shape) == 1:
        return items
    step = len(items) // shape[0] if shape[0] else 0
    return [_reshape(items[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _variable_names(path: Path) -> set[str]:
    """The top-level variable names of a MAT-file, read without decoding it."""
    try:
        return {name for name, _shape, _dtype in whosmat(str(path))}
    except Exception as exc:  # scipy raises several unrelated types on bad input
        raise click.ClickException(f"Cannot read MAT-file {path}: {exc}") from exc


def _first_present(variables: dict, names: tuple[str, ...]) -> Any:
    """The value of the first variable in ``names`` that ``loadmat`` returned."""
    for name in names:
        if name in variables:
            return variables[name]
    return None


def _refuse_hdf5(path: Path) -> None:
    """Raise when the MAT-file header declares version 7.3 (HDF5)."""
    try:
        with path.open("rb") as handle:
            header = handle.read(_HEADER_LENGTH)
    except OSError as exc:
        raise click.ClickException(f"Cannot read MAT-file {path}: {exc}") from exc
    if len(header) < _HEADER_LENGTH:
        return
    version = header[_VERSION_OFFSET:_ENDIAN_OFFSET]
    major = version[1] if header[_ENDIAN_OFFSET:_HEADER_LENGTH] == b"IM" else version[0]
    if major == _HDF5_MAJOR_VERSION:
        raise click.ClickException(
            f"MAT-file {path} is version 7.3 (HDF5), which cannot be read. "
            "Re-save it in MATLAB with save('-v7', ...) and try again."
        )
