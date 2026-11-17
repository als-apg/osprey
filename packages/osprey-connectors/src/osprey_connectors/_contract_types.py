"""The package's deliberate ``Any``: the values its public contracts leave untyped.

The package's mypy configuration refuses an explicit ``Any`` anywhere except in
this module. A value the package cannot type is still typed here, under a name
that says which contract leaves it open, and the rest of the package names that
alias. A new ``Any`` therefore needs a decision written here, not a stray
annotation, and ``git grep`` finds every open value by its alias.

The module imports only the standard library and nothing from the package.
"""

from __future__ import annotations

from typing import Any, TypeAlias

__all__ = ["ContractAny"]

ContractAny: TypeAlias = Any
"""A value a public contract leaves untyped.

These are the values a subclass's overridable ``ControlSystemConnector`` and
archiver methods take (``write_channel``'s ``value``, ``connect``'s config),
the channel value fields of ``ChannelValue``, ``ChannelWriteResult`` and
``ChannelMetadata`` that a connector fills from its control system, and the
untyped libraries' returns the package passes through. Each use states which
contract it is.
"""
